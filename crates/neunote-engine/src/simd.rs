//! Runtime-dispatched SIMD for the kernels scalar code leaves latency-bound.
//!
//! Four of them, and no more: converting a weight row out of f16, dotting two
//! vectors, adding a scaled vector into another, and the row update behind a
//! matmul whose columns are wide enough to vectorise across. Each call covers a
//! whole row or a whole vector, so the dispatch -- a cached load and a branch --
//! is nothing against the arithmetic it guards.
//!
//! Every other module keeps its own `#![forbid(unsafe_code)]`; this one
//! cannot: stable Rust has no other way to name an intrinsic, so every
//! `unsafe` block below carries the reason it holds.
//!
//! Dispatch is per process, and the answer can differ between machines -- an
//! AVX2 host and a baseline one accumulate in different orders. The reference
//! takes the same freedom: its ggml build vectorises too, and the dot products
//! are exactly the sums whose order was never fixed.

use half::f16;

#[cfg(target_arch = "x86_64")]
use std::sync::LazyLock;

/// The kernels this CPU can run, decided once.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Isa {
    Scalar,
    Avx2,
}

#[cfg(target_arch = "x86_64")]
static ISA: LazyLock<Isa> = LazyLock::new(detect);

#[cfg(target_arch = "x86_64")]
fn detect() -> Isa {
    // FMA and F16C are what the dot product and the conversion want. Every
    // CPU with AVX2 has had both since 2013, but they are checked rather
    // than assumed, because assuming is how a build traps on an older host.
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        return Isa::Avx2;
    }
    Isa::Scalar
}

/// Widen an f16 row to f32.
///
/// Every f16 value is representable in f32, so both paths are exact; only the
/// speed differs.
pub(crate) fn f16_to_f32(out: &mut [f32], values: &[f16]) {
    debug_assert_eq!(out.len(), values.len());

    #[cfg(target_arch = "x86_64")]
    if *ISA == Isa::Avx2 {
        // SAFETY: `detect` saw the features this asks for, and the two slices
        // have one length, so neither pointer walk leaves its allocation.
        unsafe { avx2_f16_to_f32(out, values) };
        return;
    }

    for (slot, value) in out.iter_mut().zip(values) {
        *slot = value.to_f32();
    }
}

/// `sum_i a[i] * b[i]`.
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    if *ISA == Isa::Avx2 {
        // SAFETY: as `f16_to_f32`.
        return unsafe { avx2_dot_f32(a, b) };
    }

    let mut total = 0.0f32;
    for (left, right) in a.iter().zip(b) {
        total += *left * *right;
    }
    total
}

/// `sum_i weights[i] * x[i]`, converting as it goes.
///
/// This is the decode path: one output element, a whole weight row, and in
/// scalar code the conversion is half the work.
pub(crate) fn dot_f16(weights: &[f16], x: &[f32]) -> f32 {
    debug_assert_eq!(weights.len(), x.len());

    #[cfg(target_arch = "x86_64")]
    if *ISA == Isa::Avx2 {
        // SAFETY: as `f16_to_f32`.
        return unsafe { avx2_dot_f16(weights, x) };
    }

    let mut total = 0.0f32;
    for (weight, value) in weights.iter().zip(x) {
        total += weight.to_f32() * *value;
    }
    total
}

/// `y[i] += scale * x[i]`.
pub(crate) fn axpy_scale(y: &mut [f32], scale: f32, x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());

    #[cfg(target_arch = "x86_64")]
    if *ISA == Isa::Avx2 {
        // SAFETY: as `f16_to_f32`.
        unsafe { avx2_axpy_scale(y, scale, x) };
        return;
    }

    for (slot, value) in y.iter_mut().zip(x) {
        *slot += scale * *value;
    }
}

/// `y[i] += sum_k weight[k] * x[k * stride + i]`: one weight row over a run of
/// activation columns.
///
/// The elements go in blocks, and within a block the reduction runs to
/// completion for all of them before the next block starts, so each element
/// accumulates `weight[0]`, `weight[1]`, ... in order. That is the scalar
/// expression with the element loop vectorised, not a reassociation of it,
/// which is what keeps a wide matmul bit-for-bit where it was. The optimiser
/// does not get there on its own: the slice arithmetic wrapped around the
/// multiply-accumulate keeps it scalar.
///
/// `stride` is the activation's column count, and `x` carries `weight.len()`
/// rows of it.
pub(crate) fn axpy_row(y: &mut [f32], weight: &[f32], x: &[f32], stride: usize) {
    assert!(
        stride >= y.len(),
        "an activation row cannot be narrower than the row it fills"
    );
    assert!(
        x.len() >= weight.len().saturating_mul(stride),
        "the activation has to hold every weight row"
    );

    #[cfg(target_arch = "x86_64")]
    if *ISA == Isa::Avx2 {
        // SAFETY: as `f16_to_f32`; the two assertions above are the walks the
        // kernel takes -- reads through `weight.len()` rows of `stride`, writes
        // through `y` -- proved against the slices they run in.
        unsafe { avx2_axpy_row(y, weight, x, stride) };
        return;
    }

    for (k, scale) in weight.iter().enumerate() {
        let row = &x[k * stride..k * stride + y.len()];
        for (slot, value) in y.iter_mut().zip(row) {
            *slot += *scale * *value;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn avx2_f16_to_f32(out: &mut [f32], values: &[f16]) {
    use std::arch::x86_64::*;

    let end = out.len().min(values.len());
    let src = values.as_ptr().cast::<u16>();
    let mut at = 0;
    // SAFETY: the loop condition keeps `at + 8` at or below `end`, which is no
    // greater than either slice's length, so neither walk leaves its
    // allocation; `loadu` takes any alignment, and `f16` is `repr(transparent)`
    // over `u16`, so the eight words are the bits `cvtph_ps` wants.
    unsafe {
        while at + 8 <= end {
            let packed = _mm_loadu_si128(src.add(at).cast());
            _mm256_storeu_ps(out.as_mut_ptr().add(at), _mm256_cvtph_ps(packed));
            at += 8;
        }
    }
    while at < end {
        out[at] = values[at].to_f32();
        at += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn avx2_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    // Four accumulators over eight values each: thirty-two elements of the sum
    // are in flight at once, which is what lets them run while the previous
    // round trips settle, and still four registers short of the file.
    let end = a.len().min(b.len());

    // SAFETY: `at + 32 <= end`, and `end` is no greater than either slice's
    // length, so every load and pointer walk below stays inside both slices;
    // `loadu` accepts any alignment.
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut at = 0;

        while at + 32 <= end {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(at)),
                _mm256_loadu_ps(b_ptr.add(at)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(at + 8)),
                _mm256_loadu_ps(b_ptr.add(at + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(at + 16)),
                _mm256_loadu_ps(b_ptr.add(at + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(at + 24)),
                _mm256_loadu_ps(b_ptr.add(at + 24)),
                acc3,
            );
            at += 32;
        }

        let mut total = hsum256(acc0) + hsum256(acc1) + hsum256(acc2) + hsum256(acc3);
        while at < end {
            total += a[at] * b[at];
            at += 1;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma,f16c")]
unsafe fn avx2_dot_f16(weights: &[f16], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let src = weights.as_ptr().cast::<u16>();
    let end = weights.len().min(x.len());
    // SAFETY: `at + 32 <= end`, and `end` is no greater than either slice's
    // length, so every load stays inside both, and `cvtph_ps` turns each word
    // it is handed into an exact f32.
    unsafe {
        let x_ptr = x.as_ptr();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut at = 0;

        while at + 32 <= end {
            let words = _mm_loadu_si128(src.add(at).cast());
            acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(words), _mm256_loadu_ps(x_ptr.add(at)), acc0);

            let words = _mm_loadu_si128(src.add(at + 8).cast());
            acc1 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(words),
                _mm256_loadu_ps(x_ptr.add(at + 8)),
                acc1,
            );

            let words = _mm_loadu_si128(src.add(at + 16).cast());
            acc2 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(words),
                _mm256_loadu_ps(x_ptr.add(at + 16)),
                acc2,
            );

            let words = _mm_loadu_si128(src.add(at + 24).cast());
            acc3 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(words),
                _mm256_loadu_ps(x_ptr.add(at + 24)),
                acc3,
            );

            at += 32;
        }

        let mut total = hsum256(acc0) + hsum256(acc1) + hsum256(acc2) + hsum256(acc3);
        while at < end {
            total += weights[at].to_f32() * x[at];
            at += 1;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn avx2_axpy_scale(y: &mut [f32], scale: f32, x: &[f32]) {
    use std::arch::x86_64::*;

    let end = y.len().min(x.len());

    // SAFETY: every index is below `end`, which is no greater than either
    // slice's length.
    unsafe {
        let lanes = _mm256_set1_ps(scale);
        let x_ptr = x.as_ptr();
        let mut at = 0;
        while at + 8 <= end {
            let scaled = _mm256_mul_ps(lanes, _mm256_loadu_ps(x_ptr.add(at)));
            let sum = _mm256_add_ps(_mm256_loadu_ps(y.as_ptr().add(at)), scaled);
            _mm256_storeu_ps(y.as_mut_ptr().add(at), sum);
            at += 8;
        }
        while at < end {
            y[at] += scale * x[at];
            at += 1;
        }
    }
}

/// One weight row added into a run of columns, the reduction in order for each
/// element. Thirty-two columns at a time: four accumulators wide is enough for
/// one round of the reduction to cover the latency of the last, and still
/// leaves the registers for the row it is reading.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
unsafe fn avx2_axpy_row(y: &mut [f32], weight: &[f32], x: &[f32], stride: usize) {
    use std::arch::x86_64::*;

    // SAFETY: the wrapper's assertions give `base + 32 <= y.len() <= stride`
    // and `weight.len() * stride <= x.len()`, so `k * stride + base + 32` is
    // inside `x` for every `k` below. `loadu` and `storeu` need no alignment.
    let mut base = 0;
    unsafe {
        while base + 32 <= y.len() {
            let mut acc0 = _mm256_loadu_ps(y.as_ptr().add(base));
            let mut acc1 = _mm256_loadu_ps(y.as_ptr().add(base + 8));
            let mut acc2 = _mm256_loadu_ps(y.as_ptr().add(base + 16));
            let mut acc3 = _mm256_loadu_ps(y.as_ptr().add(base + 24));

            for (k, scale) in weight.iter().enumerate() {
                let lanes = _mm256_set1_ps(*scale);
                let row = x.as_ptr().add(k * stride + base);
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(lanes, _mm256_loadu_ps(row)));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(lanes, _mm256_loadu_ps(row.add(8))));
                acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(lanes, _mm256_loadu_ps(row.add(16))));
                acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(lanes, _mm256_loadu_ps(row.add(24))));
            }

            _mm256_storeu_ps(y.as_mut_ptr().add(base), acc0);
            _mm256_storeu_ps(y.as_mut_ptr().add(base + 8), acc1);
            _mm256_storeu_ps(y.as_mut_ptr().add(base + 16), acc2);
            _mm256_storeu_ps(y.as_mut_ptr().add(base + 24), acc3);
            base += 32;
        }

        while base + 8 <= y.len() {
            let mut acc = _mm256_loadu_ps(y.as_ptr().add(base));
            for (k, scale) in weight.iter().enumerate() {
                let lanes = _mm256_set1_ps(*scale);
                let row = x.as_ptr().add(k * stride + base);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(lanes, _mm256_loadu_ps(row)));
            }
            _mm256_storeu_ps(y.as_mut_ptr().add(base), acc);
            base += 8;
        }
    }

    for (k, scale) in weight.iter().enumerate() {
        let row = &x[k * stride + base..k * stride + y.len()];
        for (at, value) in row.iter().enumerate() {
            y[base + at] += *scale * *value;
        }
    }
}

/// The four lanes of an accumulator, added pairwise down to one.
///
/// The shuffle swaps neighbouring lanes (`[1, 0, 3, 2]`), so the first add
/// pairs them up and `movehl` brings the upper pair down to finish. Swapping
/// the pairs instead leaves a sum that still looks like a sum -- twice the even
/// lanes -- which is the kind of wrong that reaches the logits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn hsum256(vector: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let low = _mm256_castps256_ps128(vector);
    let high = _mm256_extractf128_ps(vector, 1);
    let sum = _mm_add_ps(low, high);
    let pairs = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, 0b10_11_00_01));
    _mm_cvtss_f32(_mm_add_ps(pairs, _mm_movehl_ps(pairs, pairs)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic values in a range where a wrong lane or a wrong scale is
    /// visible rather than buried in rounding.
    fn series(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let mixed = seed
                    .wrapping_mul(index as u32 + 1)
                    .wrapping_mul(2_654_435_761);
                (mixed >> 8) as f32 / 8_388_608.0 - 1.0
            })
            .collect()
    }

    fn samples() -> Vec<usize> {
        [
            0, 1, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 100, 257, 1024,
        ]
        .to_vec()
    }

    #[test]
    fn the_widening_is_the_bit_pattern_half_would_give() {
        for len in samples() {
            let wide = series(len, 7);
            let values: Vec<f16> = wide.iter().copied().map(f16::from_f32).collect();
            let mut got = vec![0.0f32; len];
            f16_to_f32(&mut got, &values);

            for (got, want) in got.iter().zip(&values) {
                assert_eq!(
                    got.to_bits(),
                    want.to_f32().to_bits(),
                    "len {len}: {got} != {want}"
                );
            }
        }
    }

    #[test]
    fn a_dot_agrees_with_the_sum_it_stands_for() {
        for len in samples() {
            let a = series(len, 3);
            let b = series(len, 91);

            let mut want = 0.0f32;
            for (left, right) in a.iter().zip(&b) {
                want += *left * *right;
            }

            let got = dot_f32(&a, &b);
            let scale = want.abs().max(1e-3);
            assert!(
                (got - want).abs() <= scale * 1e-5,
                "len {len}: {got} against {want}"
            );
        }
    }

    #[test]
    fn a_dot_over_f16_agrees_with_the_sum_it_stands_for() {
        for len in samples() {
            let weights: Vec<f16> = series(len, 17).into_iter().map(f16::from_f32).collect();
            let x = series(len, 203);

            let mut want = 0.0f32;
            for (weight, value) in weights.iter().zip(&x) {
                want += weight.to_f32() * *value;
            }

            let got = dot_f16(&weights, &x);
            let scale = want.abs().max(1e-3);
            assert!(
                (got - want).abs() <= scale * 1e-5,
                "len {len}: {got} against {want}"
            );
        }
    }

    #[test]
    fn a_scaled_addition_is_bit_for_bit_the_scalar_expression() {
        for len in samples() {
            let x = series(len, 405);
            let mut got = series(len, 606);
            let mut want = got.clone();
            let scale = 0.375f32;

            axpy_scale(&mut got, scale, &x);
            for (slot, value) in want.iter_mut().zip(&x) {
                *slot += scale * *value;
            }

            for (got, want) in got.iter().zip(&want) {
                assert_eq!(got.to_bits(), want.to_bits(), "len {len}: {got} != {want}");
            }
        }
    }

    /// The wide kernel must not reorder anything: one accumulator per element,
    /// the reduction in weight order, so the vector and scalar forms agree to
    /// the bit. A row update that only gets the sum roughly right still has the
    /// right shape, and the ladder cannot tell which lane was doubled.
    #[test]
    fn a_row_update_is_bit_for_bit_the_scalar_expression() {
        for columns in [8, 15, 16, 31, 32, 33, 64, 65, 100] {
            for reduction in [1, 2, 5, 32, 33, 128] {
                let weight = series(reduction, 11);
                let x = series(reduction * columns, 29);
                let mut got = series(columns, 71);
                let mut want = got.clone();

                axpy_row(&mut got, &weight, &x, columns);
                for (k, scale) in weight.iter().enumerate() {
                    let row = &x[k * columns..(k + 1) * columns];
                    for (slot, value) in want.iter_mut().zip(row) {
                        *slot += *scale * *value;
                    }
                }

                for (got, want) in got.iter().zip(&want) {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{columns} columns x {reduction} reduction: {got} != {want}"
                    );
                }
            }
        }
    }
}
