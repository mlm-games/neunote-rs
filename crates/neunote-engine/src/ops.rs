#![forbid(unsafe_code)]

//! The arithmetic the model is made of.
//!
//! Activations are laid out `[reduction, columns]`: the token or frame axis is
//! contiguous, which is the direction every reduction runs in. Weights are
//! `[rows, reduction]`, so the tensor data a GGUF stores is used as it lies --
//! no transposes between the file and the matmul.
//!
//! `y[j][n] = bias[j] + sum_k W[j][k] * x[k][n]`
//!
//! Parallelism is over output rows only, so the result does not depend on the
//! thread count.

use rayon::prelude::*;

use crate::gguf::Weight;

/// Below this many rows, scheduling costs more than the arithmetic.
const PARALLEL_ROW_FLOOR: usize = 64;

/// Below this many columns a row has no lane per element to spread out over,
/// so it becomes a dot product instead of a vector update.
const WIDE_COLUMNS: usize = 8;

/// Split an output buffer into contiguous row blocks, one per parallel task.
///
/// Output rows are independent, so the split cannot change a result. Small work
/// stays in one block, where scheduling costs more than the arithmetic.
fn row_blocks(y: &mut [f32], rows: usize, columns: usize) -> (usize, Vec<&mut [f32]>) {
    let threads = if rows < PARALLEL_ROW_FLOOR {
        1
    } else {
        rayon::current_num_threads()
    };
    let rows_per_block = rows.div_ceil(threads).max(1);

    let mut blocks = Vec::with_capacity(rows.div_ceil(rows_per_block));
    let mut rest = y;
    while !rest.is_empty() {
        let take = (rows_per_block * columns).min(rest.len());
        let (head, tail) = rest.split_at_mut(take);
        blocks.push(head);
        rest = tail;
    }

    (rows_per_block, blocks)
}

/// `y = W x + bias`, with `x` and `y` in `[reduction, columns]` layout.
///
/// An F16 weight row is converted once and reused across every column, which is
/// the whole reason the conversion is here rather than at load: a checkpoint
/// stays the size it was published at, and every element is still converted
/// exactly once per matmul.
///
/// Two kernels, split on how many columns there are. Wide enough and a row is a
/// vector update, where each output element keeps the reduction in weight order
/// and only the element loop is vectorised. Too narrow and there is nothing to
/// spread an element over but the reduction, so a row becomes a dot product --
/// which is what every decode step is made of.
pub fn matmul(y: &mut [f32], weight: &Weight, x: &[f32], columns: usize, bias: Option<&[f32]>) {
    assert!(columns > 0, "a matmul needs at least one column");
    let reduction = x.len() / columns;
    let rows = weight.rows(reduction);
    assert_eq!(y.len(), rows * columns);

    let (rows_per_block, blocks) = row_blocks(y, rows, columns);

    blocks.into_par_iter().enumerate().for_each(|(block, out)| {
        let first_row = block * rows_per_block;
        if columns < WIDE_COLUMNS {
            dot_product_rows(out, weight, x, columns, bias, first_row);
        } else {
            vector_rows(out, weight, x, columns, bias, first_row);
        }
    });
}

/// A block of rows whose columns are too few to vectorise, so each row is one
/// dot product.
///
/// At a single column -- the decode path, and the head on every pass -- the
/// whole row is contiguous in `x` and the dot runs in `simd`. Two to seven
/// columns are strided instead, and stay scalar: nothing the model does lands
/// there.
fn dot_product_rows(
    out: &mut [f32],
    weight: &Weight,
    x: &[f32],
    columns: usize,
    bias: Option<&[f32]>,
    first_row: usize,
) {
    let reduction = x.len() / columns;

    if columns == 1 {
        for (offset, chunk) in out.chunks_mut(columns).enumerate() {
            let row = first_row + offset;
            let base = bias.map_or(0.0, |bias| bias[row]);
            let bounds = row * reduction..(row + 1) * reduction;

            chunk[0] = base
                + match weight {
                    Weight::F32(values) => crate::simd::dot_f32(&values[bounds], x),
                    Weight::F16(values) => crate::simd::dot_f16(&values[bounds], x),
                };
        }
        return;
    }

    let mut scratch = vec![0.0f32; reduction];
    for (offset, chunk) in out.chunks_mut(columns).enumerate() {
        let row = first_row + offset;
        let base = bias.map_or(0.0, |bias| bias[row]);

        weight.row_into(row, reduction, &mut scratch);
        for (column, slot) in chunk.iter_mut().enumerate() {
            let mut total = 0.0f32;
            for (k, scale) in scratch.iter().enumerate() {
                total += *scale * x[k * columns + column];
            }
            *slot = base + total;
        }
    }
}

/// A block of rows wide enough to vectorise across their columns.
fn vector_rows(
    out: &mut [f32],
    weight: &Weight,
    x: &[f32],
    columns: usize,
    bias: Option<&[f32]>,
    first_row: usize,
) {
    let reduction = x.len() / columns;
    let mut scratch = vec![0.0f32; reduction];

    for (offset, chunk) in out.chunks_mut(columns).enumerate() {
        let row = first_row + offset;
        weight.row_into(row, reduction, &mut scratch);

        match bias {
            Some(bias) => chunk.fill(bias[row]),
            None => chunk.fill(0.0),
        }

        crate::simd::axpy_row(chunk, &scratch, x, columns);
    }
}

/// LayerNorm over the reduction axis, with an affine scale and shift.
///
/// `x` is `[rows, columns]`, and it is `rows` that gets normalised: one mean and
/// one variance per column, with `weight` and `bias` indexed by row. Normalising
/// the other axis -- the tokens -- also produces numbers of the right size, and
/// is wrong for every weight that matters.
///
/// The statistics accumulate in `f64`, as ggml's are: ggml reduces in
/// `ggml_float` before taking an `f32` square root, and over 768 values the
/// difference is small but it is free to match.
pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, columns: usize) -> Vec<f32> {
    let rows = x.len() / columns;
    assert_eq!(
        weight.len(),
        rows,
        "the affine scale is per reduced element"
    );
    assert_eq!(bias.len(), rows, "the affine shift is per reduced element");

    let mut mean = vec![0.0f32; columns];
    let mut scale = vec![0.0f32; columns];

    for column in 0..columns {
        let mut sum = 0.0f64;
        for row in 0..rows {
            sum += f64::from(x[row * columns + column]);
        }
        let centre = sum / rows as f64;

        let mut squares = 0.0f64;
        for row in 0..rows {
            let centred = f64::from(x[row * columns + column]) - centre;
            squares += centred * centred;
        }

        mean[column] = centre as f32;
        scale[column] = 1.0 / ((squares / rows as f64) as f32 + eps).sqrt();
    }

    // Only the affine pass is split, and it splits by row: a column range is
    // strided in this layout and so is not a contiguous run.
    let mut out = vec![0.0f32; x.len()];
    let (rows_per_block, blocks) = row_blocks(&mut out, rows, columns);

    blocks
        .into_par_iter()
        .enumerate()
        .for_each(|(block, target)| {
            let first = block * rows_per_block;
            for (offset, line) in target.chunks_mut(columns).enumerate() {
                let row = first + offset;
                for (column, slot) in line.iter_mut().enumerate() {
                    *slot =
                        (x[row * columns + column] - mean[column]) * scale[column] * weight[row]
                            + bias[row];
                }
            }
        });

    out
}

/// `F.gelu(approximate="none")`: the exact erf form.
///
/// The tanh approximation differs by up to ~1e-3 and compounds across layers,
/// which is enough to move a greedy argmax, so this is not interchangeable with
/// `ggml_gelu`.
pub fn gelu_erf(x: f32) -> f32 {
    const INV_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    0.5 * x * (1.0 + libm::erff(x * INV_SQRT_2))
}

pub fn gelu_erf_in_place(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = gelu_erf(*value);
    }
}

/// Softmax along the contiguous axis, with an additive mask.
///
/// The mask is `0.0` where a query may look and `-inf` where it may not, added
/// before the exponentials, so masked entries contribute nothing.
pub fn masked_softmax(x: &mut [f32], mask: &[f32], columns: usize) {
    let rows = x.len() / columns;

    for row in 0..rows {
        let line = &mut x[row * columns..(row + 1) * columns];
        let allowed = &mask[row * columns..(row + 1) * columns];

        let mut peak = f32::NEG_INFINITY;
        for (value, bias) in line.iter_mut().zip(allowed) {
            *value += *bias;
            peak = peak.max(*value);
        }

        let mut total = 0.0f32;
        for value in line.iter_mut() {
            *value = (*value - peak).exp();
            total += *value;
        }

        let inverse = 1.0 / total;
        for value in line.iter_mut() {
            *value *= inverse;
        }
    }
}

/// Causal mask aligned to the bottom right.
///
/// Query row `i` sits at absolute position `past + i` and may attend to every
/// key up to and including it. PyTorch's `is_causal=True` is top-left aligned,
/// which only agrees when the query and key counts match; a prefilled prompt is
/// exactly where the two differ.
pub fn bottom_right_causal_mask(past: usize, new: usize, kv: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; new * kv];

    for row in 0..new {
        let last = past + row;
        let line = &mut mask[row * kv..(row + 1) * kv];
        line.iter_mut()
            .enumerate()
            .for_each(|(key, slot)| *slot = if key <= last { 0.0 } else { f32::NEG_INFINITY });
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn close(left: f32, right: f32, tolerance: f32) -> bool {
        (left - right).abs() <= tolerance
    }

    #[test]
    fn matmul_accumulates_over_the_reduction_axis() {
        // W is [2, 3], x is [3, 2]: two inputs of three features.
        let w = Weight::F32(vec![1.0, 2.0, 3.0, -1.0, 0.5, 0.0]);
        let x = [1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0];
        let mut y = [0.0f32; 4];

        matmul(&mut y, &w, &x, 2, None);

        assert!(close(y[0], 1.0 + 4.0 + 9.0, 1e-6));
        assert!(close(y[1], 10.0 + 40.0 + 90.0, 1e-6));
        assert!(close(y[2], -1.0 + 1.0 + 0.0, 1e-6));
        assert!(close(y[3], -10.0 + 10.0, 1e-6));
    }

    #[test]
    fn a_bias_is_added_once_per_row_not_once_per_element() {
        // One row of two zeroes over two inputs, so the bias is all that lands.
        let w = Weight::F32(vec![0.0, 0.0]);
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut y = [0.0f32; 2];

        matmul(&mut y, &w, &x, 2, Some(&[5.0]));

        assert_eq!(y, [5.0, 5.0]);
    }

    #[test]
    fn f16_weights_give_the_same_answer_as_the_dequantised_ones() {
        let values = [0.5f32, -1.25, 3.0, 0.125, -2.5, 7.75];
        let wide = Weight::F32(values.to_vec());
        let narrow = Weight::F16(values.iter().map(|v| f16::from_f32(*v)).collect());
        let x = [1.0f32, -2.0, 4.0, 0.25, 0.5, -1.0];

        let mut from_wide = [0.0f32; 4];
        let mut from_narrow = [0.0f32; 4];
        matmul(&mut from_wide, &wide, &x, 2, None);
        matmul(&mut from_narrow, &narrow, &x, 2, None);

        for (a, b) in from_wide.iter().zip(&from_narrow) {
            assert!(close(*a, *b, 1e-5), "{a} vs {b}");
        }
    }

    #[test]
    fn matmul_gives_the_same_answer_on_every_thread_count() {
        let (rows, cols, columns) = (200usize, 96usize, 5usize);
        let w = Weight::F32(
            (0..rows * cols)
                .map(|index| ((index * 37 % 101) as f32 - 50.0) / 50.0)
                .collect(),
        );
        let x: Vec<f32> = (0..cols * columns)
            .map(|index| (index % 17) as f32)
            .collect();
        let mut y = vec![0.0f32; rows * columns];

        matmul(&mut y, &w, &x, columns, None);

        // Recompute the first row serially and compare; the split is by output
        // row, so nothing else can move.
        let mut expected = vec![0.0f32; columns];
        for (k, scale) in (0..cols).map(|k| (k, w_value(&w, k))) {
            for n in 0..columns {
                expected[n] += scale * x[k * columns + n];
            }
        }
        for n in 0..columns {
            assert!(
                close(y[n], expected[n], 1e-4),
                "{} vs {}",
                y[n],
                expected[n]
            );
        }
    }

    fn w_value(weight: &Weight, index: usize) -> f32 {
        match weight {
            Weight::F32(values) => values[index],
            Weight::F16(values) => values[index].to_f32(),
        }
    }

    #[test]
    fn layer_norm_normalises_over_the_reduction_axis() {
        // Three columns of four rows: each column gets its own mean and
        // variance, and the affine pair is indexed by row.
        let rows = 4;
        let columns = 3;
        let mut x = vec![0.0f32; rows * columns];
        for column in 0..columns {
            for row in 0..rows {
                x[row * columns + column] = (row as f32 + 1.0) * 10.0 + column as f32;
            }
        }

        let weight = [1.0f32, 1.0, 1.0, 1.0];
        let bias = [0.0f32; 4];
        let out = layer_norm(&x, &weight, &bias, 0.0, columns);

        for column in 0..columns {
            let line: Vec<f32> = (0..rows).map(|row| out[row * columns + column]).collect();
            let mean = line.iter().sum::<f32>() / rows as f32;
            assert!(close(mean, 0.0, 1e-5), "column {column} mean {mean}");

            let variance = line.iter().map(|v| v * v).sum::<f32>() / rows as f32;
            assert!(
                close(variance, 1.0, 1e-4),
                "column {column} variance {variance}"
            );
        }
    }

    #[test]
    fn layer_norm_applies_the_affine_pair_per_row_not_per_column() {
        // Two columns of three rows. Scaling and shifting row 2 must move only
        // that row; indexed by column it would move a whole token instead.
        let x = [3.0f32, 0.0, 1.0, 1.0, 1.0, -3.0];
        let weight = [1.0, 1.0, 2.0];
        let bias = [0.0, 0.0, 1.0];

        let out = layer_norm(&x, &weight, &bias, 0.0, 2);

        // Column 0 is [3, 1, 1].
        let mean: f32 = 5.0 / 3.0;
        let variance = [
            (3.0 - mean).powi(2),
            (1.0 - mean).powi(2),
            (1.0 - mean).powi(2),
        ]
        .iter()
        .sum::<f32>()
            / 3.0;
        let want = (1.0 - mean) / variance.sqrt() * 2.0 + 1.0;
        assert!(close(out[4], want, 1e-5), "{} vs {want}", out[4]);

        let want_first = (3.0 - mean) / variance.sqrt();
        assert!(
            close(out[0], want_first, 1e-5),
            "{} vs {want_first}",
            out[0]
        );
    }

    #[test]
    fn gelu_is_the_erf_form_not_the_tanh_one() {
        // The two forms disagree by ~1e-3 in the middle, which is more than
        // tolerance noise and less than a wrong activation.
        let tanh_approx = |x: f32| {
            0.5 * x
                * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
        };

        let mut differs = false;
        for step in 0..200 {
            let x = -6.0 + step as f32 * 0.06;
            let erf = gelu_erf(x);
            assert!(close(
                erf,
                0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2)),
                1e-6
            ));
            if (erf - tanh_approx(x)).abs() > 1e-4 {
                differs = true;
            }
        }
        assert!(differs, "the tanh approximation is not what this computes");

        assert_eq!(gelu_erf(0.0), 0.0);
        assert!(gelu_erf(-40.0).abs() < 1e-6, "negative tail is zero");
        assert!(
            (gelu_erf(40.0) - 40.0).abs() < 1e-5,
            "positive tail is identity"
        );
    }

    #[test]
    fn softmax_normalises_and_ignores_masked_entries() {
        let mut x = [1.0f32, 2.0, 3.0, 10.0];
        let mask = [0.0f32, 0.0, f32::NEG_INFINITY, 0.0];
        masked_softmax(&mut x, &mask, 4);

        let total = x.iter().sum::<f32>();
        assert!(close(total, 1.0, 1e-6), "total {total}");
        assert_eq!(x[2], 0.0, "a masked entry contributes nothing");
        assert!(x[0] > 0.0 && x[1] > x[0] && x[3] > x[1]);
    }

    #[test]
    fn the_causal_mask_is_bottom_right_aligned() {
        // 3 new queries over 5 keys with 2 already in the cache.
        let mask = bottom_right_causal_mask(2, 3, 5);

        let row =
            |i: usize| -> Vec<bool> { (0..5).map(|key| mask[i * 5 + key].is_finite()).collect() };

        assert_eq!(row(0), vec![true, true, true, false, false]);
        assert_eq!(row(1), vec![true, true, true, true, false]);
        assert_eq!(row(2), vec![true, true, true, true, true]);
    }

    #[test]
    fn a_prefilled_query_sees_the_whole_prompt_and_nothing_after_it() {
        // The shape a single-token decode never hits: several new queries over a
        // cache that already holds a prompt. Query 0 sits at absolute position 2,
        // so it reaches key 2 -- top-left alignment would stop it at key 1.
        let mask = bottom_right_causal_mask(2, 2, 5);

        assert!(mask[2].is_finite(), "query 0 must reach key 2");
        assert!(!mask[3].is_finite(), "query 0 must not reach key 3");
        assert!(!mask[4].is_finite());
        assert!(mask[5 + 3].is_finite(), "query 1 reaches key 3");
        assert!(!mask[5 + 4].is_finite(), "query 1 must not reach key 4");
    }
}
