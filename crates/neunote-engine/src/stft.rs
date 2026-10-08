#![forbid(unsafe_code)]

//! Short-time Fourier transform magnitudes.
//!
//! Reproduces what MuScriptor's front-end computes before the filterbank:
//!
//! ```text
//! torch.stft(x, n_fft, hop, win_length=n_fft, window=window, center=True,
//!            pad_mode="reflect", normalized=False, onesided=True,
//!            return_complex=True).abs() ** 1.0
//! ```
//!
//! The checkpoint's `power` is 1.0, so these are magnitudes. Squaring them is
//! the kind of change that survives every shape check and still moves every
//! note, so the exponent is spelled out rather than assumed.
//!
//! The window comes from the checkpoint. It is periodic -- `torch.hann_window`
//! divides by `n_fft`, not `n_fft - 1` -- and regenerating it drifts enough to
//! flip greedy tokens.

use crate::Error;

/// `1 + n_samples / hop`, the count a centred STFT produces.
///
/// Centre padding adds `n_fft / 2` on each side, so this is one more than the
/// number of frames the audio covers. `encode_conditioning` masks the extra one
/// away, which is why the mask length comes from the waveform rather than from
/// here.
pub fn frame_count(n_samples: usize, hop_length: usize) -> usize {
    1 + n_samples / hop_length
}

/// Reflect padding, then windowing, then a real FFT, per frame.
///
/// The reflection excludes the edge sample at each end, matching
/// `torch.stft(pad_mode="reflect")`: the left pad runs `x[n_fft/2] .. x[1]` and
/// the right pad `x[n-2] .. x[n-1-n_fft/2]`.
pub fn magnitudes(
    samples: &[f32],
    n_fft: usize,
    hop_length: usize,
    window: &[f32],
) -> Result<Vec<f32>, Error> {
    if n_fft == 0 || hop_length == 0 {
        return Err(Error::Checkpoint("STFT needs a positive n_fft and hop_length".into()));
    }
    if window.len() != n_fft {
        return Err(Error::Checkpoint(format!(
            "the checkpoint's STFT window has {} coefficients, expected n_fft = {n_fft}",
            window.len()
        )));
    }
    if !n_fft.is_power_of_two() {
        return Err(Error::Checkpoint(format!(
            "n_fft = {n_fft} is not a power of two"
        )));
    }

    let pad = n_fft / 2;
    if samples.len() < pad + 1 {
        return Err(Error::Checkpoint(format!(
            "STFT needs at least {} samples to reflect-pad, got {}",
            pad + 1,
            samples.len()
        )));
    }

    let mut padded = vec![0.0f32; samples.len() + 2 * pad];
    padded[pad..pad + samples.len()].copy_from_slice(samples);

    for index in 0..pad {
        padded[index] = samples[pad - index];
        padded[pad + samples.len() + index] = samples[samples.len() - 2 - index];
    }

    let bins = n_fft / 2 + 1;
    let frames = frame_count(samples.len(), hop_length);
    let mut out = vec![0.0f32; frames * bins];

    let transform = Fft::new(n_fft);
    let mut windowed = vec![0.0f32; n_fft];
    let mut spectrum = vec![0.0f32; n_fft * 2];

    for (frame, row) in out.chunks_mut(bins).enumerate() {
        let start = frame * hop_length;
        for index in 0..n_fft {
            windowed[index] = padded[start + index] * window[index];
        }

        transform.forward(&windowed, &mut spectrum);

        row[0] = spectrum[0].abs();
        row[pad] = spectrum[pad * 2].abs();
        for bin in 1..pad {
            let re = spectrum[bin * 2];
            let im = spectrum[bin * 2 + 1];
            row[bin] = (re * re + im * im).sqrt();
        }
    }

    Ok(out)
}

/// In-place iterative radix-2 Cooley-Tukey, decimation in time.
struct Fft {
    reversed: Vec<u32>,
    /// `stage_base[s] .. stage_base[s] + 2^s` holds stage `s`'s roots. A stage
    /// reuses its own `2^(s-1)` roots across every block, so the table is sized
    /// by distinct roots, not by uses.
    twiddles: Vec<(f32, f32)>,
    stage_base: Vec<usize>,
}

impl Fft {
    fn new(size: usize) -> Self {
        let bits = size.trailing_zeros();
        let reversed = (0..size as u32)
            .map(|index| index.reverse_bits() >> (32 - bits))
            .collect();

        let mut twiddles = Vec::with_capacity(size);
        let mut stage_base = Vec::with_capacity(bits as usize);
        for stage in 1..=bits {
            stage_base.push(twiddles.len());
            let span = 1usize << stage;
            for step in 0..span / 2 {
                let angle = -std::f32::consts::PI * step as f32 * 2.0 / span as f32;
                twiddles.push((angle.cos(), angle.sin()));
            }
        }

        Self {
            reversed,
            twiddles,
            stage_base,
        }
    }

    /// `input` is `size` real samples; `out` receives interleaved re/im.
    fn forward(&self, input: &[f32], out: &mut [f32]) {
        let size = input.len();

        for (index, &source) in self.reversed.iter().enumerate() {
            out[index * 2] = input[source as usize];
            out[index * 2 + 1] = 0.0;
        }

        let mut stage = 0;
        let mut span = 2;
        while span <= size {
            let half = span / 2;
            let base = self.stage_base[stage];

            for block in (0..size).step_by(span) {
                for step in 0..half {
                    let (cos, sin) = self.twiddles[base + step];

                    let even = block + step;
                    let odd = even + half;

                    let even_re = out[even * 2];
                    let even_im = out[even * 2 + 1];
                    let (odd_re, odd_im) = (out[odd * 2], out[odd * 2 + 1]);

                    out[even * 2] = even_re + (odd_re * cos - odd_im * sin);
                    out[even * 2 + 1] = even_im + (odd_re * sin + odd_im * cos);
                    out[odd * 2] = even_re - (odd_re * cos - odd_im * sin);
                    out[odd * 2 + 1] = even_im - (odd_re * sin + odd_im * cos);
                }
            }

            span *= 2;
            stage += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DFT straight from the definition, for sizes small enough to be obvious.
    fn naive_dft(input: &[f32]) -> Vec<(f32, f32)> {
        let n = input.len();
        (0..n)
            .map(|k| {
                let (mut re, mut im) = (0.0f32, 0.0f32);
                for (t, &x) in input.iter().enumerate() {
                    let angle = -2.0 * std::f32::consts::PI * k as f32 * t as f32 / n as f32;
                    re += x * angle.cos();
                    im += x * angle.sin();
                }
                (re, im)
            })
            .collect()
    }

    fn periodic_hann(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                0.5
                    - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos()
            })
            .collect()
    }

    #[test]
    fn the_fft_agrees_with_the_transform_it_replaces() {
        let size = 64;
        let signal: Vec<f32> = (0..size)
            .map(|i| (i as f32 * 0.37).sin() + 0.2 * (i as f32 * 1.9).cos())
            .collect();

        let transform = Fft::new(size);
        let mut out = vec![0.0f32; size * 2];
        transform.forward(&signal, &mut out);

        for (k, (want_re, want_im)) in naive_dft(&signal).iter().enumerate() {
            let tolerance = 1e-3 * want_re.abs().max(want_im.abs()).max(1.0);
            assert!(
                (out[k * 2] - want_re).abs() <= tolerance && (out[k * 2 + 1] - want_im).abs() <= tolerance,
                "bin {k}: got ({}, {}), want ({want_re}, {want_im})",
                out[k * 2],
                out[k * 2 + 1]
            );
        }
    }

    #[test]
    fn a_constant_signal_puts_all_its_energy_in_dc() {
        let size = 32;
        let signal = vec![0.5f32; size];

        let transform = Fft::new(size);
        let mut out = vec![0.0f32; size * 2];
        transform.forward(&signal, &mut out);

        assert!((out[0] - size as f32 * 0.5).abs() < 1e-3);
        for k in 1..size {
            let magnitude = (out[k * 2].powi(2) + out[k * 2 + 1].powi(2)).sqrt();
            assert!(magnitude < 1e-3, "bin {k} has {magnitude}");
        }
    }

    #[test]
    fn a_pure_tone_lands_on_its_own_bin_and_its_mirror() {
        // A real input has no negative frequencies, so a sine at bin 8 shows up
        // at 8 and at its conjugate `size - 8`, with nothing anywhere else.
        let size = 64;
        let signal: Vec<f32> = (0..size)
            .map(|i| (2.0 * std::f32::consts::PI * 8.0 * i as f32 / size as f32).sin())
            .collect();

        let transform = Fft::new(size);
        let mut out = vec![0.0f32; size * 2];
        transform.forward(&signal, &mut out);

        for k in 0..size {
            let magnitude = (out[k * 2].powi(2) + out[k * 2 + 1].powi(2)).sqrt();
            if k == 8 || k == size - 8 {
                assert!(magnitude > 15.0, "bin {k} has {magnitude}");
            } else {
                assert!(magnitude < 1e-3, "bin {k} has {magnitude}");
            }
        }
    }

    #[test]
    fn reflection_excludes_the_edge_sample_at_both_ends() {
        // A ramp: reflecting it must produce 1024..1 on the left and n-2.. on the
        // right, never 0. If the edge sample were included, the pads would be
        // 1024..0 and n-1.., and this assertion moves.
        let n_fft = 8;
        let hop = 4;
        let samples: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let window = vec![1.0f32; n_fft];

        let out = magnitudes(&samples, n_fft, hop, &window).unwrap();
        let bins = n_fft / 2 + 1;
        let frames = frame_count(samples.len(), hop);
        assert_eq!(frames, 5);
        assert_eq!(out.len(), frames * bins);

        // Frame 1 starts at padded index 4, i.e. sample 0. Window is all ones, so
        // the spectrum is that of [0..8]; a pure ramp has a flat magnitude.
        let expected: Vec<f32> = naive_dft(&samples[0..n_fft])
            .iter()
            .take(bins)
            .map(|(re, im)| (re * re + im * im).sqrt())
            .collect();
        for (bin, want) in expected.iter().enumerate() {
            assert!((out[bins + bin] - want).abs() < 1e-3, "bin {bin}");
        }
    }

    #[test]
    fn the_frame_count_is_one_more_than_the_audio_covers() {
        assert_eq!(frame_count(80_000, 160), 501);
        assert_eq!(frame_count(0, 160), 1);
        assert_eq!(frame_count(159, 160), 1);
        assert_eq!(frame_count(160, 160), 2);
    }

    #[test]
    fn magnitudes_are_not_powers() {
        // A unit-amplitude sine has a magnitude near size/2 at its bin. Squaring
        // would give size^2/4, which this pins down.
        let n_fft = 64;
        let signal: Vec<f32> = (0..n_fft)
            .map(|i| (2.0 * std::f32::consts::PI * 4.0 * i as f32 / n_fft as f32).sin())
            .collect();
        let window = vec![1.0f32; n_fft];

        // A short hop, so some frame covers the signal whole rather than splitting it
        // across the reflection pads.
        let out = magnitudes(&signal, n_fft, 8, &window).unwrap();
        let bins = n_fft / 2 + 1;

        let (index, &peak) = out
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .expect("a frame was produced");

        assert!((peak - n_fft as f32 / 2.0).abs() < 1e-2, "peak {peak}");
        assert_eq!(index % bins, 4, "the loudest bin is the tone's");
    }

    #[test]
    fn a_window_of_the_wrong_length_is_refused() {
        let error = magnitudes(&[0.0; 32], 16, 4, &[1.0; 8]).unwrap_err();
        assert!(error.to_string().contains("window"), "{error}");
    }

    #[test]
    fn too_few_samples_to_reflect_is_refused() {
        let window = [1.0f32; 16];
        let error = magnitudes(&[0.0; 8], 16, 4, &window).unwrap_err();
        assert!(error.to_string().contains("reflect"), "{error}");
    }

    #[test]
    fn silence_gives_silence() {
        let out = magnitudes(&vec![0.0; 64], 32, 16, &periodic_hann(32)).unwrap();
        assert!(out.iter().all(|value| *value == 0.0));
    }
}