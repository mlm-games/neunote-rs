//! Timing properties of the resampler.
//!
//! These lock in what the resampler already gets right. The transient-alignment
//! cases are the ones worth having: onset placement is what MuScriptor's 10 ms
//! frame grid is measured against, so a resampler that shifted transients would
//! move every note the model emits while still passing a length or spectrum
//! check.

use neunote_audio::{AudioBuffer, to_engine_input};
use neunote_types::TRANSCRIPTION_SAMPLE_RATE;

fn resample(samples: Vec<f32>, sample_rate: u32) -> Vec<f32> {
    to_engine_input(&AudioBuffer {
        samples,
        sample_rate,
    })
    .unwrap()
}

/// Index of the largest-magnitude sample.
fn peak_index(samples: &[f32]) -> usize {
    samples
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[test]
fn an_impulse_lands_where_the_sample_rate_ratio_says_it_should() {
    for from_rate in [22_050u32, 44_100, 48_000, 96_000] {
        let impulse_at = from_rate as usize / 4;
        let mut input = vec![0.0f32; from_rate as usize];
        input[impulse_at] = 1.0;

        let out = resample(input, from_rate);
        let ideal = impulse_at as f64 * f64::from(TRANSCRIPTION_SAMPLE_RATE) / f64::from(from_rate);
        let drift_ms =
            (peak_index(&out) as f64 - ideal) * 1000.0 / f64::from(TRANSCRIPTION_SAMPLE_RATE);

        // Half a millisecond is a fifth of a frame; the reference's own mel
        // grid is 10 ms, so anything near a frame's worth of drift would move
        // notes. rubato's sinc resampler compensates its group delay, so the
        // error here should be far below that.
        assert!(
            drift_ms.abs() < 0.5,
            "{from_rate} Hz: impulse drifted {drift_ms:.3} ms"
        );
    }
}

#[test]
fn a_click_keeps_its_rise_within_a_fraction_of_a_frame() {
    // A decaying burst rather than a bare impulse, so the measurement follows
    // the onset and not a single sample.
    let from_rate = 44_100u32;
    let click_at = from_rate as usize / 3;
    let mut input = vec![0.0f32; from_rate as usize];
    for offset in 0..64 {
        input[click_at + offset] = (-(offset as f32) * 0.2).exp();
    }

    let out = resample(input, from_rate);
    let ideal = click_at as f64 * f64::from(TRANSCRIPTION_SAMPLE_RATE) / f64::from(from_rate);
    let drift_ms =
        (peak_index(&out) as f64 - ideal) * 1000.0 / f64::from(TRANSCRIPTION_SAMPLE_RATE);

    assert!(drift_ms.abs() < 0.5, "click drifted {drift_ms:.3} ms");
}

#[test]
fn silence_stays_silent() {
    let out = resample(vec![0.0f32; 44_100], 44_100);
    assert!(out.iter().all(|sample| *sample == 0.0));
}

#[test]
fn the_resampled_length_is_the_ratio_of_the_two_rates() {
    for from_rate in [22_050u32, 44_100, 48_000, 96_000] {
        for seconds in [1usize, 2, 3] {
            let input = vec![0.1f32; from_rate as usize * seconds];
            let out = resample(input, from_rate);
            let expected = from_rate as usize * seconds * TRANSCRIPTION_SAMPLE_RATE as usize
                / from_rate as usize;
            assert!(
                out.len().abs_diff(expected) <= 1,
                "{from_rate} Hz x {seconds}s: got {}, expected about {expected}",
                out.len()
            );
        }
    }
}

#[test]
fn an_input_shorter_than_one_block_is_still_resampled() {
    // A file shorter than the block size takes the zero-padded path; it must
    // still come back at the right length rather than empty.
    let out = resample(vec![0.25f32; 100], 44_100);
    assert!(!out.is_empty(), "a 100-sample clip must not vanish");
    assert!(out.len() <= 64, "got {} samples", out.len());
}
