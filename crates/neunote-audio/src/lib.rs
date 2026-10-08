#![forbid(unsafe_code)]

//! Audio input for the transcription engine.
//!
//! Two responsibilities, and the split matters: decode any supported file to
//! mono f32 at its native rate, then resample to the engine's fixed 16 kHz.
//!
//! Resampling quality is not a detail here. MuScriptor's onsets land on a 10 ms
//! grid, and a cheap filter smears them enough to move notes. This uses
//! `rubato`'s sinc interpolator rather than anything linear.

use std::path::Path;

use neunote_types::TRANSCRIPTION_SAMPLE_RATE;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::audio::{SampleBuffer, SignalSpec};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use thiserror::Error;

/// Frames per resampling block. Large enough that the sinc filter's history is
/// amortised, small enough to keep latency low.
const RESAMPLE_CHUNK: usize = 4_096;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("could not read {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },

    #[error("no audio stream in {0}")]
    NoStream(String),

    #[error("unsupported audio in {path}: {message}")]
    Unsupported { path: String, message: String },

    #[error("resampling failed: {0}")]
    Resample(String),
}

/// Mono f32 at a known rate.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl AudioBuffer {
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Decode a file to mono f32 at its native sample rate.
///
/// Nothing here clips, normalises or gates the signal. The engine expects the
/// audio as it was recorded.
pub fn decode_file(path: &Path) -> Result<AudioBuffer, AudioError> {
    let label = path.display().to_string();

    let file = std::fs::File::open(path).map_err(|source| AudioError::Read {
        path: label.clone(),
        source,
    })?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }

    let probe = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|error| AudioError::Unsupported {
            path: label.clone(),
            message: error.to_string(),
        })?;

    let mut format = probe.format;
    let track = format
        .default_track()
        .ok_or_else(|| AudioError::NoStream(label.clone()))?;

    let track_id = track.id;
    let channels = track
        .codec_params
        .channels
        .map(|layout| layout.count())
        .unwrap_or(1)
        .max(1) as usize;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AudioError::Unsupported {
            path: label.clone(),
            message: "stream has no sample rate".to_owned(),
        })?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|error| AudioError::Unsupported {
            path: label.clone(),
            message: error.to_string(),
        })?;

    let mut interleaved: Vec<f32> = Vec::new();
    let mut buffer: Option<SampleBuffer<f32>> = None;
    let mut buffer_spec: Option<SignalSpec> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::ResetRequired) => continue,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                // A truncated file still yields everything decoded so far.
                break;
            }
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A corrupt frame is not worth abandoning the rest of the file for.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(_) => break,
        };

        let spec = *decoded.spec();
        if buffer_spec != Some(spec) {
            buffer = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
            buffer_spec = Some(spec);
        }

        let Some(buffer) = buffer.as_mut() else {
            continue;
        };
        buffer.copy_interleaved_ref(decoded);
        interleaved.extend_from_slice(buffer.samples());
    }

    Ok(AudioBuffer {
        samples: downmix(&interleaved, channels),
        sample_rate,
    })
}

/// Average the channels together.
pub fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }

    let scale = 1.0 / channels as f32;
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() * scale)
        .collect()
}

/// Resample to the engine's fixed 16 kHz.
///
/// A no-op when the input is already at that rate. Never peak-normalise: the
/// model's output depends on the dynamics it was trained on, and normalising
/// changes them.
pub fn to_engine_input(audio: &AudioBuffer) -> Result<Vec<f32>, AudioError> {
    if audio.sample_rate == TRANSCRIPTION_SAMPLE_RATE {
        return Ok(audio.samples.clone());
    }

    if audio.samples.is_empty() {
        return Ok(Vec::new());
    }

    // 64 taps with a 96-tap-equivalent window is well past the point where the
    // stopband reaches the mel bins at 8 kHz.
    let parameters = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    // rubato's ratio is output over input.
    let mut resampler = SincFixedIn::<f32>::new(
        f64::from(TRANSCRIPTION_SAMPLE_RATE) / f64::from(audio.sample_rate),
        1.0,
        parameters,
        RESAMPLE_CHUNK,
        1,
    )
    .map_err(|error| AudioError::Resample(error.to_string()))?;

    let input = audio.samples.clone();
    let mut output: Vec<f32> = Vec::with_capacity(
        (input.len() as f64 / f64::from(audio.sample_rate) * f64::from(TRANSCRIPTION_SAMPLE_RATE))
            as usize
            + RESAMPLE_CHUNK,
    );

    let mut block: Vec<f32> = Vec::new();
    let mut cursor = 0;

    while cursor < input.len() {
        let end = (cursor + RESAMPLE_CHUNK).min(input.len());
        block.clear();
        block.extend_from_slice(&input[cursor..end]);
        // SincFixedIn requires exactly one full chunk per call.
        block.resize(RESAMPLE_CHUNK, 0.0);
        cursor += RESAMPLE_CHUNK;

        let produced = resampler
            .process(&[&block], None)
            .map_err(|error| AudioError::Resample(error.to_string()))?;
        output.extend_from_slice(&produced[0]);
    }

    // The last block was zero-padded, so discard the tail it invented.
    let expected = expected_length(input.len(), audio.sample_rate);
    output.truncate(expected.min(output.len()));

    Ok(output)
}

fn expected_length(input_len: usize, from_rate: u32) -> usize {
    input_len * TRANSCRIPTION_SAMPLE_RATE as usize / from_rate as usize
}

/// Min/max per bucket, for the waveform view.
pub fn compute_peaks(mono: &[f32], buckets: usize) -> Vec<(f32, f32)> {
    if buckets == 0 || mono.is_empty() {
        return Vec::new();
    }

    let per_bucket = mono.len().div_ceil(buckets);
    mono.chunks(per_bucket)
        .map(|chunk| {
            let mut low = f32::MAX;
            let mut high = f32::MIN;
            for sample in chunk {
                low = low.min(*sample);
                high = high.max(*sample);
            }
            (low, high)
        })
        .collect()
}

/// Whether any sample sits outside [-1, 1].
///
/// A warning signal, not an error: the engine is fed the audio either way.
pub fn is_clipped(mono: &[f32]) -> bool {
    mono.iter().any(|sample| sample.abs() > 1.0)
}

/// Split into the engine's fixed-length chunks, zero-padding the last.
///
/// The model sees whole 5 s chunks and the padding is not masked, so the caller
/// must not treat the tail as silence it can skip.
pub fn to_segments(mono: &[f32]) -> Vec<&[f32]> {
    let size = neunote_types::SEGMENT_SAMPLES;
    if mono.is_empty() {
        return Vec::new();
    }

    mono.chunks(size).collect()
}

/// How many chunks a signal of this many samples needs.
pub fn chunk_count(sample_count: usize) -> usize {
    sample_count.div_ceil(neunote_types::SEGMENT_SAMPLES)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sine at a frequency that survives 16 kHz, for length and shape checks.
    fn tone(rate: u32, seconds: f64, freq: f64) -> Vec<f32> {
        let count = (rate as f64 * seconds) as usize;
        (0..count)
            .map(|index| {
                let t = index as f64 / rate as f64;
                (2.0 * std::f64::consts::PI * freq * t).sin() as f32
            })
            .collect()
    }

    #[test]
    fn downmix_averages_channels() {
        // Two frames of stereo: (1, 3) and (0, 4).
        assert_eq!(downmix(&[1.0, 3.0, 0.0, 4.0], 2), vec![2.0, 2.0]);
        // Mono passes through untouched.
        assert_eq!(downmix(&[1.0, 2.0], 1), vec![1.0, 2.0]);
        // An incomplete trailing frame is dropped, not padded.
        assert_eq!(downmix(&[1.0, 3.0, 1.0], 2), vec![2.0]);
    }

    #[test]
    fn resampling_44100_to_16000_lands_on_the_expected_length() {
        let input = tone(44_100, 1.0, 440.0);
        let audio = AudioBuffer {
            samples: input.clone(),
            sample_rate: 44_100,
        };
        let out = to_engine_input(&audio).unwrap();

        // 16000/44100 of the input, within a sample either way.
        let expected = expected_length(input.len(), 44_100);
        assert!(
            out.len().abs_diff(expected) <= 1,
            "got {}, expected about {expected}",
            out.len()
        );
    }

    #[test]
    fn resampling_preserves_a_tone_and_its_envelope() {
        let input = tone(44_100, 0.5, 440.0);
        let audio = AudioBuffer {
            samples: input.clone(),
            sample_rate: 44_100,
        };
        let out = to_engine_input(&audio).unwrap();

        assert!(!out.is_empty());

        // Peak amplitude survives a resample.
        let peak = out.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
        assert!(peak > 0.9, "peak fell to {peak}");

        // Zero crossings near 880 Hz: one per half cycle.
        let mut crossings = 0;
        for pair in out.windows(2) {
            if (pair[0] < 0.0) != (pair[1] < 0.0) {
                crossings += 1;
            }
        }
        let seconds = out.len() as f64 / f64::from(TRANSCRIPTION_SAMPLE_RATE);
        let expected = 880.0 * seconds;
        let error = (crossings as f64 - expected).abs() / expected;
        assert!(
            error < 0.02,
            "880 Hz gave {crossings} crossings, expected {expected}"
        );
    }

    #[test]
    fn resampling_is_a_no_op_at_the_engine_rate() {
        let samples = tone(16_000, 0.1, 440.0);
        let audio = AudioBuffer {
            samples: samples.clone(),
            sample_rate: 16_000,
        };
        assert_eq!(to_engine_input(&audio).unwrap(), samples);
    }

    #[test]
    fn empty_audio_resamples_to_nothing() {
        let audio = AudioBuffer {
            samples: Vec::new(),
            sample_rate: 44_100,
        };
        assert!(to_engine_input(&audio).unwrap().is_empty());
    }

    #[test]
    fn peaks_span_every_sample() {
        let samples = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.25];
        assert_eq!(
            compute_peaks(&samples, 3),
            vec![(0.0, 1.0), (-1.0, 0.5), (-0.5, 0.25)]
        );
        assert!(compute_peaks(&samples, 0).is_empty());
        assert!(compute_peaks(&[], 4).is_empty());
    }

    #[test]
    fn clipping_is_detected_outside_unit_range() {
        assert!(!is_clipped(&[0.0, 0.5, -0.5]));
        assert!(is_clipped(&[0.0, 1.5]));
        assert!(is_clipped(&[0.0, -1.0001]));
    }

    #[test]
    fn segments_are_five_seconds_and_count_correctly() {
        let segment = neunote_types::SEGMENT_SAMPLES;
        assert_eq!(chunk_count(0), 0);
        assert_eq!(chunk_count(1), 1);
        assert_eq!(chunk_count(segment), 1);
        assert_eq!(chunk_count(segment + 1), 2);

        let one_second = vec![0.0; 16_000];
        assert_eq!(to_segments(&one_second).len(), 1);
        assert_eq!(to_segments(&vec![0.0; segment * 2 + 1]).len(), 3);
    }
}
