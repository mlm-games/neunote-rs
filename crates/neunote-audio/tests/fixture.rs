//! Decodes the reference's own audio fixture, so the decode path is exercised
//! against a file that is known to be exactly what the engine expects.
//!
//! `testdata/audio/fixture_3chunks_16k.wav` is vendored from
//! `muscriptor.cpp`: 15 s of 16 kHz mono f32, which is three whole 5 s chunks.

use std::path::Path;

use neunote_audio::{chunk_count, decode_file, is_clipped, to_engine_input, to_segments};

fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/audio/fixture_3chunks_16k.wav")
}

#[test]
fn the_fixture_decodes_to_three_whole_chunks() {
    let audio = decode_file(&fixture()).expect("decoding the fixture");

    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.samples.len(), 240_000);
    assert!((audio.duration_secs() - 15.0).abs() < 1e-9);
    assert!(
        !is_clipped(&audio.samples),
        "the fixture is already in range"
    );

    let segments = to_segments(&audio.samples);
    assert_eq!(segments.len(), 3);
    assert_eq!(chunk_count(audio.samples.len()), 3);
    assert!(segments.iter().all(|chunk| chunk.len() == 80_000));
}

#[test]
fn the_fixture_is_already_at_the_engine_rate() {
    let audio = decode_file(&fixture()).unwrap();
    // The engine's fixed input rate, so no resampling happens at all.
    assert_eq!(to_engine_input(&audio).unwrap().len(), audio.samples.len());
}

#[test]
fn the_fixture_carries_real_signal() {
    let audio = decode_file(&fixture()).unwrap();

    let peak = audio.samples.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
    assert!(peak > 0.1, "fixture is nearly silent, peak {peak}");

    let energy: f32 = audio.samples.iter().map(|s| s * s).sum();
    assert!(energy > 0.0);

    // Each chunk has to carry signal of its own, or the cross-chunk state
    // machine has nothing to carry.
    for (index, chunk) in to_segments(&audio.samples).iter().enumerate() {
        let chunk_energy: f32 = chunk.iter().map(|s| s * s).sum();
        assert!(
            chunk_energy > 0.0,
            "chunk {index} is silent, so prelude forcing would have nothing to test"
        );
    }
}

#[test]
fn a_missing_file_reports_a_read_error() {
    let missing = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/audio/nope.wav");
    let error = decode_file(&missing).unwrap_err();
    assert!(matches!(error, neunote_audio::AudioError::Read { .. }));
}

#[test]
fn a_non_audio_file_reports_unsupported() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.toml");
    let error = decode_file(&manifest).unwrap_err();
    assert!(matches!(
        error,
        neunote_audio::AudioError::Unsupported { .. }
    ));
}
