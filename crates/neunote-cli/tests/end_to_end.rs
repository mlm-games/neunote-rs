//! The whole command, run for real against the reference's audio fixture.
//!
//! These run the binary rather than calling into it, so argument parsing,
//! process exit codes and the output file are all covered.

use std::path::{Path, PathBuf};
use std::process::Command;

use neunote_types::ModelSize;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_neunote")
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/audio/fixture_3chunks_16k.wav")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("neunote-cli-test-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(args: &[&str], models_dir: &Path) -> (bool, String, String) {
    let output = Command::new(binary())
        .args(args)
        .env("NEUNOTE_MODELS_DIR", models_dir)
        .output()
        .expect("running the neunote binary");

    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn help_is_available_and_succeeds() {
    let models = scratch("help");
    let (ok, stdout, _) = run(&["--help"], &models);
    assert!(ok);
    for command in ["devices", "models", "transcribe"] {
        assert!(stdout.contains(command), "{command} missing from help");
    }
}

#[test]
fn models_list_reports_every_size_and_the_licence() {
    let models = scratch("list");
    let (ok, stdout, _) = run(&["models", "list"], &models);

    assert!(ok);
    assert!(stdout.contains("muscriptor-small-f16.gguf"));
    assert!(stdout.contains("muscriptor-medium-f16.gguf"));
    assert!(stdout.contains("muscriptor-large-f16.gguf"));
    assert!(stdout.contains("not downloaded"));
    assert!(
        stdout.contains("CC BY-NC 4.0"),
        "the licence must be visible"
    );
}

#[test]
fn models_path_prints_a_directory() {
    let models = scratch("path");
    let (ok, stdout, _) = run(&["models", "path"], &models);

    assert!(ok);
    assert_eq!(Path::new(stdout.trim()), models.as_path());
}

#[test]
fn models_verify_reports_what_is_missing_without_failing() {
    let models = scratch("verify");
    let (ok, stdout, _) = run(&["models", "verify"], &models);

    // Nothing is installed, which is a report rather than an error.
    assert!(ok);
    assert!(stdout.contains("not installed"));
    assert!(stdout.contains("small"));
    assert!(stdout.contains("medium"));
}

#[test]
fn transcribing_refuses_an_unreadable_model_rather_than_writing_an_empty_file() {
    let models = scratch("no-engine");
    let out = models.join("out.mid");

    // A file the right size but not a checkpoint. Standing in for a verified
    // install lets the run reach the engine and fail there, rather than stopping
    // at the missing model -- and the failure has to name what is actually
    // wrong rather than something plausible.
    let model = models.join("muscriptor-medium-f16.gguf");
    std::fs::write(
        &model,
        vec![0u8; neunote_models::entry(ModelSize::Medium).num_bytes as usize],
    )
    .unwrap();

    let (ok, _, stderr) = run(
        &[
            "transcribe",
            fixture().to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ],
        &models,
    );

    assert!(!ok, "an unreadable checkpoint is an error, not a silent success");
    assert!(
        stderr.contains("GGUF") || stderr.contains("checkpoint"),
        "the error must name the real cause, got: {stderr}"
    );
    assert!(
        !out.exists(),
        "no MIDI file may be written when transcription did not run"
    );
}
#[test]
fn transcribing_reports_the_audio_it_read() {
    let models = scratch("audio-report");
    let (_, _, stderr) = run(&["transcribe", fixture().to_str().unwrap()], &models);

    // 15 s of 16 kHz mono, which is three chunks.
    assert!(stderr.contains("15.0s"), "got: {stderr}");
    assert!(stderr.contains("240000 samples"), "got: {stderr}");
    assert!(stderr.contains("16000 Hz"), "got: {stderr}");
}

#[test]
fn transcribing_reports_a_missing_input_file() {
    let models = scratch("missing-input");
    let (ok, _, stderr) = run(&["transcribe", "/nonexistent/audio.wav"], &models);

    assert!(!ok);
    assert!(stderr.contains("could not read"), "got: {stderr}");
}

#[test]
fn transcribing_rejects_an_unknown_instrument() {
    let models = scratch("bad-instrument");
    let (ok, _, stderr) = run(
        &[
            "transcribe",
            fixture().to_str().unwrap(),
            "--instruments",
            "kazoo",
        ],
        &models,
    );

    assert!(!ok);
    assert!(stderr.contains("kazoo"), "got: {stderr}");
}

#[test]
fn a_missing_model_is_reported_before_any_expensive_work() {
    let models = scratch("no-model");
    let (_, _, stderr) = run(&["transcribe", fixture().to_str().unwrap()], &models);

    assert!(stderr.contains("no verified medium model"), "got: {stderr}");
    assert!(stderr.contains("models fetch"), "the fix should be named");
}

#[test]
fn devices_says_the_engine_is_missing_instead_of_faking_a_list() {
    let models = scratch("devices");
    let (ok, _, stderr) = run(&["devices"], &models);

    assert!(ok);
    assert!(stderr.contains("not been built"), "got: {stderr}");
}

#[test]
fn an_unknown_subcommand_fails_with_usage() {
    let models = scratch("bad-subcommand");
    let (ok, _, stderr) = run(&["frobnicate"], &models);

    assert!(!ok);
    assert!(stderr.contains("unrecognized") || stderr.contains("Usage"));
}

#[test]
fn fetch_requires_the_licence_to_be_accepted_first() {
    let models = scratch("licence");
    let (ok, _, stderr) = run(&["models", "fetch", "--size", "small"], &models);

    // No network was touched: the licence gate comes first.
    assert!(!ok);
    assert!(stderr.contains("non-commercially"), "got: {stderr}");
    assert!(stderr.contains("accept-licence"), "the fix should be named");
}

#[test]
fn accepting_the_licence_is_recorded_and_reported() {
    let models = scratch("accept");

    let (ok, _, stderr) = run(&["models", "fetch", "--size", "small"], &models);
    assert!(!ok, "not accepted yet");

    let (ok, stdout, _) = run(&["accept-licence"], &models);
    assert!(ok, "accepting should succeed: {stderr}");
    assert!(stdout.contains("non-commercially"), "got: {stdout}");

    // The record lands in the cache directory, not somewhere global.
    let marker = models.join("LICENCE-ACCEPTED");
    assert!(marker.exists(), "no marker at {}", marker.display());
    let contents = std::fs::read_to_string(&marker).unwrap();
    assert!(contents.contains("CC BY-NC 4.0"), "got: {contents}");
}

#[test]
fn fetch_rejects_all_for_a_single_model() {
    let models = scratch("fetch-all");
    let (ok, _, stderr) = run(&["models", "fetch", "--size", "all"], &models);

    assert!(!ok);
    assert!(stderr.contains("one model"), "got: {stderr}");
}
