//! Ladder step 4: a whole file's notes, against the reference.
//!
//! Steps 1 to 3 live in `neunote-engine`, where the model is. This one needs
//! everything around it too -- chunking, prelude forcing, the cross-chunk note
//! tracker, the assembler -- so it runs the pipeline the binary runs.
//!
//! The reference's `transcribe.notes` covers all fifteen seconds of the fixture:
//! three chunks, notes carried across both boundaries, and the tail that
//! `finish()` closes.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use neunote_pipeline::muscriptor::Muscriptor;
use neunote_pipeline::{self as pipeline, Outcome};
use neunote_types::{ModelSize, NoteEvent};

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the crate lives two levels below the workspace root")
        .to_path_buf()
}

fn weights() -> PathBuf {
    let name = "muscriptor-small-f16.gguf";
    if let Some(dir) = std::env::var_os("NEUNOTE_WEIGHTS_DIR") {
        return PathBuf::from(dir).join(name);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        let linux = home.join(".local/share/neunote/models").join(name);
        let macos = home
            .join("Library/Application Support/neunote/models")
            .join(name);
        for candidate in [&linux, &macos] {
            if candidate.is_file() {
                return candidate.clone();
            }
        }
        return linux;
    }
    if let Some(appdata) = std::env::var_os("APPDATA") {
        return PathBuf::from(appdata).join("neunote/models").join(name);
    }
    PathBuf::from(name)
}

fn refs() -> Vec<f32> {
    let path = workspace().join("testdata/refs/small.bin");
    assert!(
        path.exists(),
        "the reference dump is not at {}",
        path.display()
    );

    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"NNEEDMP\0");
    let mut at = 8usize;

    let count = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    at += 4;

    for _ in 0..count {
        let name_len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        let name = String::from_utf8(bytes[at..at + name_len].to_vec()).unwrap();
        at += name_len;

        let n_dims = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()));
            at += 8;
        }

        let elements = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
        at += 8;

        if name == "transcribe.notes" {
            assert_eq!(shape.len(), 2, "five fields per note");
            assert_eq!(shape[1], 5);
            assert_eq!(elements as u64, shape[0] * 5);
            return bytes[at..at + elements as usize * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect();
        }

        at += elements * 4;
    }

    panic!("the dump has no transcribe.notes");
}

/// The fixture, as 16 kHz mono f32.
fn fixture() -> Vec<f32> {
    let bytes = std::fs::read(workspace().join("testdata/audio/fixture_3chunks_16k.wav")).unwrap();

    let mut at = 12usize;
    let mut format = 0u16;
    let mut samples = Vec::new();

    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;

        if id == b"fmt " {
            format = u16::from_le_bytes(bytes[at + 8..at + 10].try_into().unwrap());
        } else if id == b"data" {
            let raw = &bytes[at + 8..at + 8 + size];
            samples = match format {
                1 => raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|word| i16::from_le_bytes(*word) as f32 / 32768.0)
                    .collect(),
                _ => raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| f32::from_le_bytes(*word))
                    .collect(),
            };
        }

        at += 8 + size + (size & 1);
    }

    samples
}

/// One transcribed note from the reference: onset, offset, pitch, program, drum.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Reference {
    onset: f64,
    offset: f64,
    pitch: i32,
    program: u16,
    is_drum: bool,
}

fn transcribe_fixture() -> Option<Vec<NoteEvent>> {
    let path = weights();
    if !path.exists() {
        eprintln!(
            "skipping: no small checkpoint at {}, fetch it with `neunote models fetch --size small`",
            path.display()
        );
        return None;
    }

    let mut engine = Muscriptor::load(&path).expect("loading the checkpoint");
    let samples = fixture();
    assert_eq!(samples.len(), 240_000, "15 s at 16 kHz");

    match pipeline::transcribe_with(
        &mut engine,
        &samples,
        ModelSize::Small,
        &[],
        true,
        &AtomicBool::new(false),
        |_| {},
    )
    .expect("transcribing the fixture")
    {
        Outcome::Finished(notes) => Some(notes),
        Outcome::Cancelled => panic!("unexpectedly cancelled"),
    }
}

#[test]
fn step_4_the_notes_match() {
    let Some(notes) = transcribe_fixture() else {
        return;
    };
    let flat = refs();
    let want: Vec<Reference> = flat
        .as_chunks::<5>()
        .0
        .iter()
        .map(|note| Reference {
            onset: f64::from(note[0]),
            offset: f64::from(note[1]),
            pitch: note[2] as i32,
            program: note[3] as u16,
            is_drum: note[4] != 0.0,
        })
        .collect();

    assert_eq!(
        notes.len(),
        want.len(),
        "the port produced {} notes, the reference {}",
        notes.len(),
        want.len()
    );

    // The count is asserted above, so this is not a prefix check: every note the
    // reference produced is compared, including the tail `finish()` closed.
    assert_eq!(notes.len(), 451, "15 s is three chunks plus a tail");

    for (index, (got, want)) in notes.iter().zip(&want).enumerate() {
        // The reference widens short or inverted notes to its own 10 ms floor
        // and trims overlaps, so an onset lands on the 10 ms grid and an offset
        // is allowed the same slack. Anything else would be a real difference.
        assert!(
            (got.onset - want.onset).abs() <= 0.01 + 1e-6,
            "note {index} onset {} against {}",
            got.onset,
            want.onset
        );
        assert!(
            (got.offset - want.offset).abs() <= 0.01 + 1e-6,
            "note {index} offset {} against {}",
            got.offset,
            want.offset
        );
        assert_eq!(i32::from(got.pitch), want.pitch, "note {index} pitch");
        assert_eq!(got.program, want.program, "note {index} program");
        assert_eq!(got.is_drum, want.is_drum, "note {index} is_drum");
    }
}
