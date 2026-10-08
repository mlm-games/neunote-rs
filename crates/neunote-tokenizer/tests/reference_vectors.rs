//! Replays every vector in the reference's own decode suite.
//!
//! The vectors are vendored from `muscriptor.cpp` `testdata/vectors/`, so
//! passing all 17 means this port agrees with `OpenNoteTracker`,
//! `NoteAssembler` and `Vocabulary` exactly, with no model weights involved.

use std::path::Path;

use neunote_tokenizer::{
    ActionKind, ChunkBoundary, EventType, NoteAction, NoteAssembler, OpenNoteTracker,
    tie_section_tokens,
};
use neunote_types::{NoteEvent, NoteKey, SEGMENT_DURATION_SECS};
use serde::Deserialize;

#[derive(Deserialize)]
struct VectorFile {
    provenance: Provenance,
    vectors: Vec<Vector>,
}

#[derive(Deserialize)]
struct Provenance {
    frame_rate: i32,
    instrument_vocabulary: String,
    max_shift_steps: i32,
    minimum_note_duration_sec: f64,
}

#[derive(Deserialize)]
struct Vector {
    name: String,
    seek_times: Vec<f64>,
    chunk_tokens: Vec<Vec<i32>>,
    actions: Vec<ExpectedAction>,
    notes: Vec<ExpectedNote>,
    open_keys_at_boundary: Vec<Vec<ExpectedKey>>,
}

#[derive(Deserialize, PartialEq)]
struct ExpectedAction {
    kind: String,
    pitch: i32,
    #[serde(default)]
    program: i32,
    time: f64,
}

#[derive(Deserialize)]
struct ExpectedNote {
    onset: f64,
    offset: f64,
    pitch: u8,
    program: u16,
    is_drum: bool,
}

#[derive(Deserialize)]
struct ExpectedKey {
    program: i32,
    pitch: i32,
}

fn load() -> VectorFile {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vectors/note_vectors.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    serde_json::from_str(&raw).expect("parsing note_vectors.json")
}

fn kind_name(kind: ActionKind) -> &'static str {
    match kind {
        ActionKind::Start => "start",
        ActionKind::End => "end",
        ActionKind::DrumHit => "drum",
    }
}

/// Decodes every chunk of one vector, returning the flat action stream and the
/// open keys at each chunk boundary.
fn decode(vector: &Vector) -> (Vec<NoteAction>, Vec<Vec<NoteKey>>) {
    let mut tracker = OpenNoteTracker::new();
    let mut actions = Vec::new();
    let mut boundaries = Vec::new();

    let last = vector.chunk_tokens.len().saturating_sub(1);
    for (index, (seek_time, tokens)) in vector
        .seek_times
        .iter()
        .copied()
        .zip(&vector.chunk_tokens)
        .enumerate()
    {
        // The final chunk has no window: the model is free to run past it.
        let next_seek_time = (index != last).then_some(seek_time + SEGMENT_DURATION_SECS);
        let boundary = ChunkBoundary {
            seek_time,
            next_seek_time,
        };

        actions.extend(tracker.feed_boundary(boundary));
        boundaries.push(tracker.open_keys());

        for token in tokens {
            actions.extend(tracker.feed_token(*token));
        }
    }

    actions.extend(tracker.finish());
    (actions, boundaries)
}

fn notes_of(vector: &Vector) -> Vec<NoteEvent> {
    let mut tracker = OpenNoteTracker::new();
    let mut assembler = NoteAssembler::new();
    let last = vector.chunk_tokens.len().saturating_sub(1);

    for (index, (seek_time, tokens)) in vector
        .seek_times
        .iter()
        .copied()
        .zip(&vector.chunk_tokens)
        .enumerate()
    {
        let next_seek_time = (index != last).then_some(seek_time + SEGMENT_DURATION_SECS);
        let emitted = tracker.feed_boundary(ChunkBoundary {
            seek_time,
            next_seek_time,
        });
        assembler
            .apply(&emitted, index as u32)
            .unwrap_or_else(|error| panic!("{}: {error}", vector.name));

        for token in tokens {
            let emitted = tracker.feed_token(*token);
            assembler
                .apply(&emitted, index as u32)
                .unwrap_or_else(|error| panic!("{}: {error}", vector.name));
        }
    }

    let emitted = tracker.finish();
    assembler
        .apply(&emitted, last as u32)
        .unwrap_or_else(|error| panic!("{}: {error}", vector.name));

    assembler.finalize()
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "{what}: got {actual}, want {expected}"
    );
}

#[test]
fn provenance_constants_match_the_reference() {
    let file = load();
    assert_eq!(file.provenance.frame_rate, neunote_types::FRAME_RATE);
    assert_eq!(
        file.provenance.max_shift_steps,
        neunote_tokenizer::MAX_SHIFT_STEPS
    );
    assert_close(
        file.provenance.minimum_note_duration_sec,
        neunote_types::MIN_NOTE_DURATION_SECS,
        "minimum_note_duration_sec",
    );
    assert_eq!(file.provenance.instrument_vocabulary, "MT3_FULL_PLUS");
}

#[test]
fn the_suite_covers_every_decoding_rule() {
    let file = load();
    let names: Vec<&str> = file.vectors.iter().map(|v| v.name.as_str()).collect();
    for required in [
        "shift_zero_is_noop",
        "shift_is_absolute_within_chunk",
        "window_drop_and_last_chunk_exemption",
        "malformed_chunk_closes_all_and_skips_rest",
        "eos_in_prologue_closes_all",
        "tie_prologue_partial",
        "tie_prologue_two_programs",
        "retrigger",
        "retrigger_zero_length_is_dropped",
        "pitch_without_registers_is_ignored",
        "drum_ignores_velocity_and_program",
        "drum_duplicate_at_same_tick",
        "program_96_decodes_as_drums",
        "finish_minimum_duration",
        "insertion_order_survives_to_finish",
        "trim_out_of_order_shifts",
        "empty_chunk",
    ] {
        assert!(names.contains(&required), "missing vector {required}");
    }
}

#[test]
fn every_vector_reproduces_the_reference_action_stream() {
    let file = load();
    assert_eq!(file.vectors.len(), 17);

    for vector in &file.vectors {
        let (actions, _) = decode(vector);

        assert_eq!(
            actions.len(),
            vector.actions.len(),
            "{}: action count",
            vector.name
        );

        for (index, (actual, expected)) in actions.iter().zip(&vector.actions).enumerate() {
            assert_eq!(
                kind_name(actual.kind),
                expected.kind,
                "{}: action {index} kind",
                vector.name
            );
            assert_eq!(
                actual.pitch, expected.pitch,
                "{}: action {index} pitch",
                vector.name
            );
            assert_close(
                actual.time,
                expected.time,
                &format!("{}: action {index} time", vector.name),
            );

            // A drum hit carries no program, so the reference omits the field.
            if expected.kind != "drum" {
                assert_eq!(
                    actual.program, expected.program,
                    "{}: action {index} program",
                    vector.name
                );
            }
        }
    }
}

#[test]
fn every_vector_reproduces_the_reference_note_list() {
    let file = load();

    for vector in &file.vectors {
        let notes = notes_of(vector);
        assert_eq!(
            notes.len(),
            vector.notes.len(),
            "{}: note count",
            vector.name
        );

        for (index, (actual, expected)) in notes.iter().zip(&vector.notes).enumerate() {
            assert_close(
                actual.onset,
                expected.onset,
                &format!("{}: note {index} onset", vector.name),
            );
            assert_close(
                actual.offset,
                expected.offset,
                &format!("{}: note {index} offset", vector.name),
            );
            assert_eq!(
                actual.pitch, expected.pitch,
                "{}: note {index} pitch",
                vector.name
            );
            assert_eq!(
                actual.program, expected.program,
                "{}: note {index} program",
                vector.name
            );
            assert_eq!(
                actual.is_drum, expected.is_drum,
                "{}: note {index} is_drum",
                vector.name
            );
        }
    }
}

#[test]
fn every_vector_reproduces_open_keys_at_each_boundary() {
    let file = load();

    for vector in &file.vectors {
        let (_, boundaries) = decode(vector);
        assert_eq!(
            boundaries.len(),
            vector.open_keys_at_boundary.len(),
            "{}: boundary count",
            vector.name
        );

        for (index, (actual, expected)) in boundaries
            .iter()
            .zip(&vector.open_keys_at_boundary)
            .enumerate()
        {
            let expected: Vec<NoteKey> = expected
                .iter()
                .map(|key| NoteKey::new(key.program, key.pitch))
                .collect();
            assert_eq!(actual, &expected, "{}: open keys at {index}", vector.name);
        }
    }
}

#[test]
fn an_encoded_prologue_carries_open_notes_into_the_next_chunk() {
    // Prelude forcing teacher-forces `tie_section_tokens(open_keys)` as the next
    // chunk's prologue. It is a declaration, not an opening: fed to a tracker
    // that holds exactly those notes they survive, and any note left out is
    // closed at the chunk's seek time. That is the whole cross-chunk mechanism.
    let file = load();

    // Every non-empty boundary key set the suite ever records.
    let mut cases: Vec<(Vec<NoteKey>, f64)> = Vec::new();
    for vector in &file.vectors {
        for (index, keys) in vector.open_keys_at_boundary.iter().enumerate() {
            if keys.is_empty() {
                continue;
            }
            cases.push((
                keys.iter()
                    .map(|k| NoteKey::new(k.program, k.pitch))
                    .collect(),
                vector.seek_times[index],
            ));
        }
    }
    assert!(!cases.is_empty(), "suite records no carried notes");

    for (keys, seek_time) in cases {
        let held = keys.clone();

        // Open `held` plus one extra in a first chunk. The leading `tie` leaves
        // the prologue, so the pitch tokens below act as note-ons.
        let mut forced = held.clone();
        let extra = NoteKey::new(held[0].program, (held[0].pitch + 1) % 128);
        forced.push(extra);

        let mut tracker = OpenNoteTracker::new();
        tracker.feed_boundary(ChunkBoundary {
            seek_time: 0.0,
            next_seek_time: Some(seek_time),
        });
        tracker.feed_token(neunote_tokenizer::TIE_FIRST_ID);
        for key in &forced {
            tracker
                .feed_token(neunote_tokenizer::token_for(EventType::Program, key.program).unwrap());
            tracker.feed_token(neunote_tokenizer::token_for(EventType::Velocity, 1).unwrap());
            tracker.feed_token(neunote_tokenizer::token_for(EventType::Pitch, key.pitch).unwrap());
        }
        let mut expected = forced.clone();
        expected.sort_unstable();
        assert_eq!(tracker.open_keys(), expected);

        let mut actions = tracker.feed_boundary(ChunkBoundary {
            seek_time,
            next_seek_time: Some(seek_time + 5.0),
        });
        for token in tie_section_tokens(&held) {
            actions.extend(tracker.feed_token(token));
        }

        let mut expected = held.clone();
        expected.sort_unstable();
        assert_eq!(
            tracker.open_keys(),
            expected,
            "forced prologue changed the open set"
        );

        // The note the forced prologue omitted is closed exactly at the boundary.
        if !held.contains(&extra) {
            assert!(
                actions.iter().any(|action| {
                    action.kind == ActionKind::End
                        && action.program == extra.program
                        && action.pitch == extra.pitch
                        && (action.time - seek_time).abs() < 1e-9
                }),
                "omitted note was not closed at the boundary"
            );
        }
    }
}

#[test]
fn every_recorded_prologue_is_a_well_formed_tie_section() {
    // A well-formed prologue is program/pitch tokens closed by a `tie`, and
    // decoding it must leave the tracker in its body. A `shift` before the tie
    // is the malformed case by definition, so those chunks are excluded here
    // and covered by the malformed vector's own assertions instead.
    let file = load();

    let mut checked = 0;
    for vector in &file.vectors {
        for (index, tokens) in vector.chunk_tokens.iter().enumerate() {
            let Some(tie_at) = tokens
                .iter()
                .position(|token| *token == neunote_tokenizer::TIE_FIRST_ID)
            else {
                continue;
            };

            if tokens[..tie_at]
                .iter()
                .any(|token| neunote_tokenizer::event_for(*token).r#type == EventType::Shift)
            {
                continue;
            }

            let mut tracker = OpenNoteTracker::new();
            tracker.feed_boundary(ChunkBoundary {
                seek_time: vector.seek_times[index],
                next_seek_time: None,
            });

            for token in &tokens[..=tie_at] {
                let event = neunote_tokenizer::event_for(*token);
                assert!(
                    matches!(
                        event.r#type,
                        EventType::Program | EventType::Pitch | EventType::Tie
                    ),
                    "{}: chunk {index} prologue carried {:?}",
                    vector.name,
                    event.r#type
                );
                tracker.feed_token(*token);
            }

            // Past the tie the tracker is in its body, where a shift is legal.
            assert!(
                tracker
                    .feed_token(neunote_tokenizer::SHIFT_FIRST + 10)
                    .is_empty(),
                "{}: chunk {index} did not leave the prologue at its tie",
                vector.name
            );
            checked += 1;
        }
    }

    assert!(checked >= 10, "only {checked} prologues exercised");
}
