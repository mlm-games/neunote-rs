//! The pipeline driven end to end, with a scripted engine standing in for the
//! model, and the real MIDI writer on the other end.
//!
//! This is the closest thing to a working `transcribe` that can exist before the
//! inference engine does: real audio in, real chunking, real cross-chunk state,
//! real note assembly, real MIDI out.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use neunote_pipeline::{self as pipeline, Chunk, ChunkRequest, Engine, Outcome, Stop};
use neunote_types::{DRUM_PROGRAM, GroupId, ModelSize, NoteEvent, SEGMENT_SAMPLES};

const TIE: i32 = neunote_tokenizer::TIE_FIRST_ID;
const EOS: i32 = neunote_tokenizer::EOS_ID;

fn token(kind: neunote_tokenizer::EventType, value: i32) -> i32 {
    neunote_tokenizer::token_for(kind, value).unwrap()
}
fn program(value: i32) -> i32 {
    token(neunote_tokenizer::EventType::Program, value)
}
fn pitch(value: i32) -> i32 {
    token(neunote_tokenizer::EventType::Pitch, value)
}
fn velocity(on: bool) -> i32 {
    token(neunote_tokenizer::EventType::Velocity, i32::from(on))
}
fn shift(steps: i32) -> i32 {
    neunote_tokenizer::SHIFT_FIRST + steps
}
fn drum(value: i32) -> i32 {
    token(neunote_tokenizer::EventType::Drum, value)
}

/// Replays a fixed token stream per chunk.
struct Scripted {
    chunks: Vec<Vec<i32>>,
    next: usize,
    segment: usize,
}

impl Scripted {
    fn new(chunks: Vec<Vec<i32>>) -> Self {
        Self {
            chunks,
            next: 0,
            segment: SEGMENT_SAMPLES,
        }
    }
}

impl Engine for Scripted {
    fn segment_samples(&self) -> usize {
        self.segment
    }

    fn generate(&mut self, request: ChunkRequest<'_>) -> Result<Chunk, String> {
        // The pipeline must hand over exactly one padded chunk every time.
        assert_eq!(request.samples.len(), self.segment, "chunk length");
        // A conforming engine returns the forced prompt inside its own stream.
        let mut tokens = request.prompt.to_vec();
        tokens.extend(self.chunks.get(self.next).cloned().unwrap_or_default());
        self.next += 1;
        Ok(Chunk {
            tokens,
            stop: Stop::Eos,
        })
    }
}

fn audio(chunks: usize) -> Vec<f32> {
    vec![0.1; SEGMENT_SAMPLES * chunks]
}

fn run(engine: &mut Scripted, samples: &[f32]) -> Vec<NoteEvent> {
    match pipeline::transcribe_with(
        engine,
        samples,
        ModelSize::Medium,
        &[],
        true,
        &AtomicBool::new(false),
        |_| {},
    )
    .unwrap()
    {
        Outcome::Finished(notes) => notes,
        Outcome::Cancelled => panic!("unexpectedly cancelled"),
    }
}

fn write_midi(notes: &[NoteEvent], name: &str) -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(name);
    neunote_midi::write_midi_file(&path, notes, 120.0).unwrap();
    std::fs::read(path).unwrap()
}

fn instruments_of(bytes: &[u8]) -> Vec<String> {
    let file = midly::Smf::parse(bytes).expect("the writer produced a valid file");
    file.tracks
        .iter()
        .skip(1)
        .filter_map(|track| {
            track.iter().find_map(|event| match &event.kind {
                midly::TrackEventKind::Midi {
                    message: midly::MidiMessage::ProgramChange { program },
                    ..
                } => Some(format!("program {}", u8::from(*program))),
                _ => None,
            })
        })
        .collect()
}

/// The channel of the track that carries no program change, which is the drum
/// track: the percussion channel does not select a program.
fn drum_track_channel(bytes: &[u8]) -> Option<u8> {
    let file = midly::Smf::parse(bytes).unwrap();
    file.tracks
        .iter()
        .skip(1)
        .find(|track| {
            !track.iter().any(|event| {
                matches!(
                    &event.kind,
                    midly::TrackEventKind::Midi {
                        message: midly::MidiMessage::ProgramChange { .. },
                        ..
                    }
                )
            })
        })
        .and_then(|track| {
            track.iter().find_map(|event| match &event.kind {
                midly::TrackEventKind::Midi {
                    channel,
                    message: midly::MidiMessage::NoteOn { .. },
                } => Some(u8::from(*channel)),
                _ => None,
            })
        })
}

#[test]
fn a_three_chunk_piano_phrase_becomes_a_valid_midi_file() {
    // Three chunks of notes, none closing across a boundary.
    let mut engine = Scripted::new(vec![
        vec![
            TIE,
            shift(10),
            program(0),
            velocity(true),
            pitch(60),
            shift(20),
            velocity(false),
            pitch(60),
            EOS,
        ],
        vec![
            TIE,
            shift(10),
            program(0),
            velocity(true),
            pitch(62),
            shift(20),
            velocity(false),
            pitch(62),
            EOS,
        ],
        vec![TIE, shift(10), program(0), velocity(true), pitch(64), EOS],
    ]);
    let notes = run(&mut engine, &audio(3));

    assert_eq!(notes.len(), 3);
    assert_eq!(
        notes.iter().map(|n| n.pitch).collect::<Vec<_>>(),
        vec![60, 62, 64]
    );

    let bytes = write_midi(&notes, "piano.mid");
    assert_eq!(instruments_of(&bytes), vec!["program 0"]);
    assert!(midly::Smf::parse(&bytes).is_ok());
}

#[test]
fn a_mixed_instrument_phrase_lands_in_separate_tracks() {
    let mut engine = Scripted::new(vec![vec![
        TIE,
        shift(10),
        program(0),
        velocity(true),
        pitch(60),
        shift(12),
        program(40),
        velocity(true),
        pitch(67),
        shift(14),
        drum(36),
        shift(16),
        velocity(false),
        pitch(60),
        shift(18),
        velocity(false),
        pitch(67),
        EOS,
    ]]);
    let notes = run(&mut engine, &audio(1));

    assert_eq!(notes.len(), 3);

    let bytes = write_midi(&notes, "mixed.mid");
    let file = midly::Smf::parse(&bytes).unwrap();

    // Conductor, piano, violin, drums.
    assert_eq!(file.tracks.len(), 4);
    assert_eq!(instruments_of(&bytes), vec!["program 0", "program 40"]);
    assert_eq!(drum_track_channel(&bytes), Some(9));
}

#[test]
fn a_note_played_across_two_chunks_survives_into_the_midi() {
    // Starts in chunk 0, closes in chunk 2, carried through chunk 1's prompt.
    let mut engine = Scripted::new(vec![
        vec![TIE, shift(10), program(0), velocity(true), pitch(72), EOS],
        vec![TIE, EOS],
        vec![TIE, shift(50), velocity(false), pitch(72), EOS],
    ]);
    let notes = run(&mut engine, &audio(3));

    assert_eq!(notes.len(), 1, "one note, three chunks");
    assert!((notes[0].onset - 0.1).abs() < 1e-9);
    assert!(
        (notes[0].offset - 10.5).abs() < 1e-9,
        "closed in the third chunk"
    );

    let bytes = write_midi(&notes, "carried.mid");
    assert!(midly::Smf::parse(&bytes).is_ok());
}

#[test]
fn silence_writes_a_file_with_no_note_tracks() {
    let mut engine = Scripted::new(vec![vec![TIE, EOS]; 3]);
    let notes = run(&mut engine, &audio(3));

    assert!(notes.is_empty());

    // A valid file, just a conductor track.
    let bytes = write_midi(&notes, "silence.mid");
    let file = midly::Smf::parse(&bytes).unwrap();
    assert_eq!(file.tracks.len(), 1);
}

#[test]
fn an_instrument_restriction_narrows_the_output() {
    // The engine ignores the mask here, so this checks the mask reaches it and
    // that a piano-only run does not end up with other instruments.
    let mut engine = Scripted::new(vec![vec![
        TIE,
        shift(10),
        program(0),
        velocity(true),
        pitch(60),
        shift(12),
        velocity(false),
        pitch(60),
        EOS,
    ]]);

    let notes = match pipeline::transcribe_with(
        &mut engine,
        &audio(1),
        ModelSize::Medium,
        &[GroupId(0)],
        true,
        &AtomicBool::new(false),
        |_| {},
    )
    .unwrap()
    {
        Outcome::Finished(notes) => notes,
        Outcome::Cancelled => panic!("unexpectedly cancelled"),
    };

    assert!(notes.iter().all(|note| note.program == 0));
}

#[test]
fn cancelling_midway_leaves_no_file_behind() {
    let mut engine = Scripted::new(vec![vec![TIE, EOS]; 4]);
    let cancel = AtomicBool::new(false);

    let outcome = pipeline::transcribe_with(
        &mut engine,
        &audio(4),
        ModelSize::Medium,
        &[],
        true,
        &cancel,
        |progress| {
            if progress.chunks_done == 2 {
                cancel.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        },
    )
    .unwrap();

    assert!(matches!(outcome, Outcome::Cancelled));
}

#[test]
fn a_model_error_stops_the_run_without_a_result() {
    struct Failing;

    impl Engine for Failing {
        fn segment_samples(&self) -> usize {
            SEGMENT_SAMPLES
        }

        fn generate(&mut self, _: ChunkRequest<'_>) -> Result<Chunk, String> {
            Err("out of memory".to_owned())
        }
    }

    let cancel = AtomicBool::new(false);
    let error = pipeline::transcribe_with(
        &mut Failing,
        &audio(1),
        ModelSize::Medium,
        &[],
        true,
        &cancel,
        |_| {},
    )
    .unwrap_err();

    assert!(error.contains("out of memory"), "got {error}");
}

#[test]
fn progress_reports_every_chunk_of_the_fixture_length() {
    // The reference fixture is 15 s: exactly three chunks.
    let mut engine = Scripted::new(vec![vec![TIE, EOS]; 3]);
    let mut seen = Vec::new();

    pipeline::transcribe_with(
        &mut engine,
        &audio(3),
        ModelSize::Medium,
        &[],
        true,
        &AtomicBool::new(false),
        |progress| seen.push(progress),
    )
    .unwrap();

    assert_eq!(seen.len(), 3);
    assert_eq!(seen[0].chunks_total, 3);
    assert_eq!(seen[2].chunks_done, 3);
    assert!((seen[2].fraction() - 1.0).abs() < 1e-9);
    assert!((seen[2].finalized_through - 10.0).abs() < 1e-9);
}

#[test]
fn drums_and_melody_do_not_collide_on_one_pitch() {
    // Pitch 36 as a drum and pitch 36 as a melodic note are separate voices and
    // must not trim each other.
    let mut engine = Scripted::new(vec![vec![
        TIE,
        shift(10),
        program(128),
        velocity(true),
        pitch(36),
        shift(12),
        drum(36),
        shift(20),
        velocity(false),
        pitch(36),
        EOS,
    ]]);
    let notes = run(&mut engine, &audio(1));

    assert_eq!(notes.len(), 2);
    let melodic = notes.iter().find(|n| !n.is_drum).unwrap();
    let percussive = notes.iter().find(|n| n.is_drum).unwrap();
    assert_eq!(melodic.pitch, 36);
    assert_eq!(melodic.program, 128, "program 128 is a melodic note here");
    assert_eq!(percussive.program, DRUM_PROGRAM);

    let bytes = write_midi(&notes, "collision.mid");
    assert!(midly::Smf::parse(&bytes).is_ok());
}

#[test]
fn a_long_file_scales_to_many_chunks() {
    let chunks = 60;
    let mut engine = Scripted::new(vec![vec![TIE, EOS]; chunks]);
    let mut seen = Vec::new();

    pipeline::transcribe_with(
        &mut engine,
        &audio(chunks),
        ModelSize::Medium,
        &[],
        true,
        &AtomicBool::new(false),
        |progress| seen.push(progress),
    )
    .unwrap();

    assert_eq!(seen.len(), chunks);
    assert_eq!(engine.next, chunks);
}

#[test]
fn a_checkpoint_that_cannot_be_loaded_names_its_path() {
    // The engine is built now, so the failure a user sees is a real one: the
    // file is missing, unreadable, or not a checkpoint. Either way the path has
    // to be in the message.
    let error = match neunote_pipeline::muscriptor::Muscriptor::load(Path::new("/nowhere/x.gguf")) {
        Err(error) => error,
        Ok(_) => panic!("a path that does not exist must not load"),
    };
    assert!(error.contains("/nowhere/x.gguf"), "got {error}");
}

#[test]
fn a_file_that_is_not_a_checkpoint_is_refused_rather_than_read_as_one() {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), b"not a checkpoint").unwrap();

    let error = match neunote_pipeline::muscriptor::Muscriptor::load(file.path()) {
        Err(error) => error,
        Ok(_) => panic!("a file that is not a checkpoint must not load"),
    };
    assert!(
        error.contains("checkpoint") || error.contains("GGUF"),
        "got {error}"
    );
}
