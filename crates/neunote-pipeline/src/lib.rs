#![forbid(unsafe_code)]

//! The transcription pipeline: chunking, prelude forcing, note assembly.
//!
//! An engine produces token ids for one chunk at a time, so this crate
//! defines the seam it plugs into -- [`Engine`] -- and drives everything
//! around it: the chunk loop, the cross-chunk state machine, instrument
//! conditioning and note assembly. [`muscriptor`] is the MuScriptor
//! implementation of that seam, the one that runs real inference.
//!
//! Everything here is embeddable: the command line tool layers file decoding,
//! weight downloading and MIDI export on top, from outside this crate.

pub mod muscriptor;

use std::sync::atomic::{AtomicBool, Ordering};

use neunote_tokenizer::{
    ChunkBoundary, NUM_TOKENS, NoteAssembler, OpenNoteTracker, tie_section_tokens,
};
use neunote_types::{GroupId, ModelSize, NoteEvent, SEGMENT_DURATION_SECS};

/// How far a run has got, for the progress line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Progress {
    /// Chunks finished out of the total.
    pub chunks_done: usize,
    pub chunks_total: usize,
    /// Seconds of audio whose notes are final. Nothing before this may change.
    pub finalized_through: f64,
}

impl Progress {
    pub fn fraction(&self) -> f64 {
        if self.chunks_total == 0 {
            return 1.0;
        }
        self.chunks_done as f64 / self.chunks_total as f64
    }
}

/// Something a running transcription has to say.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// How far it has got.
    Progress(Progress),
    /// The notes the chunk that just finished finalized.
    Notes(Vec<NoteEvent>),
}

/// Why a run stopped.
#[derive(Debug)]
pub enum Outcome {
    Finished(Vec<NoteEvent>),
    Cancelled,
}

/// One chunk's worth of work.
pub struct ChunkRequest<'a> {
    /// A whole 5 s segment of 16 kHz mono audio, zero-padded. The padding is
    /// not masked away: the model sees trailing silence as audio, which is what
    /// the reference does and what its note dumps are recorded under.
    pub samples: &'a [f32],
    /// The forced tie prologue, empty on the first chunk and empty when forcing
    /// is off.
    pub prompt: &'a [i32],
    /// One class-embedding row per selected instrument. Empty means the
    /// unconditional path, which is a single null-class row -- not the same
    /// thing as selecting nothing, and the engine draws that distinction.
    pub instrument_rows: &'a [i32],
    /// Token ids the model may not produce. Empty means no restriction, which
    /// differs from restricting to nothing.
    pub forbidden: &'a [i32],
}

/// Why a chunk's generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    Eos,
    /// The token or context budget ran out. The reference treats this as a
    /// warning rather than a failure, so a stream without EOS is a real outcome
    /// and reporting it as a missing EOS would be a lie about the model.
    Budget,
}

/// What an engine returns for one chunk.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The forced prompt followed by the engine's own tokens, EOS included when
    /// one was produced.
    ///
    /// The prompt is inside the stream because the reference's own `generate`
    /// puts it there: the decode state machine has to see it to leave the tie
    /// prologue. The pipeline replays the returned stream exactly once, so an
    /// engine that also fed the prompt separately would double-feed every
    /// boundary.
    pub tokens: Vec<i32>,
    pub stop: Stop,
}

/// The seam the inference engine plugs into.
///
/// A model produces token ids for one chunk at a time. Everything after that --
/// decoding, carrying notes across chunk boundaries, assembling the note list
/// -- lives here and is engine-independent.
pub trait Engine: Send {
    /// The chunk sizes the model expects, in samples.
    fn segment_samples(&self) -> usize;

    /// Generate tokens for one chunk.
    fn generate(&mut self, request: ChunkRequest<'_>) -> Result<Chunk, String>;
}

/// Run a transcription over 16 kHz mono audio.
///
/// `forbidden` masks token ids the model may not produce. An empty slice means
/// no restriction, which is different from restricting to nothing: the reference
/// forbids every program and every drum for an empty *selection*, so the caller
/// must skip the mask entirely rather than pass an empty one.
pub fn transcribe_with(
    engine: &mut dyn Engine,
    samples: &[f32],
    model_size: ModelSize,
    instruments: &[GroupId],
    prelude_forcing: bool,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(Progress),
) -> Result<Outcome, String> {
    transcribe_streaming(
        engine,
        samples,
        model_size,
        instruments,
        prelude_forcing,
        cancel,
        |event| {
            if let Event::Progress(progress) = event {
                on_progress(progress);
            }
        },
    )
}

/// The same run, reporting what it has as it goes.
///
/// `Event::Notes` is a preview, not the answer: a note reported at the end of
/// chunk *n* can still be cut short by one that closes in chunk *n + 1*, so a
/// caller showing these replaces the list with `Outcome::Finished` rather than
/// adding to it. What it buys is a run that shows its work, and a cancelled run
/// that keeps everything it finished.
pub fn transcribe_streaming(
    engine: &mut dyn Engine,
    samples: &[f32],
    model_size: ModelSize,
    instruments: &[GroupId],
    prelude_forcing: bool,
    cancel: &AtomicBool,
    mut on_event: impl FnMut(Event),
) -> Result<Outcome, String> {
    let segment = engine.segment_samples();
    if segment == 0 {
        return Err("engine reports a zero-length segment".to_owned());
    }

    let forbidden = forbidden_for(model_size, instruments)?;
    let instrument_rows = neunote_tokenizer::conditioning_rows(instruments);

    let mut tracker = OpenNoteTracker::new();
    let mut assembler = NoteAssembler::new();
    let total = samples.len().div_ceil(segment).max(1);

    for chunk_index in 0..total {
        if cancel.load(Ordering::SeqCst) {
            return Ok(Outcome::Cancelled);
        }

        let start = chunk_index * segment;
        let chunk = samples
            .get(start..)
            .unwrap_or(&[])
            .iter()
            .copied()
            .take(segment)
            .collect::<Vec<f32>>();
        // The last chunk is zero-padded and the padding is not masked.
        let padded = padded_to(&chunk, segment);

        let seek_time = chunk_index as f64 * SEGMENT_DURATION_SECS;
        let next_seek_time = (chunk_index + 1 < total).then_some(seek_time + SEGMENT_DURATION_SECS);

        // Prelude forcing: at every boundary after the first, the notes still
        // sounding are encoded and teacher-forced rather than predicted. With it
        // off the model writes its own prologue, which the reference's token
        // dumps are recorded under.
        let prompt = if chunk_index == 0 || !prelude_forcing {
            Vec::new()
        } else {
            tie_section_tokens(&tracker.open_keys())
        };

        // The boundary is fed before generation so the tracker knows the window
        // the tokens will be checked against.
        let mut actions = tracker.feed_boundary(ChunkBoundary {
            seek_time,
            next_seek_time,
        });

        let chunk = engine.generate(ChunkRequest {
            samples: &padded,
            prompt: &prompt,
            instrument_rows: &instrument_rows,
            forbidden: &forbidden,
        })?;
        validate_tokens(&chunk.tokens)?;

        // The engine returns the forced prompt inside the stream it generated,
        // exactly as the reference's own `generate` does. Replaying it here as
        // well would feed every boundary twice and desynchronise the tracker.
        for token in &chunk.tokens {
            if *token == neunote_tokenizer::EOS_ID {
                break;
            }
            actions.extend(tracker.feed_token(*token));
        }

        assembler
            .apply(&actions, chunk_index as u32)
            .map_err(|error| format!("chunk {chunk_index}: {error}"))?;

        on_event(Event::Progress(Progress {
            chunks_done: chunk_index + 1,
            chunks_total: total,
            finalized_through: seek_time,
        }));

        // The run so far, not this chunk's piece of it: the view replaces what
        // it holds, so a delta would leave only the last chunk's notes and a
        // cancelled run would keep almost nothing. The final answer replaces
        // this list, so a note a later chunk still trims is fine here.
        let preview = assembler.finalize();
        if !preview.is_empty() {
            on_event(Event::Notes(preview));
        }
    }

    if cancel.load(Ordering::SeqCst) {
        return Ok(Outcome::Cancelled);
    }

    let emitted = tracker.finish();
    assembler
        .apply(&emitted, (total - 1) as u32)
        .map_err(|error| format!("final chunk: {error}"))?;

    Ok(Outcome::Finished(assembler.finalize()))
}

fn padded_to(samples: &[f32], len: usize) -> Vec<f32> {
    if samples.len() == len {
        return samples.to_vec();
    }
    let mut padded = samples.to_vec();
    padded.resize(len, 0.0);
    padded
}

/// Check a model's token stream for the shape the tracker needs: inside the
/// vocabulary.
///
/// Whether the chunk ended on EOS is the engine's `Stop`, not something to
/// infer here. Requiring EOS would turn a model that ran out of budget into an
/// error and, worse, invite a caller to append an EOS the model never emitted.
pub fn validate_tokens(tokens: &[i32]) -> Result<(), String> {
    for token in tokens {
        if *token < 0 || *token as usize >= NUM_TOKENS as usize {
            return Err(format!("token {token} is outside the vocabulary"));
        }
    }

    Ok(())
}

/// Vocabulary ids the model may not emit, whatever the instrument selection.
///
/// `logits[1393:]` is always negative infinity. For `small` that range is empty
/// -- its card is exactly the vocabulary -- while `medium` and `large` carry two
/// spare embedding rows that must never be produced.
pub fn always_forbidden(size: ModelSize) -> Vec<i32> {
    neunote_types::logits_mask_start(size)
        .map(|id| id as i32)
        .collect()
}

/// Merge the vocabulary's hard mask with any instrument selection.
pub fn forbidden_for(size: ModelSize, instruments: &[GroupId]) -> Result<Vec<i32>, String> {
    let mut ids = always_forbidden(size);
    if !instruments.is_empty() {
        ids.extend(neunote_tokenizer::forbidden_token_ids(instruments)?);
        ids.sort_unstable();
        ids.dedup();
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use neunote_types::{NoteKey, SEGMENT_SAMPLES};

    /// An engine that returns scripted token streams, one per chunk.
    struct Scripted {
        segments: Vec<Vec<i32>>,
        prompts: Vec<Vec<i32>>,
        rows: Vec<Vec<i32>>,
        calls: usize,
    }

    impl Scripted {
        fn new(segments: Vec<Vec<i32>>) -> Self {
            Self {
                segments,
                prompts: Vec::new(),
                rows: Vec::new(),
                calls: 0,
            }
        }
    }

    impl Engine for Scripted {
        fn segment_samples(&self) -> usize {
            SEGMENT_SAMPLES
        }

        fn generate(&mut self, request: ChunkRequest<'_>) -> Result<Chunk, String> {
            self.prompts.push(request.prompt.to_vec());
            self.rows.push(request.instrument_rows.to_vec());
            // A conforming engine returns the forced prompt inside the stream,
            // then its own tokens.
            let mut tokens = request.prompt.to_vec();
            tokens.extend(self.segments.get(self.calls).cloned().unwrap_or_default());
            self.calls += 1;
            Ok(Chunk {
                tokens,
                stop: Stop::Eos,
            })
        }
    }

    fn program(value: i32) -> i32 {
        neunote_tokenizer::token_for(neunote_tokenizer::EventType::Program, value).unwrap()
    }
    fn pitch(value: i32) -> i32 {
        neunote_tokenizer::token_for(neunote_tokenizer::EventType::Pitch, value).unwrap()
    }
    fn velocity(on: bool) -> i32 {
        neunote_tokenizer::token_for(neunote_tokenizer::EventType::Velocity, i32::from(on)).unwrap()
    }
    fn shift(steps: i32) -> i32 {
        neunote_tokenizer::SHIFT_FIRST + steps
    }
    fn drum(value: i32) -> i32 {
        neunote_tokenizer::token_for(neunote_tokenizer::EventType::Drum, value).unwrap()
    }
    const TIE: i32 = neunote_tokenizer::TIE_FIRST_ID;
    const EOS: i32 = neunote_tokenizer::EOS_ID;

    fn audio(chunks: usize) -> Vec<f32> {
        vec![0.0; SEGMENT_SAMPLES * chunks]
    }

    fn run(engine: &mut Scripted, samples: &[f32]) -> Vec<NoteEvent> {
        let cancel = AtomicBool::new(false);
        match transcribe_with(
            engine,
            samples,
            ModelSize::Medium,
            &[],
            true,
            &cancel,
            |_| {},
        )
        .unwrap()
        {
            Outcome::Finished(notes) => notes,
            Outcome::Cancelled => panic!("unexpectedly cancelled"),
        }
    }

    #[test]
    fn one_chunk_produces_its_notes() {
        let mut engine = Scripted::new(vec![vec![
            TIE,
            shift(50),
            program(0),
            velocity(true),
            pitch(60),
            EOS,
        ]]);
        let notes = run(&mut engine, &audio(1));

        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].pitch, 60);
        assert!((notes[0].onset - 0.5).abs() < 1e-9);
        assert!((notes[0].offset - 0.51).abs() < 1e-9, "widened to 10 ms");
        assert!(!notes[0].is_drum);
    }

    #[test]
    fn silence_produces_no_notes() {
        let mut engine = Scripted::new(vec![vec![TIE, EOS]]);
        assert!(run(&mut engine, &audio(1)).is_empty());
    }

    #[test]
    fn the_first_chunk_is_never_given_a_forced_prompt() {
        let mut engine = Scripted::new(vec![vec![TIE, EOS], vec![TIE, EOS]]);
        run(&mut engine, &audio(2));
        assert!(engine.prompts[0].is_empty(), "chunk 0 has nothing to carry");
    }

    #[test]
    fn a_note_open_at_a_boundary_is_carried_into_the_next_prompt() {
        // Chunk 0 opens C4 and never closes it. Chunk 1 must therefore be
        // forced to declare it, which is the whole cross-chunk mechanism.
        let mut engine = Scripted::new(vec![
            vec![TIE, shift(10), program(0), velocity(true), pitch(60), EOS],
            vec![TIE, shift(20), velocity(false), pitch(60), EOS],
        ]);
        let notes = run(&mut engine, &audio(2));

        assert_eq!(engine.prompts[1], vec![program(0), pitch(60), TIE]);

        assert_eq!(notes.len(), 1, "one carried note, closed in chunk 1");
        assert!((notes[0].onset - 0.1).abs() < 1e-9);
        assert!((notes[0].offset - 5.2).abs() < 1e-9);
    }

    #[test]
    fn an_engine_that_forgets_to_declare_carried_notes_closes_them_at_the_boundary() {
        // The engine returns the forced prompt inside the stream it generates,
        // and the pipeline replays that stream exactly once. This engine
        // returns only its own tie, so it has declared nothing: the reference
        // closes whatever a chunk did not name.
        //
        // This is the test that pins the contract. If the pipeline fed the
        // prompt separately as well, the carried note would be declared anyway
        // and would survive to `finish` instead of closing here.
        struct Forgetful {
            calls: usize,
        }

        impl Engine for Forgetful {
            fn segment_samples(&self) -> usize {
                SEGMENT_SAMPLES
            }

            fn generate(&mut self, _: ChunkRequest<'_>) -> Result<Chunk, String> {
                self.calls += 1;
                let tokens = if self.calls == 1 {
                    vec![TIE, shift(10), program(0), velocity(true), pitch(60), EOS]
                } else {
                    vec![EOS]
                };
                Ok(Chunk {
                    tokens,
                    stop: Stop::Eos,
                })
            }
        }

        let mut engine = Forgetful { calls: 0 };
        let cancel = AtomicBool::new(false);
        let notes = match transcribe_with(
            &mut engine,
            &audio(2),
            ModelSize::Medium,
            &[],
            true,
            &cancel,
            |_| {},
        )
        .unwrap()
        {
            Outcome::Finished(notes) => notes,
            Outcome::Cancelled => panic!("unexpectedly cancelled"),
        };

        // A forced prompt is only authoritative because the engine replays it inside
        // the stream it returns. An engine that drops it has not declared the
        // carried note, so the tracker closes that note at the boundary.
        assert_eq!(notes.len(), 1);
        assert!((notes[0].onset - 0.1).abs() < 1e-9);
        assert!(
            (notes[0].offset - 5.0).abs() < 1e-9,
            "closed at the chunk-1 boundary, got {}",
            notes[0].offset
        );
    }

    #[test]
    fn with_prelude_forcing_off_no_prompt_is_ever_sent() {
        let mut engine = Scripted::new(vec![
            vec![TIE, shift(10), program(0), velocity(true), pitch(60), EOS],
            vec![TIE, shift(20), velocity(false), pitch(60), EOS],
        ]);
        let cancel = AtomicBool::new(false);

        match transcribe_with(
            &mut engine,
            &audio(2),
            ModelSize::Medium,
            &[],
            false,
            &cancel,
            |_| {},
        )
        .unwrap()
        {
            Outcome::Finished(_) => {}
            Outcome::Cancelled => panic!("unexpectedly cancelled"),
        }

        assert!(
            engine.prompts.iter().all(|prompt| prompt.is_empty()),
            "forcing off means the model writes its own prologue: {:?}",
            engine.prompts
        );
    }

    #[test]
    fn a_drum_hit_is_instant_and_lands_on_the_drum_program() {
        let mut engine = Scripted::new(vec![vec![TIE, shift(50), drum(36), EOS]]);
        let notes = run(&mut engine, &audio(1));

        assert_eq!(notes.len(), 1);
        assert!(notes[0].is_drum);
        assert_eq!(notes[0].program, neunote_types::DRUM_PROGRAM);
        assert!((notes[0].onset - 0.5).abs() < 1e-9);
        assert!((notes[0].offset - 0.51).abs() < 1e-9);
    }

    #[test]
    fn notes_spanning_two_chunks_survive_intact() {
        let mut engine = Scripted::new(vec![
            vec![TIE, shift(490), program(0), velocity(true), pitch(72), EOS],
            vec![TIE, shift(100), velocity(false), pitch(72), EOS],
        ]);
        let notes = run(&mut engine, &audio(2));

        assert_eq!(notes.len(), 1);
        assert!((notes[0].onset - 4.9).abs() < 1e-9);
        assert!((notes[0].offset - 6.0).abs() < 1e-9);
    }

    #[test]
    fn a_partial_final_chunk_is_zero_padded() {
        let mut engine = Scripted::new(vec![vec![TIE, EOS]]);
        // A fifth of a chunk.
        let samples = vec![0.0; SEGMENT_SAMPLES / 5];
        assert!(run(&mut engine, &samples).is_empty());
        assert_eq!(engine.calls, 1, "still one chunk");
    }

    #[test]
    fn empty_audio_still_runs_one_chunk_and_finishes() {
        let mut engine = Scripted::new(vec![vec![TIE, EOS]]);
        let notes = run(&mut engine, &[]);
        assert!(notes.is_empty());
        assert_eq!(engine.calls, 1);
    }

    #[test]
    fn cancelling_before_the_first_chunk_yields_no_notes() {
        let mut engine = Scripted::new(vec![vec![TIE, EOS]; 3]);
        let cancel = AtomicBool::new(true);

        let outcome = transcribe_with(
            &mut engine,
            &audio(3),
            ModelSize::Medium,
            &[],
            true,
            &cancel,
            |_| {},
        )
        .unwrap();
        assert!(matches!(outcome, Outcome::Cancelled));
        assert_eq!(engine.calls, 0, "no work was done");
    }

    #[test]
    fn cancelling_midway_keeps_what_finished() {
        let mut engine = Scripted::new(vec![vec![TIE, EOS]; 3]);
        let cancel = AtomicBool::new(false);
        let mut seen = Vec::new();

        let outcome = transcribe_with(
            &mut engine,
            &audio(3),
            ModelSize::Medium,
            &[],
            true,
            &cancel,
            |progress| {
                seen.push(progress);
                if progress.chunks_done == 2 {
                    cancel.store(true, Ordering::SeqCst);
                }
            },
        )
        .unwrap();

        assert!(matches!(outcome, Outcome::Cancelled));
        assert_eq!(engine.calls, 2, "the third chunk never ran");
        assert_eq!(seen.last().unwrap().chunks_done, 2);
        assert_eq!(seen.last().unwrap().chunks_total, 3);
    }

    #[test]
    fn streaming_reports_the_notes_as_they_are_finalized() {
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
                pitch(64),
                shift(20),
                velocity(false),
                pitch(64),
                EOS,
            ],
        ]);

        let mut seen: Vec<Vec<NoteEvent>> = Vec::new();
        let outcome = transcribe_streaming(
            &mut engine,
            &audio(2),
            ModelSize::Medium,
            &[],
            true,
            &AtomicBool::new(false),
            |event| {
                if let Event::Notes(notes) = event {
                    seen.push(notes);
                }
            },
        )
        .unwrap();

        // One note closed in chunk 0, one in chunk 1, and every report is the
        // run so far rather than that chunk's piece of it.
        assert_eq!(seen.len(), 2, "one report per chunk with notes in it");
        assert_eq!(seen[0].len(), 1, "the first chunk's note");
        assert_eq!(seen[0][0].pitch, 60);
        assert_eq!(
            seen[1].len(),
            2,
            "the run so far keeps the first chunk's note"
        );
        assert_eq!(seen[1][0].pitch, 60);
        assert_eq!(seen[1][1].pitch, 64);

        let Outcome::Finished(notes) = outcome else {
            panic!("a run that finished is not cancelled")
        };
        assert_eq!(seen[1], notes, "the last preview is the finished list");
    }

    #[test]
    fn a_cancelled_run_still_handed_over_what_it_finished() {
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
            vec![TIE, shift(10), program(0), velocity(true), pitch(64), EOS],
            vec![TIE, EOS],
        ]);
        let cancel = AtomicBool::new(false);
        let mut seen: Vec<NoteEvent> = Vec::new();

        let outcome = transcribe_streaming(
            &mut engine,
            &audio(3),
            ModelSize::Medium,
            &[],
            true,
            &cancel,
            |event| match event {
                Event::Progress(progress) if progress.chunks_done == 1 => {
                    cancel.store(true, Ordering::SeqCst);
                }
                Event::Notes(notes) => seen.extend(notes),
                _ => {}
            },
        )
        .unwrap();

        assert!(matches!(outcome, Outcome::Cancelled));
        assert_eq!(engine.calls, 1, "the next chunk never started");
        assert_eq!(seen.len(), 1, "the closed note from chunk 0 is kept");
        assert_eq!(seen[0].pitch, 60);
    }

    #[test]
    fn progress_is_reported_for_every_chunk_in_order() {
        let mut engine = Scripted::new(vec![vec![TIE, EOS]; 3]);
        let mut seen = Vec::new();
        transcribe_with(
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
        for (index, progress) in seen.iter().enumerate() {
            assert_eq!(progress.chunks_done, index + 1);
            assert_eq!(progress.chunks_total, 3);
            assert!((progress.finalized_through - index as f64 * 5.0).abs() < 1e-9);
            assert!((progress.fraction() - (index + 1) as f64 / 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn an_instrument_restriction_masks_program_tokens() {
        let forbidden = neunote_tokenizer::forbidden_token_ids(&[GroupId(0)]).unwrap();
        assert!(
            forbidden.contains(&program(40)),
            "violin's program is masked"
        );
        assert!(!forbidden.contains(&program(0)));
    }

    #[test]
    fn an_empty_selection_masks_nothing() {
        // "No restriction" and "restrict to nothing" are different. An empty
        // selection means no filter at all.
        assert!(neunote_tokenizer::forbidden_token_ids(&[]).is_err());
    }

    #[test]
    fn token_validation_only_cares_that_the_ids_are_real() {
        // A stream with no EOS is legitimate: the reference stops at its token
        // or context budget without one, and manufacturing an EOS would hide
        // that. `Stop` is where the distinction lives.
        assert!(validate_tokens(&[TIE, program(0), pitch(60)]).is_ok());
        assert!(validate_tokens(&[TIE, -1, EOS]).is_err());
        assert!(validate_tokens(&[TIE, NUM_TOKENS, EOS]).is_err());
        assert!(validate_tokens(&[TIE, NUM_TOKENS - 1, EOS]).is_ok());
    }

    #[test]
    fn a_chunk_that_runs_out_of_budget_still_decodes() {
        struct Budget;
        impl Engine for Budget {
            fn segment_samples(&self) -> usize {
                SEGMENT_SAMPLES
            }
            fn generate(&mut self, request: ChunkRequest<'_>) -> Result<Chunk, String> {
                Ok(Chunk {
                    tokens: request
                        .prompt
                        .iter()
                        .copied()
                        .chain([TIE, shift(10), program(0), velocity(true), pitch(60)])
                        .collect(),
                    stop: Stop::Budget,
                })
            }
        }

        let cancel = AtomicBool::new(false);
        let notes = match transcribe_with(
            &mut Budget,
            &audio(1),
            ModelSize::Medium,
            &[],
            true,
            &cancel,
            |_| {},
        )
        .unwrap()
        {
            Outcome::Finished(notes) => notes,
            Outcome::Cancelled => panic!("unexpectedly cancelled"),
        };

        assert_eq!(notes.len(), 1, "a stream without EOS is not an error");
    }

    #[test]
    fn the_selection_reaches_the_engine_as_conditioning_rows() {
        let mut engine = Scripted::new(vec![vec![TIE, EOS]]);
        let cancel = AtomicBool::new(false);
        transcribe_with(
            &mut engine,
            &audio(1),
            ModelSize::Medium,
            &[GroupId(0), GroupId(7)],
            true,
            &cancel,
            |_| {},
        )
        .unwrap();

        // Group g conditions on row g + 2, so piano and bass are rows 2 and 9.
        assert_eq!(engine.rows[0], vec![2, 9]);
    }

    #[test]
    fn an_empty_selection_asks_for_the_unconditional_path() {
        let mut engine = Scripted::new(vec![vec![TIE, EOS]]);
        let cancel = AtomicBool::new(false);
        transcribe_with(
            &mut engine,
            &audio(1),
            ModelSize::Medium,
            &[],
            true,
            &cancel,
            |_| {},
        )
        .unwrap();

        assert!(
            engine.rows[0].is_empty(),
            "no selection is not a conditioning row list; the engine draws the null row"
        );
    }

    #[test]
    fn ids_past_the_active_vocabulary_are_masked_only_when_the_card_has_room() {
        // small's card is exactly the vocabulary, so there is nothing to mask.
        assert_eq!(NUM_TOKENS, 1_393);
        assert!(always_forbidden(ModelSize::Small).is_empty());

        // medium and large carry two spare embedding rows that must never be
        // produced, or a note could come out of a program that does not exist.
        assert_eq!(always_forbidden(ModelSize::Medium), vec![1_393, 1_394]);
        assert_eq!(always_forbidden(ModelSize::Large), vec![1_393, 1_394]);
    }

    #[test]
    fn a_token_outside_the_vocabulary_is_an_error_rather_than_a_panic() {
        // A model that emits a nonsense id must not crash the pipeline, and
        // must not be silently swallowed either -- that would look like the
        // chunk simply decoded to nothing.
        let mut engine = Scripted::new(vec![vec![
            TIE,
            shift(10),
            program(0),
            velocity(true),
            pitch(60),
            9_999,
            EOS,
        ]]);
        let cancel = AtomicBool::new(false);
        let error = transcribe_with(
            &mut engine,
            &audio(1),
            ModelSize::Medium,
            &[],
            true,
            &cancel,
            |_| {},
        )
        .unwrap_err();
        assert!(error.contains("outside the vocabulary"), "got {error}");
    }

    #[test]
    fn the_instrument_mask_and_the_vocabulary_mask_both_apply() {
        let ids = forbidden_for(ModelSize::Medium, &[GroupId(0)]).unwrap();

        // The two spare embedding rows.
        assert!(ids.contains(&1_393));
        assert!(ids.contains(&1_394));
        // Every drum, since drums was not selected.
        assert!(ids.contains(&neunote_tokenizer::DRUM_FIRST));
        // Every program but the piano's.
        assert!(!ids.contains(&program(0)));
        assert!(ids.contains(&program(40)));
        // Never a timing, pitch, velocity or tie token.
        assert!(!ids.contains(&TIE));
        assert!(!ids.contains(&shift(1)));
        assert!(!ids.contains(&pitch(60)));
        assert!(!ids.contains(&velocity(true)));
    }

    #[test]
    fn no_selection_leaves_only_the_vocabulary_mask() {
        let ids = forbidden_for(ModelSize::Medium, &[]).unwrap();
        assert_eq!(ids, vec![1_393, 1_394]);
    }

    #[test]
    fn the_chunk_count_matches_the_audio_length() {
        let cancel = AtomicBool::new(false);
        for (chunks, expected) in [(1, 1), (2, 2), (3, 3), (5, 5)] {
            let mut engine = Scripted::new(vec![vec![TIE, EOS]; chunks]);
            let mut seen: Vec<usize> = Vec::new();
            transcribe_with(
                &mut engine,
                &audio(chunks),
                ModelSize::Medium,
                &[],
                true,
                &cancel,
                |progress| seen.push(progress.chunks_total),
            )
            .unwrap();
            assert_eq!(seen[0], expected);
        }
    }

    #[test]
    fn the_forced_prompt_matches_the_encoder_for_every_open_key() {
        // Two notes open under one program: one program token, then both
        // pitches, then the tie.
        let mut engine = Scripted::new(vec![
            vec![
                TIE,
                shift(10),
                program(0),
                velocity(true),
                pitch(60),
                pitch(64),
                EOS,
            ],
            vec![TIE, EOS],
        ]);
        run(&mut engine, &audio(2));

        assert_eq!(
            engine.prompts[1],
            vec![program(0), pitch(60), pitch(64), TIE]
        );
        assert_eq!(
            engine.prompts[1],
            tie_section_tokens(&[NoteKey::new(0, 60), NoteKey::new(0, 64),])
        );
    }
}
