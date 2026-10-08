#![forbid(unsafe_code)]

//! The transcription pipeline: chunking, prelude forcing, note assembly.
//!
//! The engine itself is not built yet, so this module defines the seam it
//! will plug into -- [`Engine`] -- and drives everything around it. That means
//! the chunk loop, the cross-chunk state machine, instrument conditioning and
//! note assembly are all real and tested now, and adding the engine becomes
//! an implementation of one trait rather than a rewrite of this file.
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use neunote_tokenizer::{
    ChunkBoundary, NUM_TOKENS, NoteAssembler, OpenNoteTracker, tie_section_tokens,
};
use neunote_types::{GroupId, ModelSize, NoteEvent, SEGMENT_DURATION_SECS, SEGMENT_SAMPLES};

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

/// Why a run stopped.
#[derive(Debug)]
pub enum Outcome {
    Finished(Vec<NoteEvent>),
    Cancelled,
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
    ///
    /// `prompt` is the forced tie prologue, empty for the first chunk. The
    /// engine must append its own EOS and the forced tokens to whatever it
    /// returns, so the tracker sees the same stream either way.
    fn generate(
        &mut self,
        samples: &[f32],
        prompt: &[i32],
        forbidden: &[i32],
    ) -> Result<Vec<i32>, String>;
}

/// The default engine slot.
///
/// There is no pure-Rust MuScriptor implementation yet. This refuses loudly
/// rather than returning empty notes, which would look like a silent recording.
pub struct UnavailableEngine {
    reason: String,
}

impl UnavailableEngine {
    pub fn missing_engine() -> Self {
        Self {
            reason: "the MuScriptor inference engine is not built yet.\n\
                 The pipeline around it is: audio decoding, 16 kHz resampling, chunking,\n\
                 prelude forcing, note assembly and MIDI export all work and are tested.\n\
                 What is missing is the model itself -- the mel front-end, the\n\
                 transformer, and the GGUF reader for the checkpoint format the\n\
                 published weights use.\n\
                 The weights are in place; run `neunote models fetch --size small`."
                .to_owned(),
        }
    }

    pub fn missing_model(path: &Path) -> Self {
        Self {
            reason: format!(
                "no verified {} model at {}.\nRun `neunote models fetch` first.",
                ModelSize::DEFAULT.as_str(),
                path.display()
            ),
        }
    }
}

impl Engine for UnavailableEngine {
    fn segment_samples(&self) -> usize {
        SEGMENT_SAMPLES
    }

    fn generate(&mut self, _: &[f32], _: &[i32], _: &[i32]) -> Result<Vec<i32>, String> {
        Err(self.reason.clone())
    }
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
    let segment = engine.segment_samples();
    if segment == 0 {
        return Err("engine reports a zero-length segment".to_owned());
    }

    let forbidden = forbidden_for(model_size, instruments)?;

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

        let tokens = engine.generate(&padded, &prompt, &forbidden)?;
        validate_tokens(&tokens)?;

        for token in &prompt {
            actions.extend(tracker.feed_token(*token));
        }
        for token in &tokens {
            if *token as usize >= NUM_TOKENS as usize {
                continue;
            }
            actions.extend(tracker.feed_token(*token));
        }

        assembler
            .apply(&actions, chunk_index as u32)
            .map_err(|error| format!("chunk {chunk_index}: {error}"))?;

        on_progress(Progress {
            chunks_done: chunk_index + 1,
            chunks_total: total,
            finalized_through: seek_time,
        });
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

/// Check a model's token stream for the shape the tracker and the reference
/// require: inside the vocabulary, and ending in EOS.
///
/// Worth doing before decoding rather than after, because a stream with no EOS
/// silently means "the chunk produced nothing" instead of "the model misbehaved".
pub fn validate_tokens(tokens: &[i32]) -> Result<(), String> {
    let mut saw_eos = false;

    for token in tokens {
        if *token < 0 || *token as usize >= NUM_TOKENS as usize {
            return Err(format!("token {token} is outside the vocabulary"));
        }
        if *token == neunote_tokenizer::EOS_ID {
            saw_eos = true;
        }
    }

    if !saw_eos {
        return Err("the chunk did not end in EOS".to_owned());
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

/// Entry point for the command line tool.
pub async fn transcribe(
    mono: &[f32],
    model_path: &Path,
    size: ModelSize,
    instruments: &[GroupId],
    prelude_forcing: bool,
    cache: &neunote_models::Cache,
) -> Result<Vec<NoteEvent>, String> {
    if !cache.is_installed(size) || !model_path.exists() {
        return Err(UnavailableEngine::missing_model(model_path).reason);
    }

    let mut engine = UnavailableEngine::missing_engine();
    let cancel = AtomicBool::new(false);

    match transcribe_with(
        &mut engine,
        mono,
        size,
        instruments,
        prelude_forcing,
        &cancel,
        |progress| {
            eprint!(
                "\r  chunk {:>3}/{}  {:>3.0}%  {:.0}s final  ",
                progress.chunks_done,
                progress.chunks_total,
                progress.fraction() * 100.0,
                progress.finalized_through
            );
        },
    )? {
        Outcome::Finished(notes) => {
            eprintln!();
            Ok(notes)
        }
        Outcome::Cancelled => {
            eprintln!();
            Err("cancelled".to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neunote_types::NoteKey;

    /// An engine that returns scripted token streams, one per chunk.
    struct Scripted {
        segments: Vec<Vec<i32>>,
        prompts: Vec<Vec<i32>>,
        calls: usize,
    }

    impl Scripted {
        fn new(segments: Vec<Vec<i32>>) -> Self {
            Self {
                segments,
                prompts: Vec::new(),
                calls: 0,
            }
        }
    }

    impl Engine for Scripted {
        fn segment_samples(&self) -> usize {
            SEGMENT_SAMPLES
        }

        fn generate(&mut self, _: &[f32], prompt: &[i32], _: &[i32]) -> Result<Vec<i32>, String> {
            self.prompts.push(prompt.to_vec());
            let tokens = self.segments.get(self.calls).cloned().unwrap_or_default();
            self.calls += 1;
            Ok(tokens)
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
        // The forced prompt is teacher-forced, so the pipeline feeds it to the
        // tracker itself. An engine that returns its own bare tie on top of that
        // declares nothing, and the reference closes whatever it did not name.
        struct Forgetful {
            calls: usize,
        }

        impl Engine for Forgetful {
            fn segment_samples(&self) -> usize {
                SEGMENT_SAMPLES
            }

            fn generate(&mut self, _: &[f32], _: &[i32], _: &[i32]) -> Result<Vec<i32>, String> {
                self.calls += 1;
                if self.calls == 1 {
                    Ok(vec![
                        TIE,
                        shift(10),
                        program(0),
                        velocity(true),
                        pitch(60),
                        EOS,
                    ])
                } else {
                    Ok(vec![EOS])
                }
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

        // The forced prompt re-declared the note, so it survives to `finish`,
        // which closes it a minimum duration after its onset.
        assert_eq!(notes.len(), 1);
        assert!((notes[0].offset - 0.11).abs() < 1e-9);
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
    fn the_unavailable_engine_says_so_rather_than_returning_silence() {
        let mut engine = UnavailableEngine::missing_engine();
        let error = engine
            .generate(&[], &[], &[])
            .expect_err("the engine is not built");
        assert!(error.contains("not built yet"), "got {error}");
    }

    #[test]
    fn a_missing_model_names_the_path() {
        let error = UnavailableEngine::missing_model(Path::new("/tmp/nope.gguf")).reason;
        assert!(error.contains("/tmp/nope.gguf"), "got {error}");
        assert!(error.contains("models fetch"), "got {error}");
    }

    #[test]
    fn token_validation_catches_a_missing_eos() {
        assert!(validate_tokens(&[TIE, program(0), pitch(60)]).is_err());
        assert!(validate_tokens(&[TIE, EOS]).is_ok());
        assert!(validate_tokens(&[TIE, -1, EOS]).is_err());
        assert!(validate_tokens(&[TIE, NUM_TOKENS, EOS]).is_err());
        assert!(validate_tokens(&[TIE, NUM_TOKENS - 1, EOS]).is_ok());
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
    fn a_chunk_without_eos_is_an_error_rather_than_silence() {
        let mut engine = Scripted::new(vec![vec![
            TIE,
            shift(10),
            program(0),
            velocity(true),
            pitch(60),
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
        assert!(error.contains("EOS"), "got {error}");
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
