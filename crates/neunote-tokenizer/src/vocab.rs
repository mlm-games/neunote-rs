//! The MT3 vocabulary: a fixed arithmetic index<->event mapping.
//!
//! Despite the name the reference gives it, this is not a text tokenizer. There
//! is no BPE and no vocabulary file. `build_event_vocab(max_shift_steps=1001)`
//! concatenates contiguous ranges in a fixed order, so a token id is a position
//! in that concatenation and a handful of comparisons reproduce it exactly.

use neunote_types::NoteKey;

pub const MAX_SHIFT_STEPS: i32 = 1_001;

// Running offsets, so the structure stays visible and one wrong bound cannot
// hide. These mirror `Vocabulary` in cpp/src/vocabulary.hpp.
pub const PAD_ID: i32 = 0;
pub const EOS_ID: i32 = 1;
pub const UNK_ID: i32 = 2;

pub const SHIFT_FIRST: i32 = 3;
pub const SHIFT_COUNT: i32 = MAX_SHIFT_STEPS;

pub const PITCH_FIRST: i32 = SHIFT_FIRST + SHIFT_COUNT;
pub const PITCH_COUNT: i32 = 128;

pub const VELOCITY_FIRST: i32 = PITCH_FIRST + PITCH_COUNT;
pub const VELOCITY_COUNT: i32 = 2;

pub const TIE_FIRST: i32 = VELOCITY_FIRST + VELOCITY_COUNT;
pub const TIE_FIRST_ID: i32 = TIE_FIRST;
pub const TIE_COUNT: i32 = 1;

pub const PROGRAM_FIRST: i32 = TIE_FIRST + TIE_COUNT;

pub const PROGRAM_COUNT: i32 = 130;

pub const DRUM_FIRST: i32 = PROGRAM_FIRST + PROGRAM_COUNT;
pub const DRUM_COUNT: i32 = 128;

pub const NUM_TOKENS: i32 = DRUM_FIRST + DRUM_COUNT;

/// `initial_token_id = card` is the BOS token fed at prefill.
pub const BOS_ID: i32 = NUM_TOKENS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum EventType {
    Pad = 0,
    Eos,
    Unk,
    Shift,
    Pitch,
    Velocity,
    Tie,
    Program,
    Drum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenEvent {
    pub r#type: EventType,
    pub value: i32,
}

/// Range descriptors, so `event_for` and `token_for` cannot disagree.
const RANGES: [(EventType, i32, i32); 9] = [
    (EventType::Pad, PAD_ID, 1),
    (EventType::Eos, EOS_ID, 1),
    (EventType::Unk, UNK_ID, 1),
    (EventType::Shift, SHIFT_FIRST, SHIFT_COUNT),
    (EventType::Pitch, PITCH_FIRST, PITCH_COUNT),
    (EventType::Velocity, VELOCITY_FIRST, VELOCITY_COUNT),
    (EventType::Tie, TIE_FIRST, TIE_COUNT),
    (EventType::Program, PROGRAM_FIRST, PROGRAM_COUNT),
    (EventType::Drum, DRUM_FIRST, DRUM_COUNT),
];

/// Ids outside the vocabulary answer `Unk`, which the state machine ignores --
/// the same thing the reference does with a token it has no rule for.
pub fn event_for(token_id: i32) -> TokenEvent {
    for (r#type, first, count) in RANGES {
        if token_id >= first && token_id < first + count {
            return TokenEvent {
                r#type,
                value: token_id - first,
            };
        }
    }
    TokenEvent {
        r#type: EventType::Unk,
        value: 0,
    }
}

pub fn token_for(r#type: EventType, value: i32) -> Option<i32> {
    RANGES
        .iter()
        .find(|(candidate, _, _)| *candidate == r#type)
        .map(|(_, first, count)| (0..*count).contains(&value).then_some(first + value))
        .unwrap_or_default()
}

/// Encode a tie prologue declaring `open_keys` as still sounding:
/// `program p, pitch a, pitch b, program q, pitch c, ..., tie`, over the keys
/// sorted by (program, pitch), with one program token per run of pitches. An
/// empty set still yields the bare `tie`.
///
/// This is both what prelude forcing teacher-forces and what the training
/// encoder produces, which is why the two agree.
pub fn tie_section_tokens(open_keys: &[NoteKey]) -> Vec<i32> {
    let mut sorted = open_keys.to_vec();
    sorted.sort_unstable();

    let mut tokens = Vec::with_capacity(sorted.len() * 2 + 1);
    // Seeded with a value no program can take, so the first key always emits one.
    let mut program_state: Option<i32> = None;

    for key in sorted {
        if program_state != Some(key.program) {
            tokens.push(token_for(EventType::Program, key.program).unwrap_or(-1));
            program_state = Some(key.program);
        }
        tokens.push(token_for(EventType::Pitch, key.pitch).unwrap_or(-1));
    }

    tokens.push(TIE_FIRST_ID);
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_match_the_reference_layout() {
        assert_eq!(NUM_TOKENS, 1_393);
        assert_eq!(SHIFT_FIRST, 3);
        assert_eq!(PITCH_FIRST, 1_004);
        assert_eq!(VELOCITY_FIRST, 1_132);
        assert_eq!(TIE_FIRST, 1_134);
        assert_eq!(PROGRAM_FIRST, 1_135);
        assert_eq!(DRUM_FIRST, 1_265);
    }

    #[test]
    fn every_token_id_round_trips() {
        for id in 0..NUM_TOKENS {
            let event = event_for(id);
            // id 2 is the UNK token, which is genuinely Unk.
            assert!(event.r#type != EventType::Unk || id == UNK_ID, "id {id}");
            assert_eq!(token_for(event.r#type, event.value), Some(id), "id {id}");
        }
    }

    #[test]
    fn shift_is_a_step_count_not_milliseconds() {
        // 10 ms per step, so shift 491 is 4.91 s inside a 5 s chunk.
        assert_eq!(event_for(3).value, 0);
        assert_eq!(event_for(SHIFT_FIRST + 491).value, 491);
        assert_eq!(
            (event_for(SHIFT_FIRST + 491).value as f64) / neunote_types::FRAME_RATE as f64,
            4.91
        );
    }

    #[test]
    fn ids_outside_the_vocabulary_are_unk() {
        assert_eq!(event_for(NUM_TOKENS).r#type, EventType::Unk);
        assert_eq!(event_for(9_999).r#type, EventType::Unk);
        assert_eq!(event_for(-1).r#type, EventType::Unk);
    }

    #[test]
    fn tie_section_groups_pitches_under_one_program_token() {
        let keys = vec![
            NoteKey::new(0, 64),
            NoteKey::new(0, 60),
            NoteKey::new(33, 60),
        ];
        assert_eq!(
            tie_section_tokens(&keys),
            vec![
                token_for(EventType::Program, 0).unwrap(),
                token_for(EventType::Pitch, 60).unwrap(),
                token_for(EventType::Pitch, 64).unwrap(),
                token_for(EventType::Program, 33).unwrap(),
                token_for(EventType::Pitch, 60).unwrap(),
                TIE_FIRST_ID,
            ]
        );
        assert_eq!(tie_section_tokens(&[]), vec![TIE_FIRST_ID]);
    }
}
