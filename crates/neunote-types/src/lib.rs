#![forbid(unsafe_code)]

//! Shared types for the MuScriptor transcription pipeline.
//!
//! Every constant here is transcribed from the reference implementation and its
//! docs (`muscriptor.cpp` docs/MODEL.md, docs/TOKENIZER.md, `tokenizer/{mt3,
//! notes}.py` in `muscriptor/muscriptor`). Nothing in this crate is inferred:
//! the tables are the reference's tables.

use serde::{Deserialize, Serialize};

/// The model consumes 16 kHz mono f32 audio, in whole 5 s chunks.
pub const TRANSCRIPTION_SAMPLE_RATE: u32 = 16_000;
pub const SEGMENT_SAMPLES: usize = 80_000;
pub const SEGMENT_DURATION_SECS: f64 = 5.0;

/// The model's 10 ms grid, in frames per second.
pub const FRAME_RATE: i32 = 100;

/// Greedy decoding stops here, forced prompt included.
pub const MAX_TOKENS_PER_CHUNK: usize = 2_000;

/// `MT3_FULL_PLUS` has 1393 tokens, and that is the size of the vocabulary
/// itself: the token ids a model can produce.
pub const VOCAB_ACTIVE: usize = 1_393;

/// The embedding table's row count, which is the vocabulary size plus the BOS
/// row. `small` has a card of 1393; `medium` and `large` carry 1395.
pub const fn card_for(size: ModelSize) -> usize {
    match size {
        ModelSize::Small => VOCAB_ACTIVE,
        ModelSize::Medium | ModelSize::Large => VOCAB_ACTIVE + 2,
    }
}

/// `initial_token_id = card` is the BOS token fed at prefill.
pub const fn bos_id(size: ModelSize) -> u32 {
    card_for(size) as u32
}

/// Token ids at or above the active vocabulary, which `logits[1393:]` forces to
/// negative infinity (`Hparams::logit_mask_start`). For `medium` and `large`
/// that is ids 1393 and 1394, which their larger embedding table could
/// otherwise have produced.
pub fn logits_mask_start(size: ModelSize) -> std::ops::Range<u32> {
    VOCAB_ACTIVE as u32..card_for(size) as u32
}

/// The reference writes every note at MIDI velocity 100; the model predicts no
/// dynamics.
pub const MIDI_VELOCITY: u8 = 100;
pub const FIXED_NOTE_AMPLITUDE: f32 = MIDI_VELOCITY as f32 / 127.0;

/// Not a GM program. Drums arrive as `drum` tokens and carry this.
pub const DRUM_PROGRAM: u16 = 128;

/// `MINIMUM_NOTE_DURATION_SECONDS` in the reference: the floor `validate_notes`
/// widens short or inverted notes to, and the length of a drum hit.
pub const MIN_NOTE_DURATION_SECS: f64 = 0.01;

/// Checkpoint format generation this build understands. A loader reads it and
/// refuses any other, so a later generation lands beside this one instead of
/// replacing it.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSize {
    Small,
    Medium,
    Large,
}

impl std::fmt::Display for ModelSize {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::str::FromStr for ModelSize {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            other => Err(format!("unknown model size: {other}")),
        }
    }
}

impl ModelSize {
    pub const ALL: [Self; 3] = [Self::Small, Self::Medium, Self::Large];
    pub const DEFAULT: Self = Self::Medium;

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Small => "Small",
            Self::Medium => "Medium",
            Self::Large => "Large",
        }
    }

    pub fn from_str_or(value: &str, fallback: Self) -> Self {
        value.parse().unwrap_or(fallback)
    }
}

/// An MT3 instrument group id.
///
/// Group ids are *not* contiguous and *not* MIDI programs. 0..=33 are the named
/// families, 34 and 35 are the singleton groups for programs 100 and 101, 36 is
/// `drums`, and 37..=66 are the remaining singletons. Which program lands in
/// which id above 35 is an artifact of CPython set ordering in the reference's
/// `get_group_program_map`, so it is reproduced verbatim rather than recomputed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GroupId(pub u8);

impl GroupId {
    pub const DRUMS: Self = Self(36);

    pub const ALL_NAMED: [Self; 35] = [
        Self(0),
        Self(1),
        Self(2),
        Self(3),
        Self(4),
        Self(5),
        Self(6),
        Self(7),
        Self(8),
        Self(9),
        Self(10),
        Self(11),
        Self(12),
        Self(13),
        Self(14),
        Self(15),
        Self(16),
        Self(17),
        Self(18),
        Self(19),
        Self(20),
        Self(21),
        Self(22),
        Self(23),
        Self(24),
        Self(25),
        Self(26),
        Self(27),
        Self(28),
        Self(29),
        Self(30),
        Self(31),
        Self(32),
        Self(33),
        Self::DRUMS,
    ];

    /// `NAMED_GROUPS` from `cpp/src/instrument_groups.inc`.
    const NAMES: [(&'static str, u8); 35] = [
        ("acoustic_piano", 0),
        ("electric_piano", 1),
        ("chromatic_percussion", 2),
        ("organ", 3),
        ("acoustic_guitar", 4),
        ("clean_electric_guitar", 5),
        ("distorted_electric_guitar", 6),
        ("acoustic_bass", 7),
        ("electric_bass", 8),
        ("violin", 9),
        ("viola", 10),
        ("cello", 11),
        ("contrabass", 12),
        ("orchestral_harp", 13),
        ("timpani", 14),
        ("string_ensemble", 15),
        ("synth_strings", 16),
        ("voice", 17),
        ("orchestra_hit", 18),
        ("trumpet", 19),
        ("trombone", 20),
        ("tuba", 21),
        ("french_horn", 22),
        ("brass_section", 23),
        ("soprano_and_alto_sax", 24),
        ("tenor_sax", 25),
        ("baritone_sax", 26),
        ("oboe", 27),
        ("english_horn", 28),
        ("bassoon", 29),
        ("clarinet", 30),
        ("flutes", 31),
        ("synth_lead", 32),
        ("synth_pad", 33),
        ("drums", 36),
    ];

    /// `GROUP_REPRESENTATIVE`: group id to the only program the model ever
    /// emits for it.
    ///
    /// Note that group 36 (`drums`) does hold 96 -- it is *also* the singleton
    /// group of GM program 96, which is why `GroupId::for_program(96)` answers
    /// `drums`. Drums are never selected by program though, so
    /// `forbidden_token_ids` skips this entry rather than allowing it.
    const REPRESENTATIVE: [i16; 66] = [
        0, 2, 8, 16, 24, 26, 29, 32, 33, 40, 41, 42, //
        43, 46, 47, 48, 50, 52, 55, 56, 57, 58, 60, 61, //
        64, 66, 67, 68, 69, 70, 71, 72, 80, 88, 100, 101, //
        96, 97, 98, 99, 102, 103, 104, 105, 106, 107, 108, 109, //
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, //
        122, 123, 124, 125, 126, 127,
    ];

    pub fn is_valid(self) -> bool {
        (self.0 as usize) < Self::REPRESENTATIVE.len()
    }

    /// The group's first program, the only one the model emits for it.
    pub fn representative_program(self) -> Option<u16> {
        Self::REPRESENTATIVE
            .get(self.0 as usize)
            .copied()
            .filter(|program| *program >= 0)
            .map(|program| program as u16)
    }

    /// Row in the instrument-group embedding table. The null class is row 1,
    /// so group `g` is row `g + 2`.
    pub fn conditioning_row(self) -> i32 {
        i32::from(self.0) + 2
    }

    /// The user-facing name, or `None` for the unnamed groups.
    pub fn name(self) -> Option<&'static str> {
        Self::NAMES
            .iter()
            .find(|(_, id)| *id == self.0)
            .map(|(name, _)| *name)
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, id)| Self(*id))
    }

    pub const NAMED: [Self; 35] = {
        let mut out = [Self(0); 35];
        let mut i = 0;
        while i < 35 {
            out[i] = Self(Self::NAMES[i].1);
            i += 1;
        }
        out
    };

    /// `groupIdForProgram`: a linear search over the representative table, so
    /// programs in no group (notably 128 and 129) answer `None`.
    pub fn for_program(program: u16) -> Option<Self> {
        Self::REPRESENTATIVE
            .iter()
            .position(|candidate| *candidate == program as i16)
            .map(|index| Self(index as u8))
    }
}

/// The label the reference prints for a program: the group name, or
/// `program_<n>` when the program sits in an unnamed singleton group.
pub fn instrument_label(program: u16) -> String {
    match GroupId::for_program(program).and_then(GroupId::name) {
        Some(name) => name.to_owned(),
        None => format!("program_{program}"),
    }
}

/// A note's identity for as long as it is sounding. The open-note map is keyed
/// on this, and it is what the assembler matches an `End` against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NoteKey {
    pub program: i32,
    pub pitch: i32,
}

impl NoteKey {
    pub fn new(program: i32, pitch: i32) -> Self {
        Self { program, pitch }
    }
}

/// A finished note, ready for the piano roll, the synth or the MIDI writer.
///
/// `is_drum` is a real field, not `program == DRUM_PROGRAM`. Programs 128 and
/// 129 belong to no instrument group, so a *melodic* note can carry program
/// 128; conflating the two would put it in the wrong trimming channel.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NoteEvent {
    pub onset: f64,
    pub offset: f64,
    pub pitch: u8,
    pub program: u16,
    pub is_drum: bool,
}

impl NoteEvent {
    pub fn new(onset: f64, offset: f64, pitch: u8, program: u16, is_drum: bool) -> Self {
        Self {
            onset,
            offset,
            pitch,
            program,
            is_drum,
        }
    }

    pub fn end_time(&self) -> f64 {
        self.offset
    }

    pub fn duration_secs(&self) -> f64 {
        self.offset - self.onset
    }

    /// The fixed amplitude MuScriptor's notes carry. The model predicts no
    /// dynamics, so this is a constant rather than a measurement.
    pub fn amplitude(&self) -> f32 {
        FIXED_NOTE_AMPLITUDE
    }

    pub fn velocity(&self) -> u8 {
        MIDI_VELOCITY
    }
}

pub fn sort_notes(notes: &mut [NoteEvent]) {
    notes.sort_by(|a, b| {
        a.onset
            .total_cmp(&b.onset)
            .then_with(|| a.is_drum.cmp(&b.is_drum))
            .then_with(|| a.program.cmp(&b.program))
            .then_with(|| a.pitch.cmp(&b.pitch))
            .then_with(|| a.offset.total_cmp(&b.offset))
    });
}

/// `validate_notes(fix=True)`, restricted to the two branches decoding can
/// actually reach. Decoding always supplies an onset, and an `End` always
/// supplies an offset.
pub fn validate_notes(notes: &mut [NoteEvent]) {
    for note in notes.iter_mut() {
        if note.onset > note.offset {
            note.offset = note.offset.max(note.onset + MIN_NOTE_DURATION_SECS);
        } else if !note.is_drum && note.offset - note.onset < MIN_NOTE_DURATION_SECS {
            note.offset = note.onset + MIN_NOTE_DURATION_SECS;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_groups_match_the_generated_table() {
        assert_eq!(GroupId::NAMED.len(), 35);
        for group in GroupId::NAMED {
            let name = group.name().expect("named group");
            assert_eq!(GroupId::from_name(name), Some(group));
            assert_eq!(
                instrument_label(group.representative_program().unwrap()),
                name
            );
        }
        assert_eq!(GroupId::DRUMS.name(), Some("drums"));
        assert_eq!(GroupId(34).name(), None);
        assert_eq!(GroupId(35).name(), None);
    }

    #[test]
    fn representative_programs_match_the_generated_table() {
        assert_eq!(GroupId(0).representative_program(), Some(0));
        assert_eq!(GroupId(24).representative_program(), Some(64));
        assert_eq!(GroupId(33).representative_program(), Some(88));
        assert_eq!(GroupId(34).representative_program(), Some(100));
        assert_eq!(GroupId(35).representative_program(), Some(101));
        // Group 36 is drums and is also the singleton group of program 96.
        assert_eq!(GroupId::DRUMS.representative_program(), Some(96));
        assert_eq!(GroupId(37).representative_program(), Some(97));
        // 66 groups total, so valid ids run 0..=65.
        assert_eq!(GroupId(65).representative_program(), Some(127));
        assert_eq!(GroupId(66).representative_program(), None);
    }

    #[test]
    fn program_96_labels_as_drums_and_128_is_ungrouped() {
        assert_eq!(GroupId::for_program(96), Some(GroupId::DRUMS));
        assert_eq!(instrument_label(96), "drums");
        // Programs 128 and 129 belong to no group, so a melodic note may carry
        // program 128. This is why is_drum cannot be derived from it.
        assert_eq!(GroupId::for_program(128), None);
        assert_eq!(GroupId::for_program(129), None);
        assert_eq!(instrument_label(128), "program_128");
    }

    #[test]
    fn the_embedding_table_carries_a_bos_row() {
        // The card is the vocabulary size plus the row BOS lives in.
        assert_eq!(card_for(ModelSize::Small), 1_393);
        assert_eq!(card_for(ModelSize::Medium), 1_395);
        assert_eq!(card_for(ModelSize::Large), 1_395);

        assert_eq!(bos_id(ModelSize::Small), 1_393);
        assert_eq!(bos_id(ModelSize::Medium), 1_395);
    }

    #[test]
    fn ids_past_the_active_vocabulary_are_masked() {
        assert_eq!(logits_mask_start(ModelSize::Small), 1_393..1_393);
        assert_eq!(logits_mask_start(ModelSize::Medium), 1_393..1_395);
        assert_eq!(logits_mask_start(ModelSize::Large), 1_393..1_395);
    }

    #[test]
    fn conditioning_rows_are_group_plus_two() {
        assert_eq!(GroupId(0).conditioning_row(), 2);
        assert_eq!(GroupId::DRUMS.conditioning_row(), 38);
    }

    #[test]
    fn validate_widens_short_and_inverted_notes_but_spares_drums() {
        let mut notes = vec![
            NoteEvent {
                onset: 0.0,
                offset: 0.0,
                pitch: 60,
                program: 0,
                is_drum: false,
            },
            NoteEvent {
                onset: 0.0,
                offset: -1.0,
                pitch: 61,
                program: 0,
                is_drum: false,
            },
            NoteEvent {
                onset: 0.0,
                offset: 0.0,
                pitch: 36,
                program: DRUM_PROGRAM,
                is_drum: true,
            },
            NoteEvent {
                onset: 0.0,
                offset: 0.5,
                pitch: 62,
                program: 0,
                is_drum: false,
            },
        ];
        validate_notes(&mut notes);
        assert!((notes[0].offset - 0.01).abs() < 1e-12);
        assert!((notes[1].offset - 0.01).abs() < 1e-12);
        // A drum skips the minimum-duration branch entirely, so a zero-length one is
        // left alone. The assembler gives drum hits their length directly.
        assert!((notes[2].offset - 0.0).abs() < 1e-12);
        assert!((notes[3].offset - 0.5).abs() < 1e-12);
    }
}
