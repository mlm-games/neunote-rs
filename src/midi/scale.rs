use serde::{Deserialize, Serialize};

use crate::midi::events::NoteEvent;
use crate::ml::constants::{MAX_MIDI_NOTE, MIN_MIDI_NOTE};

/// Root note enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum RootNote {
    A = 0,
    ASharp,
    B,
    C,
    CSharp,
    D,
    DSharp,
    E,
    F,
    FSharp,
    G,
    GSharp,
}

impl RootNote {
    pub fn from_midi(midi: u8) -> Self {
        match (midi + 3) % 12 {
            0 => RootNote::C,
            1 => RootNote::CSharp,
            2 => RootNote::D,
            3 => RootNote::DSharp,
            4 => RootNote::E,
            5 => RootNote::F,
            6 => RootNote::FSharp,
            7 => RootNote::G,
            8 => RootNote::GSharp,
            9 => RootNote::A,
            10 => RootNote::ASharp,
            11 => RootNote::B,
            _ => unreachable!(),
        }
    }

    fn to_note_idx(self) -> usize {
        ((self as usize) + 12 - 3) % 12
    }
}

/// Scale type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleType {
    Chromatic,
    Major,
    Minor,
    Dorian,
    Mixolydian,
    Lydian,
    Phrygian,
    Locrian,
    MinorBlues,
    MinorPentatonic,
    MajorPentatonic,
    MelodicMinor,
    HarmonicMinor,
    HarmonicMajor,
}

/// Snap mode for scale quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapMode {
    /// Snap to nearest in-key note
    Adjust,
    /// Remove out-of-key notes
    Remove,
}

/// Configuration for note/scale post-processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteOptions {
    pub enabled: bool,
    pub root_note: RootNote,
    pub scale_type: ScaleType,
    pub snap_mode: SnapMode,
    pub min_midi_note: u8,
    pub max_midi_note: u8,
}

impl Default for NoteOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            root_note: RootNote::C,
            scale_type: ScaleType::Chromatic,
            snap_mode: SnapMode::Remove,
            min_midi_note: MIN_MIDI_NOTE,
            max_midi_note: MAX_MIDI_NOTE,
        }
    }
}

impl NoteOptions {
    /// Process note events through scale quantization and range filtering
    pub fn process(&self, events: &[NoteEvent]) -> Vec<NoteEvent> {
        if !self.enabled {
            return events.to_vec();
        }

        let key_vec = if self.scale_type == ScaleType::Chromatic {
            vec![]
        } else {
            create_key_vector(self.root_note, self.scale_type)
        };

        events
            .iter()
            .filter(|e| e.pitch >= self.min_midi_note && e.pitch <= self.max_midi_note)
            .filter_map(|e| {
                if self.scale_type == ScaleType::Chromatic {
                    Some(e.clone())
                } else {
                    match self.snap_mode {
                        SnapMode::Remove => {
                            if is_in_key(e.pitch, &key_vec) {
                                Some(e.clone())
                            } else {
                                None
                            }
                        }
                        SnapMode::Adjust => {
                            let mut adjusted = e.clone();
                            let adjust_up = adjusted.bends.iter().sum::<i32>() >= 0;
                            adjusted.pitch = closest_in_key(e.pitch, &key_vec, adjust_up);
                            Some(adjusted)
                        }
                    }
                }
            })
            .collect()
    }
}

/// Check if MIDI note is in key
fn is_in_key(midi_note: u8, key_vec: &[usize]) -> bool {
    let note_idx = (midi_note % 12) as usize;
    key_vec.contains(&note_idx)
}

/// Find closest MIDI note in key
fn closest_in_key(midi_note: u8, key_vec: &[usize], adjust_up: bool) -> u8 {
    if is_in_key(midi_note, key_vec) {
        return midi_note;
    }
    if adjust_up {
        if midi_note < MAX_MIDI_NOTE - 1 {
            midi_note + 1
        } else {
            midi_note - 1
        }
    } else if midi_note > MIN_MIDI_NOTE {
        midi_note - 1
    } else {
        midi_note + 1
    }
}

/// Build the set of valid note indices (0-11) for a given root + scale
fn create_key_vector(root: RootNote, scale: ScaleType) -> Vec<usize> {
    let root_idx = root.to_note_idx();
    let intervals: &[usize] = match scale {
        ScaleType::Chromatic => return vec![],
        ScaleType::Major => &[0, 2, 4, 5, 7, 9, 11],
        ScaleType::Minor => &[0, 2, 3, 5, 7, 8, 10],
        ScaleType::Dorian => &[0, 2, 3, 5, 7, 9, 10],
        ScaleType::Mixolydian => &[0, 2, 4, 5, 7, 9, 10],
        ScaleType::Lydian => &[0, 2, 4, 6, 7, 9, 11],
        ScaleType::Phrygian => &[0, 1, 3, 5, 7, 8, 10],
        ScaleType::Locrian => &[0, 1, 3, 5, 6, 8, 10],
        ScaleType::MinorBlues => &[0, 3, 5, 6, 7, 10],
        ScaleType::MinorPentatonic => &[0, 3, 5, 7, 10],
        ScaleType::MajorPentatonic => &[0, 2, 4, 7, 9],
        ScaleType::MelodicMinor => &[0, 2, 3, 5, 7, 8, 10],
        ScaleType::HarmonicMinor => &[0, 2, 3, 5, 7, 8, 11],
        ScaleType::HarmonicMajor => &[0, 2, 4, 5, 7, 8, 11],
    };
    intervals.iter().map(|i| (root_idx + i) % 12).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_major_scale() {
        let key = create_key_vector(RootNote::C, ScaleType::Major);
        assert_eq!(key, vec![0, 2, 4, 5, 7, 9, 11]);
        assert!(is_in_key(60, &key)); // C4
        assert!(is_in_key(62, &key)); // D4
        assert!(is_in_key(64, &key)); // E4
        assert!(!is_in_key(61, &key)); // C#4 - not in C major
    }

    #[test]
    fn test_remove_out_of_key() {
        let opts = NoteOptions {
            enabled: true,
            root_note: RootNote::C,
            scale_type: ScaleType::Major,
            snap_mode: SnapMode::Remove,
            ..Default::default()
        };
        let events = vec![
            NoteEvent {
                pitch: 60,
                start_time: 0.0,
                end_time: 1.0,
                start_frame: 0,
                end_frame: 10,
                amplitude: 0.8,
                bends: vec![],
            },
            NoteEvent {
                pitch: 61,
                start_time: 1.0,
                end_time: 2.0,
                start_frame: 10,
                end_frame: 20,
                amplitude: 0.8,
                bends: vec![],
            },
        ];
        let processed = opts.process(&events);
        assert_eq!(processed.len(), 1);
        assert_eq!(processed[0].pitch, 60);
    }
}
