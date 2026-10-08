use serde::{Deserialize, Serialize};

/// A single note event produced by the transcription pipeline
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoteEvent {
    pub start_time: f64,
    pub end_time: f64,
    pub start_frame: usize,
    pub end_frame: usize,
    /// MIDI note number (21-108)
    pub pitch: u8,
    pub amplitude: f32,
    /// Pitch bend values per frame. Units are 1/3 of a semitone.
    pub bends: Vec<i32>,
}

impl NoteEvent {
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }

    pub fn frequency(&self) -> f32 {
        midi_to_hz(self.pitch as f32)
    }
}

/// Convert MIDI note number to frequency in Hz
pub fn midi_to_hz(note: f32) -> f32 {
    440.0 * (2.0_f32).powf((note - 69.0) / 12.0)
}

/// Convert frequency in Hz to closest MIDI note number
pub fn hz_to_midi(hz: f32) -> u8 {
    (12.0 * (hz / 440.0).log2() + 69.0).round() as u8
}

/// Convert MIDI note number to string (e.g. C4, A#3)
pub fn midi_note_to_str(note: u8) -> String {
    const SHARP_NAMES: &[&str] = &[
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = (note / 12).saturating_sub(1);
    let idx = (note % 12) as usize;
    format!("{}{}", SHARP_NAMES[idx], octave)
}
