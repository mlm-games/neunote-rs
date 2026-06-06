use crate::midi::events::NoteEvent;

/// A named track containing note events.
#[derive(Debug, Clone)]
pub struct Track {
    pub name: String,
    pub events: Vec<NoteEvent>,
}

/// Pluggable strategy for assigning notes to tracks.
pub trait TrackAssigner {
    fn assign_tracks(&self, events: &[NoteEvent]) -> Vec<Track>;
}

/// Assigns notes to tracks based on pitch range.
///
/// This is a heuristic approach that works well for single-instrument audio.
/// Replace with a real source-separation-based assigner for multi-instrument accuracy.
pub struct PitchRangeAssigner {
    /// Maximum MIDI note for bass track (default: 47 = B3)
    pub bass_max: u8,
    /// Maximum MIDI note for keys track (default: 71 = B4)
    pub keys_max: u8,
    pub bass_name: String,
    pub keys_name: String,
    pub lead_name: String,
}

impl Default for PitchRangeAssigner {
    fn default() -> Self {
        Self {
            bass_max: 47,
            keys_max: 71,
            bass_name: "Bass".into(),
            keys_name: "Keys".into(),
            lead_name: "Lead".into(),
        }
    }
}

impl TrackAssigner for PitchRangeAssigner {
    fn assign_tracks(&self, events: &[NoteEvent]) -> Vec<Track> {
        let total = events.len();
        let mut bass = Vec::with_capacity(total / 3);
        let mut keys = Vec::with_capacity(total / 3);
        let mut lead = Vec::with_capacity(total / 3);

        for e in events {
            if e.pitch <= self.bass_max {
                bass.push(e.clone());
            } else if e.pitch <= self.keys_max {
                keys.push(e.clone());
            } else {
                lead.push(e.clone());
            }
        }

        let mut tracks = Vec::new();
        if !bass.is_empty() {
            tracks.push(Track { name: self.bass_name.clone(), events: bass });
        }
        if !keys.is_empty() {
            tracks.push(Track { name: self.keys_name.clone(), events: keys });
        }
        if !lead.is_empty() {
            tracks.push(Track { name: self.lead_name.clone(), events: lead });
        }
        if tracks.is_empty() && !events.is_empty() {
            tracks.push(Track { name: "Piano".into(), events: events.to_vec() });
        }
        tracks
    }
}
