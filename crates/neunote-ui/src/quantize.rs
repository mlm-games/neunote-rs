//! Non-destructive quantisation: pull pitches into a scale, pull onsets onto a
//! grid.
//!
//! Pure, so the panel can re-run it on every parameter change without touching
//! the transcription: the raw notes stay where the model put them and this is a
//! view over them.

use neunote_types::NoteEvent;

pub const NOTE_NAMES: [&str; 12] = [
    "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B",
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scale {
    Chromatic,
    Major,
    Minor,
    Dorian,
    Mixolydian,
    Lydian,
    Phrygian,
    Locrian,
    HarmonicMinor,
    MelodicMinor,
    PentatonicMajor,
    PentatonicMinor,
    Blues,
}

impl Scale {
    pub const ALL: [Scale; 13] = [
        Scale::Chromatic,
        Scale::Major,
        Scale::Minor,
        Scale::Dorian,
        Scale::Mixolydian,
        Scale::Lydian,
        Scale::Phrygian,
        Scale::Locrian,
        Scale::HarmonicMinor,
        Scale::MelodicMinor,
        Scale::PentatonicMajor,
        Scale::PentatonicMinor,
        Scale::Blues,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Scale::Chromatic => "Chromatic",
            Scale::Major => "Major",
            Scale::Minor => "Minor",
            Scale::Dorian => "Dorian",
            Scale::Mixolydian => "Mixolydian",
            Scale::Lydian => "Lydian",
            Scale::Phrygian => "Phrygian",
            Scale::Locrian => "Locrian",
            Scale::HarmonicMinor => "Harm. minor",
            Scale::MelodicMinor => "Mel. minor",
            Scale::PentatonicMajor => "Pent. major",
            Scale::PentatonicMinor => "Pent. minor",
            Scale::Blues => "Blues",
        }
    }

    pub fn degrees(self) -> &'static [u8] {
        match self {
            Scale::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            Scale::Major => &[0, 2, 4, 5, 7, 9, 11],
            Scale::Minor => &[0, 2, 3, 5, 7, 8, 10],
            Scale::Dorian => &[0, 2, 3, 5, 7, 9, 10],
            Scale::Mixolydian => &[0, 2, 4, 5, 7, 9, 10],
            Scale::Lydian => &[0, 2, 4, 6, 7, 9, 11],
            Scale::Phrygian => &[0, 1, 3, 5, 7, 8, 10],
            Scale::Locrian => &[0, 1, 3, 5, 6, 8, 10],
            Scale::HarmonicMinor => &[0, 2, 3, 5, 7, 8, 11],
            Scale::MelodicMinor => &[0, 2, 3, 5, 7, 9, 11],
            Scale::PentatonicMajor => &[0, 2, 4, 7, 9],
            Scale::PentatonicMinor => &[0, 3, 5, 7, 10],
            Scale::Blues => &[0, 3, 5, 6, 7, 10],
        }
    }

    pub fn allows(self, root: u8, pitch: u8) -> bool {
        let degree = (i32::from(pitch) - i32::from(root)).rem_euclid(12) as u8;
        self.degrees().contains(&degree)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Snap {
    Remove,
    Nearest,
    Up,
    Down,
}

impl Snap {
    pub const ALL: [Snap; 4] = [Snap::Remove, Snap::Nearest, Snap::Up, Snap::Down];

    pub fn name(self) -> &'static str {
        match self {
            Snap::Remove => "Drop",
            Snap::Nearest => "Nearest",
            Snap::Up => "Up",
            Snap::Down => "Down",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Division {
    Bar,
    Half,
    Quarter,
    Eighth,
    Sixteenth,
    TripletEighth,
    TripletSixteenth,
    ThirtySecond,
}

impl Division {
    pub const ALL: [Division; 8] = [
        Division::Bar,
        Division::Half,
        Division::Quarter,
        Division::Eighth,
        Division::Sixteenth,
        Division::TripletEighth,
        Division::TripletSixteenth,
        Division::ThirtySecond,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Division::Bar => "1/1",
            Division::Half => "1/2",
            Division::Quarter => "1/4",
            Division::Eighth => "1/8",
            Division::Sixteenth => "1/16",
            Division::TripletEighth => "1/8T",
            Division::TripletSixteenth => "1/16T",
            Division::ThirtySecond => "1/32",
        }
    }

    /// Length in quarter-note beats.
    pub fn beats(self) -> f64 {
        match self {
            Division::Bar => 4.0,
            Division::Half => 2.0,
            Division::Quarter => 1.0,
            Division::Eighth => 0.5,
            Division::Sixteenth => 0.25,
            Division::TripletEighth => 1.0 / 3.0,
            Division::TripletSixteenth => 1.0 / 6.0,
            Division::ThirtySecond => 0.125,
        }
    }
}

/// Everything the quantise panel can say. The default is off in both
/// directions: a transcription is shown as the model produced it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Quantize {
    pub pitches: bool,
    pub root: u8,
    pub scale: Scale,
    pub snap: Snap,
    pub times: bool,
    pub bpm: f64,
    pub division: Division,
    pub strength: f64,
}

impl Default for Quantize {
    fn default() -> Self {
        Self {
            pitches: false,
            root: 0,
            scale: Scale::Chromatic,
            snap: Snap::Nearest,
            times: false,
            bpm: 120.0,
            division: Division::Sixteenth,
            strength: 1.0,
        }
    }
}

impl Quantize {
    /// The grid onsets are pulled towards, in seconds.
    pub fn step_secs(&self) -> f64 {
        self.division.beats() * 60.0 / self.bpm.max(1.0)
    }

    /// True when applying this would change nothing.
    pub fn is_identity(&self) -> bool {
        (!self.pitches || self.scale == Scale::Chromatic) && (!self.times || self.strength <= 0.0)
    }

    /// Snap pitches into the scale. Drum hits keep their pitch: a scale has
    /// nothing to say about a snare.
    fn apply_pitches(&self, notes: &mut Vec<(usize, NoteEvent)>) {
        if !self.pitches || self.scale == Scale::Chromatic {
            return;
        }

        notes.retain_mut(|(_, note)| {
            if note.is_drum || self.scale.allows(self.root, note.pitch) {
                return true;
            }

            if self.snap == Snap::Remove {
                return false;
            }

            match snapped(self.scale, self.root, note.pitch, self.snap) {
                Some(pitch) => {
                    note.pitch = pitch;
                    true
                }
                // Nothing in the scale above or below -- the top of the
                // keyboard. The note stays where the model put it: losing one
                // is worse than a wrong one.
                None => true,
            }
        });
    }

    /// Pull onsets towards the grid, keeping every note's length. Strength 1
    /// lands on it; below that the note is a fraction of the way there, so the
    /// groove survives.
    fn apply_times(&self, notes: &mut [(usize, NoteEvent)]) {
        if !self.times || self.strength <= 0.0 {
            return;
        }

        let step = self.step_secs();
        if step <= 0.0 {
            return;
        }

        for (_, note) in notes.iter_mut() {
            let length = note.offset - note.onset;
            let target = (note.onset / step).round() * step;
            let onset = (note.onset + (target - note.onset) * self.strength).max(0.0);
            note.onset = onset;
            note.offset = (onset + length).max(onset + f64::MIN_POSITIVE);
        }
    }

    /// Quantise a list that carries its place in the raw transcription, and
    /// keep every place: an edit on a quantised note has to reach the note it
    /// came from, not whatever now sits at that position.
    pub fn apply_indexed(&self, notes: &[(usize, NoteEvent)]) -> Vec<(usize, NoteEvent)> {
        if self.is_identity() {
            return notes.to_vec();
        }

        let mut out = notes.to_vec();
        self.apply_pitches(&mut out);
        self.apply_times(&mut out);
        out
    }
}

fn snapped(scale: Scale, root: u8, pitch: u8, mode: Snap) -> Option<u8> {
    let up = (i32::from(pitch)..=127)
        .find(|candidate| scale.allows(root, *candidate as u8))
        .map(|candidate| candidate as u8);
    let down = (0..=i32::from(pitch))
        .rev()
        .find(|candidate| scale.allows(root, *candidate as u8))
        .map(|candidate| candidate as u8);

    match mode {
        Snap::Remove => None,
        Snap::Up => up,
        Snap::Down => down,
        // Ties go up, so repeated presses settle rather than wander.
        Snap::Nearest => match (up, down) {
            (Some(up), Some(down)) => {
                if i32::from(up) - i32::from(pitch) <= i32::from(pitch) - i32::from(down) {
                    Some(up)
                } else {
                    Some(down)
                }
            }
            (Some(up), None) => Some(up),
            (None, Some(down)) => Some(down),
            (None, None) => None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quantise a plain list: these tests care about the notes, not where each
    /// one came from.
    fn apply(quantize: &Quantize, notes: &[NoteEvent]) -> Vec<NoteEvent> {
        let indexed = notes.iter().copied().enumerate().collect::<Vec<_>>();
        quantize
            .apply_indexed(&indexed)
            .into_iter()
            .map(|(_, note)| note)
            .collect()
    }

    fn note(onset: f64, offset: f64, pitch: u8) -> NoteEvent {
        NoteEvent {
            onset,
            offset,
            pitch,
            program: 0,
            is_drum: false,
        }
    }

    fn drum(pitch: u8) -> NoteEvent {
        NoteEvent {
            onset: 0.0,
            offset: 0.01,
            pitch,
            program: neunote_types::DRUM_PROGRAM,
            is_drum: true,
        }
    }

    fn pitches(notes: &[NoteEvent]) -> Vec<u8> {
        notes.iter().map(|note| note.pitch).collect()
    }

    #[test]
    fn the_default_leaves_a_transcription_alone() {
        let notes = vec![note(0.0, 0.5, 61), note(0.5, 1.0, 63)];
        assert!(Quantize::default().is_identity());
        assert_eq!(apply(&Quantize::default(), &notes), notes);
    }

    #[test]
    fn chromatic_is_never_a_filter() {
        let q = Quantize {
            pitches: true,
            scale: Scale::Chromatic,
            ..Quantize::default()
        };
        assert!(q.is_identity());
        assert_eq!(pitches(&apply(&q, &[note(0.0, 1.0, 61)])), vec![61]);
    }

    #[test]
    fn an_out_of_scale_pitch_goes_to_the_nearer_degree() {
        let q = Quantize {
            pitches: true,
            scale: Scale::PentatonicMajor,
            snap: Snap::Nearest,
            ..Quantize::default()
        };
        // C pentatonic has no F, so F goes down to E and F# up to G -- the gap
        // between them is wide enough for "nearer" to mean something.
        assert_eq!(pitches(&apply(&q, &[note(0.0, 1.0, 65)])), vec![64]);
        assert_eq!(pitches(&apply(&q, &[note(0.0, 1.0, 66)])), vec![67]);
        assert_eq!(
            pitches(&apply(&q, &[note(0.0, 1.0, 60)])),
            vec![60],
            "in scale"
        );
    }

    #[test]
    fn equally_distant_pitches_go_up() {
        let q = Quantize {
            pitches: true,
            scale: Scale::Major,
            snap: Snap::Nearest,
            ..Quantize::default()
        };
        // F# sits a semitone from both F and G.
        assert_eq!(pitches(&apply(&q, &[note(0.0, 1.0, 66)])), vec![67]);
    }

    #[test]
    fn remove_drops_the_note_and_the_others_modes_move_it() {
        let notes = vec![note(0.0, 1.0, 66)];
        let base = Quantize {
            pitches: true,
            scale: Scale::Major,
            ..Quantize::default()
        };

        assert!(
            apply(
                &Quantize {
                    snap: Snap::Remove,
                    ..base
                },
                &notes
            )
            .is_empty()
        );
        assert_eq!(
            pitches(
                &apply(
                    &Quantize {
                        snap: Snap::Up,
                        ..base
                    },
                    &notes,
                )
            ),
            vec![67]
        );
        assert_eq!(
            pitches(
                &apply(
                    &Quantize {
                        snap: Snap::Down,
                        ..base
                    },
                    &notes,
                )
            ),
            vec![65]
        );
    }

    #[test]
    fn the_top_of_the_keyboard_does_not_run_out_of_scale() {
        let q = Quantize {
            pitches: true,
            scale: Scale::Major,
            snap: Snap::Up,
            ..Quantize::default()
        };
        // Nothing above 127, so the note stays where the model put it.
        assert_eq!(pitches(&apply(&q, &[note(0.0, 1.0, 127)])), vec![127]);
    }

    #[test]
    fn a_root_shifts_the_scale() {
        let q = Quantize {
            pitches: true,
            root: 7,
            scale: Scale::Major,
            snap: Snap::Nearest,
            ..Quantize::default()
        };
        // G major has F# (66); 65 is a whole step away and 67 is not in it.
        assert_eq!(pitches(&apply(&q, &[note(0.0, 1.0, 65)])), vec![66]);
    }

    #[test]
    fn drums_are_never_snapped_into_a_scale() {
        let q = Quantize {
            pitches: true,
            scale: Scale::Major,
            snap: Snap::Remove,
            ..Quantize::default()
        };
        let notes = vec![drum(42)];
        assert_eq!(apply(&q, &notes).len(), 1, "a snare is not a wrong note");
        assert_eq!(apply(&q, &notes)[0].pitch, 42);
    }

    #[test]
    fn a_full_strength_time_snap_lands_on_the_grid() {
        let q = Quantize {
            times: true,
            bpm: 120.0,
            division: Division::Sixteenth,
            strength: 1.0,
            ..Quantize::default()
        };
        // At 120 bpm a sixteenth is 0.125 s.
        assert!((q.step_secs() - 0.125).abs() < 1e-12);
        let out = apply(&q, &[note(0.51, 0.62, 60)]);
        assert!((out[0].onset - 0.5).abs() < 1e-12);
        assert!((out[0].offset - 0.61).abs() < 1e-12, "length is kept");
    }

    #[test]
    fn a_partial_strength_time_snap_stops_short() {
        let q = Quantize {
            times: true,
            bpm: 120.0,
            division: Division::Sixteenth,
            strength: 0.5,
            ..Quantize::default()
        };
        let out = apply(&q, &[note(0.55, 0.6, 60)]);
        assert!((out[0].onset - 0.525).abs() < 1e-12);
        assert!((out[0].offset - 0.575).abs() < 1e-12);
    }

    #[test]
    fn zero_strength_leaves_the_timing_exactly_as_transcribed() {
        let q = Quantize {
            times: true,
            strength: 0.0,
            ..Quantize::default()
        };
        assert!(q.is_identity());
        let notes = vec![note(0.517, 0.6, 60)];
        assert_eq!(apply(&q, &notes)[0].onset, 0.517);
    }

    #[test]
    fn triplets_are_thirds_of_a_beat() {
        let q = Quantize {
            times: true,
            bpm: 90.0,
            division: Division::TripletEighth,
            ..Quantize::default()
        };
        // A beat at 90 bpm is 2/3 s, so a triplet eighth is 2/9 s.
        assert!((q.step_secs() - 2.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn a_note_before_the_grid_never_goes_negative() {
        let q = Quantize {
            times: true,
            bpm: 120.0,
            division: Division::Quarter,
            strength: 1.0,
            ..Quantize::default()
        };
        let out = apply(&q, &[note(-0.2, 0.1, 60)]);
        assert_eq!(out[0].onset, 0.0);
        assert!(out[0].offset > out[0].onset);
    }

    #[test]
    fn every_scale_and_every_division_is_reachable_by_index() {
        assert_eq!(Scale::ALL.len(), 13);
        assert_eq!(Division::ALL.len(), 8);
        assert_eq!(Snap::ALL.len(), 4);
        for scale in Scale::ALL {
            assert!(!scale.degrees().is_empty());
            assert!(scale.allows(0, 0), "the root is always in its own scale");
        }
    }
}
