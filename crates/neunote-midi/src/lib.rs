#![forbid(unsafe_code)]

//! MIDI output: one track per instrument, drums on channel 10.
//!
//! The model assigns each note an instrument program, so the writer's job is to
//! keep that identity rather than flatten everything to piano the way the old
//! pipeline did. Tempo is fixed at 120 BPM -- the model predicts no tempo, and
//! inventing one would misplace every note.

use std::path::Path;

use neunote_types::{
    DRUM_PROGRAM, FIXED_NOTE_AMPLITUDE, MIDI_VELOCITY, NoteEvent, instrument_label,
};

/// Ticks per quarter note.
pub const PPQ: u16 = 480;

pub const DEFAULT_BPM: f64 = 120.0;

/// Channel 9 is MIDI channel 10, the percussion channel.
pub const DRUM_CHANNEL: u8 = 9;

/// Melodic channels available: all sixteen minus the percussion one.
pub const MELODIC_CHANNEL_COUNT: u8 = 15;

/// Notes for one instrument.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub program: u16,
    pub name: String,
    pub channel: u8,
    pub notes: Vec<NoteEvent>,
}

impl Track {
    pub fn is_drum(&self) -> bool {
        self.program == DRUM_PROGRAM
    }

    pub fn note_count(&self) -> usize {
        self.notes.len()
    }

    /// Span covered by this track, or zero when it has no notes.
    pub fn duration_secs(&self) -> f64 {
        self.notes
            .iter()
            .map(|note| note.end_time())
            .fold(0.0f64, f64::max)
    }
}

/// Group notes into one track per instrument.
///
/// Drums sort last. Everything else sorts by program number, so the same input
/// always produces the same track order and the same file bytes.
pub fn group_by_program(notes: &[NoteEvent]) -> Vec<Track> {
    let mut programs: Vec<u16> = notes.iter().map(|note| note.program).collect();
    programs.sort_unstable();
    programs.dedup();

    // Melodic channels are 0..=8 and 10..=15; channel 9 is percussion, so it is
    // skipped in the sequence rather than offset afterwards.
    let melodic_channel = |index: usize| -> u8 {
        let index = index as u8 % MELODIC_CHANNEL_COUNT;
        if index < DRUM_CHANNEL {
            index
        } else {
            index + 1
        }
    };

    let mut tracks = Vec::with_capacity(programs.len());
    let mut melodic_index = 0usize;

    for program in programs {
        let is_drum = program == DRUM_PROGRAM;
        let channel = if is_drum {
            DRUM_CHANNEL
        } else {
            let channel = melodic_channel(melodic_index);
            melodic_index += 1;
            channel
        };

        let mut owned: Vec<NoteEvent> = notes
            .iter()
            .filter(|note| note.program == program)
            .copied()
            .collect();
        owned.sort_by(|a, b| {
            a.onset
                .total_cmp(&b.onset)
                .then_with(|| a.pitch.cmp(&b.pitch))
        });

        tracks.push(Track {
            program,
            name: instrument_label(program),
            channel,
            notes: owned,
        });
    }

    tracks
}

/// Write notes to a format 1 file: a conductor track, then one per instrument.
pub fn write_midi_file(path: &Path, notes: &[NoteEvent], tempo_bpm: f64) -> std::io::Result<()> {
    write_midi_file_from_tracks(path, &group_by_program(notes), tempo_bpm)
}

pub fn write_midi_file_from_tracks(
    path: &Path,
    tracks: &[Track],
    tempo_bpm: f64,
) -> std::io::Result<()> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"MThd");
    bytes.extend_from_slice(&6u32.to_be_bytes());
    bytes.extend_from_slice(&1u16.to_be_bytes()); // format 1
    bytes.extend_from_slice(&((tracks.len() + 1) as u16).to_be_bytes());
    bytes.extend_from_slice(&PPQ.to_be_bytes());

    push_track(&mut bytes, &conductor_track(tempo_bpm));
    for track in tracks {
        push_track(&mut bytes, &instrument_track(track, tempo_bpm));
    }

    std::fs::write(path, bytes)
}

fn push_track(out: &mut Vec<u8>, events: &[u8]) {
    out.extend_from_slice(b"MTrk");
    out.extend_from_slice(&(events.len() as u32).to_be_bytes());
    out.extend_from_slice(events);
}

/// Tempo, time signature and the end-of-track marker.
fn conductor_track(bpm: f64) -> Vec<u8> {
    let micros_per_beat = (60_000_000.0 / tempo(bpm))
        .round()
        .clamp(1.0, 0xFF_FFFF_u32 as f64) as u32;

    let mut events = Vec::new();
    write_vlq(&mut events, 0);
    events.extend_from_slice(&[0xFF, 0x51, 0x03]); // set tempo
    events.extend_from_slice(&micros_per_beat.to_be_bytes()[1..]);
    write_vlq(&mut events, 0);
    events.extend_from_slice(&[0xFF, 0x58, 0x04, 4, 2, 24, 8]); // 4/4
    write_vlq(&mut events, 0);
    events.extend_from_slice(&[0xFF, 0x2F, 0x00]);
    events
}

fn instrument_track(track: &Track, bpm: f64) -> Vec<u8> {
    let mut events = Vec::new();

    // Drums ignore program changes; the channel already says percussion.
    if !track.is_drum() && track.program < 128 {
        write_vlq(&mut events, 0);
        events.extend_from_slice(&[0xC0 | (track.channel & 0x0F), track.program as u8]);
    }

    let mut timed: Vec<(u64, bool, u8, u8)> = Vec::with_capacity(track.notes.len() * 2);
    for note in &track.notes {
        let pitch = note.pitch;
        let velocity = note.velocity();
        timed.push((secs_to_ticks(note.onset, bpm), false, pitch, velocity));
        timed.push((secs_to_ticks(note.end_time(), bpm), true, pitch, 0));
    }

    // At the same tick a note-off must precede a note-on, or a repeated pitch
    // retriggers without ever releasing. So ends sort before starts.
    timed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));

    let mut previous = 0u64;
    for (tick, is_off, pitch, velocity) in timed {
        write_vlq(&mut events, tick.saturating_sub(previous));
        previous = tick;
        let status = if is_off { 0x80 } else { 0x90 } | (track.channel & 0x0F);
        events.push(status);
        events.push(pitch);
        events.push(velocity);
    }

    write_vlq(&mut events, 0);
    events.extend_from_slice(&[0xFF, 0x2F, 0x00]);
    events
}

/// A tempo that can actually be written, falling back to the default.
fn tempo(bpm: f64) -> f64 {
    if bpm.is_finite() && bpm > 0.0 {
        bpm
    } else {
        DEFAULT_BPM
    }
}

fn secs_to_ticks(secs: f64, bpm: f64) -> u64 {
    if !secs.is_finite() || secs <= 0.0 {
        return 0;
    }
    (secs * tempo(bpm) / 60.0 * f64::from(PPQ)).round() as u64
}

fn write_vlq(out: &mut Vec<u8>, value: u64) {
    let mut buffer = Vec::new();
    let mut rest = value >> 7;
    while rest > 0 {
        buffer.push((rest as u8 & 0x7F) | 0x80);
        rest >>= 7;
    }
    buffer.push(value as u8 & 0x7F);
    out.extend_from_slice(&buffer);
}

/// Total length of a written file, from its tracks' notes.
pub fn duration_secs(tracks: &[Track]) -> f64 {
    tracks
        .iter()
        .map(|track| track.duration_secs())
        .fold(0.0f64, f64::max)
}

/// The fixed velocity the model implies.
pub const fn velocity() -> u8 {
    MIDI_VELOCITY
}

/// The fixed amplitude the model implies.
pub const fn amplitude() -> f32 {
    FIXED_NOTE_AMPLITUDE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(onset: f64, offset: f64, pitch: u8, program: u16) -> NoteEvent {
        NoteEvent {
            onset,
            offset,
            pitch,
            program,
            is_drum: program == DRUM_PROGRAM,
        }
    }

    #[test]
    fn one_track_per_instrument_with_drums_last() {
        let notes = vec![
            note(0.0, 1.0, 36, DRUM_PROGRAM),
            note(0.0, 1.0, 60, 40),
            note(0.0, 1.0, 62, 0),
        ];
        let tracks = group_by_program(&notes);

        assert_eq!(tracks.len(), 3);
        assert_eq!(
            tracks.iter().map(|t| t.program).collect::<Vec<_>>(),
            vec![0, 40, DRUM_PROGRAM]
        );
        assert_eq!(tracks[0].name, "acoustic_piano");
        assert_eq!(tracks[1].name, "violin");
        // DRUM_PROGRAM is 128, which belongs to no instrument group, so the
        // label falls back to the program number. The track is still drums.
        assert_eq!(tracks[2].name, "program_128");
        assert!(tracks[2].is_drum());

        // Drums are on channel 10; melody starts at 1.
        assert_eq!(tracks[2].channel, DRUM_CHANNEL);
        assert_ne!(tracks[0].channel, DRUM_CHANNEL);
    }

    #[test]
    fn channels_are_assigned_in_order_and_never_collide() {
        let notes: Vec<NoteEvent> = (0..MELODIC_CHANNEL_COUNT as u16)
            .map(|program| note(0.0, 1.0, 60, program))
            .collect();
        let tracks = group_by_program(&notes);

        let mut channels: Vec<u8> = tracks.iter().map(|t| t.channel).collect();
        channels.sort_unstable();
        let before = channels.len();
        channels.dedup();
        assert_eq!(channels.len(), before, "two tracks share a channel");

        // Channel 9 is skipped rather than taken.
        assert!(!channels.contains(&DRUM_CHANNEL));
        assert!(channels.iter().all(|channel| *channel < 16));
    }

    #[test]
    fn more_programs_than_melodic_channels_wraps_rather_than_escaping() {
        let notes: Vec<NoteEvent> = (0..40u16).map(|p| note(0.0, 1.0, 60, p)).collect();
        let tracks = group_by_program(&notes);

        assert!(tracks.iter().all(|track| track.channel < 16));
        assert_eq!(
            tracks
                .iter()
                .filter(|track| track.channel == DRUM_CHANNEL)
                .count(),
            0,
            "melodic programs must not land on percussion"
        );
    }

    #[test]
    fn grouping_is_deterministic() {
        let notes = vec![
            note(1.0, 2.0, 60, 40),
            note(0.0, 1.0, 62, 0),
            note(0.5, 1.5, 64, 0),
        ];
        assert_eq!(group_by_program(&notes), group_by_program(&notes));
    }

    #[test]
    fn notes_in_a_track_are_ordered_by_onset() {
        let notes = vec![
            note(2.0, 3.0, 60, 0),
            note(0.0, 1.0, 62, 0),
            note(1.0, 1.5, 64, 0),
        ];
        let tracks = group_by_program(&notes);
        let onsets: Vec<f64> = tracks[0].notes.iter().map(|n| n.onset).collect();
        assert_eq!(onsets, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn tempo_maps_seconds_onto_the_grid() {
        // At 120 BPM a beat is half a second and a beat is PPQ ticks.
        assert_eq!(secs_to_ticks(0.0, 120.0), 0);
        assert_eq!(secs_to_ticks(0.5, 120.0), PPQ as u64);
        assert_eq!(secs_to_ticks(1.0, 120.0), 2 * PPQ as u64);
        // At 60 BPM a second is a beat.
        assert_eq!(secs_to_ticks(1.0, 60.0), PPQ as u64);
        // Nonsense in, zero out.
        assert_eq!(secs_to_ticks(-1.0, 120.0), 0);
        assert_eq!(secs_to_ticks(f64::NAN, 120.0), 0);
    }

    #[test]
    fn vlq_encodes_the_variable_length_prefix() {
        let mut out = Vec::new();
        write_vlq(&mut out, 0);
        assert_eq!(out, vec![0x00]);

        for (value, expected) in [
            (0x7Fu64, vec![0x7Fu8]),
            (0x80, vec![0x81u8, 0x00]),
            (0x2000, vec![0xC0u8, 0x00]),
            (0x1FFFFF, vec![0xFFu8, 0xFF, 0x7F]),
        ] {
            let mut out = Vec::new();
            write_vlq(&mut out, value);
            assert_eq!(out, expected, "value {value:#x}");
        }
    }

    #[test]
    fn every_note_gets_the_fixed_velocity() {
        let notes = [note(0.0, 1.0, 60, 0)];
        assert_eq!(notes[0].velocity(), MIDI_VELOCITY);
        assert!((notes[0].amplitude() - FIXED_NOTE_AMPLITUDE).abs() < f32::EPSILON);
    }

    #[test]
    fn an_empty_note_list_produces_no_tracks() {
        assert!(group_by_program(&[]).is_empty());
        assert_eq!(duration_secs(&[]), 0.0);
    }
}
