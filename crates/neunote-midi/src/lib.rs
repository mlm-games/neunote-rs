#![forbid(unsafe_code)]

//! MIDI output: one track per instrument, drums on channel 10.
//!
//! The model assigns each note an instrument program, so the writer's job is to
//! keep that identity rather than flatten everything to piano the way the old
//! pipeline did. Tempo is fixed at 120 BPM -- the model predicts no tempo, and
//! inventing one would misplace every note.

use std::io;
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

/// Why a set of notes cannot be laid out as tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackError {
    /// More distinct melodic instruments than there are melodic channels.
    ///
    /// Two programs sharing a channel would each emit a program change, and
    /// only the last one read would apply, so one instrument would play in the
    /// other's voice. Rather than silently corrupt the file, refuse.
    TooManyInstruments { instruments: usize, channels: usize },
}

impl std::fmt::Display for TrackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackError::TooManyInstruments {
                instruments,
                channels,
            } => write!(
                f,
                "{instruments} melodic instruments need {channels} melodic MIDI channels; \
                 split the transcription, or export to several files"
            ),
        }
    }
}

impl std::error::Error for TrackError {}

/// Notes for one instrument.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub program: u16,
    /// Whether these notes are percussion.
    ///
    /// A field, not `program == DRUM_PROGRAM`: programs 128 and 129 belong to
    /// no instrument group, so a melodic note can carry program 128.
    pub is_drum: bool,
    pub name: String,
    pub channel: u8,
    pub notes: Vec<NoteEvent>,
}

impl Track {
    pub fn is_drum(&self) -> bool {
        self.is_drum
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

/// Melodic channels are 0..=8 and 10..=15; channel 9 is percussion, so it is
/// skipped in the sequence rather than offset afterwards.
fn melodic_channel(index: usize) -> u8 {
    let index = index as u8;
    if index < DRUM_CHANNEL {
        index
    } else {
        index + 1
    }
}

/// Group notes into one track per instrument.
///
/// `is_drum` is read from each note, never inferred from its program. Every
/// drum shares the percussion channel, where the pitch selects the sound;
/// melodic notes get one channel each. Tracks sort by program so the same
/// input always produces the same track order and the same file bytes, with
/// drums last.
pub fn group_by_program(notes: &[NoteEvent]) -> Result<Vec<Track>, TrackError> {
    let mut programs: Vec<u16> = notes
        .iter()
        .filter(|note| !note.is_drum)
        .map(|note| note.program)
        .collect();
    programs.sort_unstable();
    programs.dedup();

    if programs.len() > MELODIC_CHANNEL_COUNT as usize {
        return Err(TrackError::TooManyInstruments {
            instruments: programs.len(),
            channels: MELODIC_CHANNEL_COUNT as usize,
        });
    }

    let has_drums = notes.iter().any(|note| note.is_drum);
    let mut tracks = Vec::with_capacity(programs.len() + usize::from(has_drums));

    let sorted = |mut owned: Vec<NoteEvent>| {
        owned.sort_by(|a, b| {
            a.onset
                .total_cmp(&b.onset)
                .then_with(|| a.pitch.cmp(&b.pitch))
        });
        owned
    };

    for (index, program) in programs.iter().enumerate() {
        let owned = notes
            .iter()
            .filter(|note| !note.is_drum && note.program == *program)
            .copied()
            .collect();

        tracks.push(Track {
            program: *program,
            is_drum: false,
            name: instrument_label(*program),
            channel: melodic_channel(index),
            notes: sorted(owned),
        });
    }

    if has_drums {
        let owned = notes.iter().filter(|note| note.is_drum).copied().collect();

        tracks.push(Track {
            program: DRUM_PROGRAM,
            is_drum: true,
            name: instrument_label(DRUM_PROGRAM),
            channel: DRUM_CHANNEL,
            notes: sorted(owned),
        });
    }

    Ok(tracks)
}

/// Write notes to a format 1 file: a conductor track, then one per instrument.
pub fn write_midi_file(path: &Path, notes: &[NoteEvent], tempo_bpm: f64) -> io::Result<()> {
    let tracks = group_by_program(notes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
    write_midi_file_from_tracks(path, &tracks, tempo_bpm)
}

pub fn write_midi_file_from_tracks(
    path: &Path,
    tracks: &[Track],
    tempo_bpm: f64,
) -> std::io::Result<()> {
    std::fs::write(path, encode_tracks(tracks, tempo_bpm))
}

/// The same file as [`write_midi_file`], as bytes -- for a host that hands them
/// to the user rather than writing a path.
pub fn midi_bytes(notes: &[NoteEvent], tempo_bpm: f64) -> Result<Vec<u8>, TrackError> {
    let tracks = group_by_program(notes)?;
    Ok(encode_tracks(&tracks, tempo_bpm))
}

fn encode_tracks(tracks: &[Track], tempo_bpm: f64) -> Vec<u8> {
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

    bytes
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

/// Write a Standard MIDI File variable-length quantity.
///
/// The value is split into seven-bit groups, most significant group first, with
/// the high bit set on every byte except the last. A u64 needs at most ten
/// groups.
fn write_vlq(out: &mut Vec<u8>, value: u64) {
    let mut groups = [0u8; 10];
    let mut len = 0;
    let mut rest = value;

    loop {
        groups[len] = (rest as u8) & 0x7F;
        len += 1;
        rest >>= 7;
        if rest == 0 {
            break;
        }
    }

    // The final byte emitted is the least significant group, and carries no
    // continuation bit.
    for index in (0..len).rev() {
        out.push(if index == 0 {
            groups[0]
        } else {
            groups[index] | 0x80
        });
    }
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
        let tracks = group_by_program(&notes).unwrap();

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
    fn a_melodic_note_with_program_128_is_not_drums() {
        // 128 belongs to no instrument group, so a melodic note can carry it.
        // Reading is_drum off the program would file this on the percussion
        // channel, where the pitch would select a drum instead.
        let notes = vec![NoteEvent {
            onset: 0.0,
            offset: 1.0,
            pitch: 60,
            program: 128,
            is_drum: false,
        }];
        let tracks = group_by_program(&notes).unwrap();

        assert_eq!(tracks.len(), 1);
        assert!(!tracks[0].is_drum());
        assert_ne!(tracks[0].channel, DRUM_CHANNEL);
    }

    #[test]
    fn channels_are_assigned_in_order_and_never_collide() {
        let notes: Vec<NoteEvent> = (0..MELODIC_CHANNEL_COUNT as u16)
            .map(|program| note(0.0, 1.0, 60, program))
            .collect();
        let tracks = group_by_program(&notes).unwrap();

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
    fn more_programs_than_melodic_channels_is_refused_rather_than_shared() {
        // Two programs on one channel would each write a program change and
        // only the last would survive, silently playing one in the other's
        // voice. Refuse instead of corrupting.
        let notes: Vec<NoteEvent> = (0..40u16).map(|p| note(0.0, 1.0, 60, p)).collect();
        let error = group_by_program(&notes).unwrap_err();
        assert_eq!(
            error,
            TrackError::TooManyInstruments {
                instruments: 40,
                channels: 15
            }
        );
    }

    #[test]
    fn grouping_is_deterministic() {
        let notes = vec![
            note(1.0, 2.0, 60, 40),
            note(0.0, 1.0, 62, 0),
            note(0.5, 1.5, 64, 0),
        ];
        assert_eq!(
            group_by_program(&notes).unwrap(),
            group_by_program(&notes).unwrap()
        );
    }

    #[test]
    fn notes_in_a_track_are_ordered_by_onset() {
        let notes = vec![
            note(2.0, 3.0, 60, 0),
            note(0.0, 1.0, 62, 0),
            note(1.0, 1.5, 64, 0),
        ];
        let tracks = group_by_program(&notes).unwrap();
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
            // Three groups with unequal higher bytes: the encoder used to emit
            // them least-significant-first, which decoded to a different value.
            (0x4000, vec![0x81u8, 0x80, 0x00]),
            (0x8000, vec![0x82u8, 0x80, 0x00]),
            (100_000, vec![0x86u8, 0x8D, 0x20]),
            (0x200000, vec![0x81u8, 0x80, 0x80, 0x00]),
        ] {
            let mut out = Vec::new();
            write_vlq(&mut out, value);
            assert_eq!(out, expected, "value {value:#x}");
        }
    }

    #[test]
    fn vlq_round_trips_every_value_through_a_decoder() {
        // Decode the written bytes back to a value: seven-bit groups, most
        // significant first. Any mis-ordered group shows up immediately.
        let decode = |bytes: &[u8]| -> u64 {
            bytes
                .iter()
                .fold(0u64, |acc, byte| (acc << 7) | u64::from(byte & 0x7F))
        };

        for value in (0u64..70_000).chain([0x1FFFFF, 0x200000, u32::MAX as u64]) {
            let mut out = Vec::new();
            write_vlq(&mut out, value);
            assert_eq!(decode(&out), value, "value {value:#x} -> {out:02X?}");
            // Every byte but the last must carry the continuation bit.
            for byte in &out[..out.len() - 1] {
                assert_ne!(byte & 0x80, 0, "missing continuation bit in {out:02X?}");
            }
            assert_eq!(
                out[out.len() - 1] & 0x80,
                0,
                "trailing bit set in {out:02X?}"
            );
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
        assert!(group_by_program(&[]).unwrap().is_empty());
        assert_eq!(duration_secs(&[]), 0.0);
    }
}
