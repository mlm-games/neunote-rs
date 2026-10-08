//! Reads back what the writer produced with `midly`, so the tests check a real
//! MIDI file rather than the writer's own idea of one.

use std::path::Path;

use neunote_midi::{
    DEFAULT_BPM, DRUM_CHANNEL, PPQ, Track, group_by_program, write_midi_file,
    write_midi_file_from_tracks,
};
use neunote_types::{DRUM_PROGRAM, MIDI_VELOCITY, NoteEvent};

fn note(onset: f64, offset: f64, pitch: u8, program: u16) -> NoteEvent {
    NoteEvent::new(onset, offset, pitch, program, program == DRUM_PROGRAM)
}

/// One channel-bound event, flattened out of midly's two-level shape and given
/// an absolute tick so ordering is testable.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Event {
    tick: u32,
    channel: u8,
    message: midly::MidiMessage,
}

impl Event {
    fn is_sounding(&self) -> bool {
        matches!(self.message, midly::MidiMessage::NoteOn { vel, .. } if u8::from(vel) > 0)
    }

    fn key(&self) -> Option<u8> {
        match self.message {
            midly::MidiMessage::NoteOn { key, .. } | midly::MidiMessage::NoteOff { key, .. } => {
                Some(u8::from(key))
            }
            _ => None,
        }
    }

    fn program(&self) -> Option<u8> {
        match self.message {
            midly::MidiMessage::ProgramChange { program } => Some(u8::from(program)),
            _ => None,
        }
    }
}

/// A parsed file. Meta events stay on the track so tempo can be checked; they
/// carry no channel, so they are kept in a separate list.
struct Parsed {
    format: midly::Format,
    timing: midly::Timing,
    tracks: Vec<Vec<Event>>,
    meta: Vec<Vec<midly::MetaMessage<'static>>>,
}

fn parse(bytes: &[u8]) -> Parsed {
    let file = midly::Smf::parse(bytes).expect("parsing the written file");

    let mut tracks = Vec::with_capacity(file.tracks.len());
    let mut meta = Vec::with_capacity(file.tracks.len());

    for track in &file.tracks {
        let mut events = Vec::new();
        let mut metas = Vec::new();
        let mut tick = 0u32;

        for event in track {
            tick += u32::from(event.delta);
            match &event.kind {
                midly::TrackEventKind::Midi { channel, message } => events.push(Event {
                    tick,
                    channel: u8::from(*channel),
                    message: *message,
                }),
                midly::TrackEventKind::Meta(message) => metas.push(message.to_static()),
                _ => {}
            }
        }

        tracks.push(events);
        meta.push(metas);
    }

    Parsed {
        format: file.header.format,
        timing: file.header.timing,
        tracks,
        meta,
    }
}

fn write(notes: &[NoteEvent]) -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.mid");
    write_midi_file(&path, notes, DEFAULT_BPM).unwrap();
    std::fs::read(path).unwrap()
}

fn sounding(track: &[Event]) -> Vec<(u8, u32)> {
    track
        .iter()
        .filter(|event| event.is_sounding())
        .filter_map(|event| event.key().map(|key| (key, event.tick)))
        .collect()
}

fn program_of(track: &[Event]) -> Option<u8> {
    track.iter().find_map(Event::program)
}

fn ticks_per_beat(timing: midly::Timing) -> u16 {
    match timing {
        midly::Timing::Metrical(ticks) => u16::from(ticks),
        other => panic!("expected ticks-per-beat timing, got {other:?}"),
    }
}

fn tempo_of(track: &[midly::MetaMessage<'static>]) -> Option<u32> {
    track.iter().find_map(|message| match message {
        midly::MetaMessage::Tempo(tempo) => Some(u32::from(*tempo)),
        _ => None,
    })
}

#[test]
fn a_piano_etude_becomes_a_conductor_track_plus_one_track() {
    let notes = vec![
        note(0.0, 0.5, 60, 0),
        note(0.5, 1.0, 62, 0),
        note(1.0, 2.0, 64, 0),
    ];
    let parsed = parse(&write(&notes));

    assert_eq!(
        parsed.format,
        midly::Format::Parallel,
        "format 1 keeps instruments separate"
    );
    assert_eq!(ticks_per_beat(parsed.timing), PPQ);
    assert_eq!(parsed.tracks.len(), 2, "conductor plus piano");

    // The conductor track carries tempo and the end marker, no notes.
    assert!(sounding(&parsed.tracks[0]).is_empty());

    let ons = sounding(&parsed.tracks[1]);
    assert_eq!(
        ons.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
        vec![60, 62, 64],
        "three onsets, in order"
    );
    assert_eq!(
        ons.iter().map(|(_, tick)| *tick).collect::<Vec<_>>(),
        vec![0, 480, 960],
        "half a second apart at 120 BPM"
    );

    let offs: Vec<u32> = parsed.tracks[1]
        .iter()
        .filter(|event| matches!(event.message, midly::MidiMessage::NoteOff { .. }))
        .map(|event| event.tick)
        .collect();
    assert_eq!(offs, vec![480, 960, 1920]);

    assert_eq!(program_of(&parsed.tracks[1]), Some(0));
}

#[test]
fn onsets_land_on_the_expected_ticks() {
    // At 120 BPM a beat is PPQ ticks, so a tick is 1/960 s.
    let notes = vec![
        note(0.0, 0.5, 60, 0),
        note(1.0, 1.5, 62, 0),
        note(2.0, 2.5, 64, 0),
    ];
    let parsed = parse(&write(&notes));
    let ons = sounding(&parsed.tracks[1]);

    assert_eq!(
        ons.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
        vec![60, 62, 64]
    );
    assert_eq!(
        ons.iter().map(|(_, tick)| *tick).collect::<Vec<_>>(),
        vec![0, 960, 1920]
    );
}

#[test]
fn drums_land_on_channel_ten_and_carry_no_program_change() {
    let notes = vec![
        note(0.0, 1.0, 60, 0),
        note(0.0, 0.1, 36, DRUM_PROGRAM),
        note(0.2, 0.3, 42, DRUM_PROGRAM),
    ];
    let parsed = parse(&write(&notes));

    assert_eq!(parsed.tracks.len(), 3);

    let drum = &parsed.tracks[2];
    assert!(
        drum.iter().all(|event| event.channel == DRUM_CHANNEL),
        "every drum event is on channel 10"
    );
    assert_eq!(
        program_of(drum),
        None,
        "the percussion channel must not select a program"
    );
    assert_eq!(
        sounding(drum)
            .iter()
            .map(|(key, _)| *key)
            .collect::<Vec<_>>(),
        vec![36, 42]
    );

    for event in drum {
        if let midly::MidiMessage::NoteOn { vel, .. } = event.message {
            assert_eq!(u8::from(vel), MIDI_VELOCITY);
        }
    }
}

#[test]
fn every_instrument_keeps_its_own_program_in_its_own_track() {
    let notes = vec![
        note(0.0, 1.0, 60, 0),  // acoustic_piano
        note(0.0, 1.0, 40, 40), // violin
        note(0.0, 1.0, 36, 32), // acoustic_bass
        note(0.0, 0.1, 38, DRUM_PROGRAM),
    ];
    let parsed = parse(&write(&notes));

    assert_eq!(parsed.tracks.len(), 5, "conductor plus four instruments");

    let programs: Vec<Option<u8>> = parsed.tracks[1..].iter().map(|t| program_of(t)).collect();
    // Sorted by program, drums last, so: 0, 32, 40, and drums with none.
    assert_eq!(programs, vec![Some(0), Some(32), Some(40), None]);
}

#[test]
fn melodic_tracks_never_land_on_the_drum_channel() {
    let notes: Vec<NoteEvent> = (0..20u16).map(|p| note(0.0, 1.0, 60, p)).collect();
    let parsed = parse(&write(&notes));

    for track in &parsed.tracks[1..] {
        assert!(
            track.iter().all(|event| event.channel != DRUM_CHANNEL),
            "a melodic track claimed the percussion channel"
        );
    }
}

#[test]
fn note_offs_precede_note_ons_at_the_same_tick() {
    // Two consecutive notes on one pitch: the first must release before the
    // second strikes, or the pair never sounds separate.
    let notes = vec![note(0.0, 0.5, 60, 0), note(0.5, 1.0, 60, 0)];
    let parsed = parse(&write(&notes));

    let keys: Vec<(u8, bool)> = parsed.tracks[1]
        .iter()
        .filter_map(|event| {
            let key = event.key()?;
            match event.message {
                midly::MidiMessage::NoteOn { vel, .. } if u8::from(vel) > 0 => Some((key, false)),
                midly::MidiMessage::NoteOff { .. } => Some((key, true)),
                _ => None,
            }
        })
        .collect();

    assert_eq!(
        keys,
        vec![(60, false), (60, true), (60, false), (60, true)],
        "off and on must interleave on one pitch"
    );
}

#[test]
fn an_empty_transcription_writes_a_valid_file_with_no_note_tracks() {
    let parsed = parse(&write(&[]));
    assert_eq!(parsed.format, midly::Format::Parallel);
    assert_eq!(ticks_per_beat(parsed.timing), PPQ);
    assert_eq!(parsed.tracks.len(), 1, "just the conductor track");
}

#[test]
fn writing_prebuilt_tracks_matches_writing_notes() {
    let notes = vec![note(0.0, 1.0, 60, 0), note(0.0, 0.1, 36, DRUM_PROGRAM)];
    let dir = tempfile::tempdir().unwrap();

    let from_notes = dir.path().join("a.mid");
    write_midi_file(&from_notes, &notes, DEFAULT_BPM).unwrap();

    let from_tracks = dir.path().join("b.mid");
    write_midi_file_from_tracks(&from_tracks, &group_by_program(&notes), DEFAULT_BPM).unwrap();

    assert_eq!(
        std::fs::read(&from_notes).unwrap(),
        std::fs::read(&from_tracks).unwrap(),
        "the two entry points must agree byte for byte"
    );
}

#[test]
fn the_tempo_in_the_header_is_the_one_requested() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("slow.mid");
    write_midi_file(&path, &[note(0.0, 1.0, 60, 0)], 60.0).unwrap();

    let parsed = parse(&std::fs::read(path).unwrap());
    assert_eq!(
        tempo_of(&parsed.meta[0]),
        Some(1_000_000),
        "60 BPM is a million microseconds a beat"
    );
}

#[test]
fn the_default_tempo_is_120_bpm() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("default.mid");
    write_midi_file(&path, &[note(0.0, 1.0, 60, 0)], DEFAULT_BPM).unwrap();

    let parsed = parse(&std::fs::read(path).unwrap());
    assert_eq!(tempo_of(&parsed.meta[0]), Some(500_000));
}

#[test]
fn a_nonsense_tempo_falls_back_to_the_default() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.mid");
    write_midi_file(&path, &[note(0.0, 1.0, 60, 0)], -5.0).unwrap();

    let parsed = parse(&std::fs::read(path).unwrap());
    assert_eq!(
        tempo_of(&parsed.meta[0]),
        Some(500_000),
        "a negative BPM falls back"
    );
}

#[test]
fn a_track_keeps_the_notes_it_was_given() {
    let track = Track {
        program: 0,
        name: "acoustic_piano".to_owned(),
        channel: 0,
        notes: vec![note(0.0, 1.0, 60, 0), note(1.0, 2.0, 64, 0)],
    };
    assert_eq!(track.note_count(), 2);
    assert!((track.duration_secs() - 2.0).abs() < 1e-9);
    assert!(!track.is_drum());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("track.mid");
    write_midi_file_from_tracks(&path, std::slice::from_ref(&track), DEFAULT_BPM).unwrap();
    assert!(path.exists());
}

#[test]
fn writing_to_a_missing_directory_fails_rather_than_panicking() {
    let missing = Path::new("/nonexistent-neunote-test-dir").join("out.mid");
    let result = write_midi_file(&missing, &[note(0.0, 1.0, 60, 0)], DEFAULT_BPM);
    assert!(result.is_err());
}
