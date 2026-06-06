use std::io::Write;

use crate::midi::events::NoteEvent;

/// Write a standard MIDI file (format 1) from note events
pub fn write_midi_file<W: Write>(
    writer: &mut W,
    events: &[NoteEvent],
    bpm: f64,
) -> std::io::Result<()> {
    let ticks_per_qn: u16 = 480;
    let microseconds_per_qn = (60.0 / bpm * 1_000_000.0) as u32;

    // --- Header chunk ---
    write_chunk(writer, b"MThd", &{
        let mut h = Vec::with_capacity(6);
        // format 1 (multiple tracks)
        h.extend_from_slice(&2u16.to_be_bytes());
        // number of tracks: tempo track + note track
        h.extend_from_slice(&2u16.to_be_bytes());
        h.extend_from_slice(&ticks_per_qn.to_be_bytes());
        h
    })?;

    // --- Track 1: Tempo map ---
    let tempo_track = build_tempo_track(microseconds_per_qn, ticks_per_qn);
    write_chunk(writer, b"MTrk", &tempo_track)?;

    // --- Track 2: Notes ---
    let note_track = build_note_track(events, ticks_per_qn, bpm);
    write_chunk(writer, b"MTrk", &note_track)?;

    Ok(())
}

fn write_chunk<W: Write>(writer: &mut W, id: &[u8; 4], data: &[u8]) -> std::io::Result<()> {
    writer.write_all(id)?;
    writer.write_all(&(data.len() as u32).to_be_bytes())?;
    writer.write_all(data)
}

fn build_tempo_track(microseconds_per_qn: u32, _ticks_per_qn: u16) -> Vec<u8> {
    let mut track = vec![
        0,       // delta time
        0xFF,    // meta event
        0x51,    // set tempo
        0x03,    // length
    ];
    track.extend_from_slice(&microseconds_per_qn.to_be_bytes()[1..]); // 3 bytes

    // Time signature: 4/4
    track.extend_from_slice(&[
        0,       // delta time
        0xFF,    // meta
        0x58,    // time signature
        0x04,    // length
        4,       // numerator
        4,       // denominator
        24,      // clocks per click
        8,       // 32nd notes per quarter
    ]);

    // End of track
    track.extend_from_slice(&[
        0,       // delta
        0xFF,    // meta
        0x2F,    // end of track
        0x00,    // length
    ]);

    track
}

fn build_note_track(events: &[NoteEvent], ticks_per_qn: u16, bpm: f64) -> Vec<u8> {
    let mut track = Vec::new();
    let ticks_per_sec = ticks_per_qn as f64 * bpm / 60.0;

    let mut sorted = events.to_vec();
    sorted.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

    // Write program change to piano (acoustic grand)
    // track: program change, channel 0
    track.push(0); // delta
    track.push(0xC0); // program change
    track.push(0x00); // piano

    // Build a sorted list of (tick, type, pitch, velocity) events
    // type=0 for note-on, type=1 for note-off
    struct MidiEvent {
        tick: u64,
        is_on: bool,
        pitch: u8,
        velocity: u8,
    }

    let mut midi_events: Vec<MidiEvent> = Vec::with_capacity(events.len() * 2);
    for event in &sorted {
        let start_tick = (event.start_time * ticks_per_sec) as u64;
        let end_tick = (event.end_time * ticks_per_sec) as u64;

        if end_tick <= start_tick {
            continue;
        }

        midi_events.push(MidiEvent {
            tick: start_tick,
            is_on: true,
            pitch: event.pitch,
            velocity: (event.amplitude * 127.0).min(127.0) as u8,
        });
        midi_events.push(MidiEvent {
            tick: end_tick,
            is_on: false,
            pitch: event.pitch,
            velocity: 64,
        });
    }

    // Sort by tick, then note-off before note-on for same tick
    midi_events.sort_by(|a, b| {
        a.tick
            .cmp(&b.tick)
            .then_with(|| a.is_on.cmp(&b.is_on))
    });

    let mut current_tick: u64 = 0;
    for ev in &midi_events {
        let delta = ev.tick - current_tick;
        write_vlq(&mut track, delta as u32);
        if ev.is_on {
            track.push(0x90);
            track.push(ev.pitch);
            track.push(ev.velocity);
        } else {
            track.push(0x80);
            track.push(ev.pitch);
            track.push(64);
        }
        current_tick = ev.tick;
    }

    // End of track
    write_vlq(&mut track, 0);
    track.push(0xFF);
    track.push(0x2F);
    track.push(0x00);

    track
}

/// Write a variable-length quantity (VLQ) value (MIDI standard)
fn write_vlq(track: &mut Vec<u8>, mut value: u32) {
    if value < 0x80 {
        track.push(value as u8);
        return;
    }
    let mut buf = [0u8; 5];
    let mut i = 0;
    while value > 0 {
        buf[i] = (value & 0x7F) as u8;
        value >>= 7;
        i += 1;
    }
    for j in (0..i).rev() {
        if j > 0 {
            track.push(buf[j] | 0x80);
        } else {
            track.push(buf[j]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_polyphonic_midi() {
        // Two overlapping notes (polyphony)
        let events = vec![
            NoteEvent {
                start_time: 0.0,
                end_time: 2.0,
                pitch: 60,
                amplitude: 0.8,
                ..Default::default()
            },
            NoteEvent {
                start_time: 0.5,
                end_time: 1.5,
                pitch: 64,
                amplitude: 0.8,
                ..Default::default()
            },
        ];
        let mut buf = Vec::new();
        write_midi_file(&mut buf, &events, 120.0).unwrap();
        // Should contain two note-on events
        let note_on_count = buf.iter().filter(|&&b| b == 0x90).count();
        assert_eq!(note_on_count, 2, "Should have 2 note-on events");
        let note_off_count = buf.iter().filter(|&&b| b == 0x80).count();
        assert_eq!(note_off_count, 2, "Should have 2 note-off events");
    }
}
