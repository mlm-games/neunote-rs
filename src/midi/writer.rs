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
    let mut track = Vec::new();

    // Set tempo
    track.push(0); // delta time
    track.push(0xFF); // meta event
    track.push(0x51); // set tempo
    track.push(0x03); // length
    track.extend_from_slice(&microseconds_per_qn.to_be_bytes()[1..]); // 3 bytes

    // Time signature: 4/4
    track.push(0); // delta time
    track.push(0xFF); // meta
    track.push(0x58); // time signature
    track.push(0x04); // length
    track.push(4); // numerator
    track.push(4); // denominator (2 = quarter note gets the beat)
    track.push(24); // clocks per click
    track.push(8); // 32nd notes per quarter

    // End of track
    track.push(0); // delta
    track.push(0xFF); // meta
    track.push(0x2F); // end of track
    track.push(0x00); // length

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

    let mut current_tick: u64 = 0;

    for event in &sorted {
        let start_tick = (event.start_time * ticks_per_sec) as u64;
        let end_tick = (event.end_time * ticks_per_sec) as u64;

        if start_tick < current_tick {
            // skip overlapping note-ons that would be in the past
            continue;
        }

        // Note on
        let delta_on = start_tick - current_tick;
        write_vlq(&mut track, delta_on as u32);
        track.push(0x90); // note on, channel 0
        track.push(event.pitch);
        track.push((event.amplitude * 127.0) as u8);

        // Note off (no delta between note-on/off pair in sequence)
        let delta_off = if end_tick > start_tick {
            end_tick - start_tick
        } else {
            1
        };
        write_vlq(&mut track, delta_off as u32);
        track.push(0x80); // note off, channel 0
        track.push(event.pitch);
        track.push(0x40);

        current_tick = start_tick + delta_off;
    }

    // End of track
    write_vlq(&mut track, 0);
    track.push(0xFF);
    track.push(0x2F);
    track.push(0x00);

    track
}

/// Write a variable-length quantity (VLQ) value
fn write_vlq(track: &mut Vec<u8>, mut value: u32) {
    if value == 0 {
        track.push(0);
        return;
    }
    let mut bytes = [0u8; 4];
    let mut i = 0;
    while value > 0 {
        bytes[i] = (value & 0x7F) as u8;
        value >>= 7;
        i += 1;
    }
    for b in bytes[..i].iter().rev() {
        if b != &bytes[i - 1] {
            track.push(b | 0x80);
        } else {
            track.push(*b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_simple_midi() {
        let events = vec![
            NoteEvent {
                start_time: 0.0,
                end_time: 1.0,
                pitch: 60,
                amplitude: 0.8,
                ..Default::default()
            },
            NoteEvent {
                start_time: 1.0,
                end_time: 2.0,
                pitch: 64,
                amplitude: 0.8,
                ..Default::default()
            },
        ];
        let mut buf = Vec::new();
        write_midi_file(&mut buf, &events, 120.0).unwrap();
        // Should have MThd header + 2 MTrk chunks
        assert!(buf.len() > 22, "MIDI file too short");
        assert_eq!(&buf[0..4], b"MThd");
        // Check that second track starts with MTrk
        let mtrk_start = buf.windows(4).position(|w| w == b"MTrk").unwrap();
        assert!(mtrk_start > 0);
    }
}
