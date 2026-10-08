//! `NoteAssembler` and `trim_overlapping_notes`, ported from
//! `cpp/src/note_assembler.cpp`.

use std::collections::VecDeque;

use neunote_types::{
    DRUM_PROGRAM, GroupId, MIN_NOTE_DURATION_SECS, NoteEvent, NoteKey, sort_notes, validate_notes,
};

use crate::{ActionKind, NoteAction};

/// A note plus the chunk it closed in, so streaming can report the notes a
/// window owns without waiting for the whole file.
#[derive(Debug, Clone, PartialEq)]
struct TrackedNote {
    note: NoteEvent,
    chunk_index: u32,
    /// Keyed on the *token* program, before program 96 is routed to drums.
    key: NoteKey,
}

/// `trim_overlapping_notes`: within each `(program, pitch, is_drum)` channel,
/// stable-sort by onset alone, cut each offset back to the next note's onset,
/// drop what is left empty. Then sort by the full five-key comparator.
///
/// Drums share the `(program, pitch)` space with melodic notes -- both can be
/// program 128 -- so `is_drum` is part of the key rather than implied by it.
pub fn trim_overlapping_notes(notes: &[NoteEvent]) -> Vec<NoteEvent> {
    let tracked: Vec<TrackedNote> = notes
        .iter()
        .map(|note| TrackedNote {
            note: *note,
            chunk_index: 0,
            key: NoteKey::new(i32::from(note.program), i32::from(note.pitch)),
        })
        .collect();

    trim_tracked(&tracked)
        .into_iter()
        .map(|tracked| tracked.note)
        .collect()
}

fn channel(note: &TrackedNote) -> (u16, u8, bool) {
    (note.note.program, note.note.pitch, note.note.is_drum)
}

fn trim_tracked(notes: &[TrackedNote]) -> Vec<TrackedNote> {
    if notes.len() <= 1 {
        return notes.to_vec();
    }

    // Group by channel while preserving the master (close) order inside each
    // group: the reference filters the master list per channel and then does a
    // *stable* sort on onset alone, so equal onsets keep close order. Sorting
    // by the full five-key comparator here instead would change which of two
    // coincident notes gets truncated.
    let mut grouped: Vec<TrackedNote> = notes.to_vec();
    grouped.sort_by_key(channel);

    let mut trimmed = Vec::with_capacity(grouped.len());
    let mut start = 0;
    while start < grouped.len() {
        let mut end = start + 1;
        while end < grouped.len() && channel(&grouped[end]) == channel(&grouped[start]) {
            end += 1;
        }

        let mut group = grouped[start..end].to_vec();
        group.sort_by(|a, b| a.note.onset.total_cmp(&b.note.onset));

        for index in 1..group.len() {
            let (previous, next) = group.split_at_mut(index);
            let previous = previous.last_mut().expect("non-empty");
            if previous.note.offset > next[0].note.onset {
                previous.note.offset = next[0].note.onset;
            }
        }

        trimmed.extend(
            group
                .into_iter()
                .filter(|tracked| tracked.note.onset < tracked.note.offset),
        );
        start = end;
    }

    sort_notes_tracked(&mut trimmed);
    trimmed
}

fn sort_notes_tracked(notes: &mut [TrackedNote]) {
    notes.sort_by(|a, b| {
        a.note
            .onset
            .total_cmp(&b.note.onset)
            .then_with(|| a.note.is_drum.cmp(&b.note.is_drum))
            .then_with(|| a.note.program.cmp(&b.note.program))
            .then_with(|| a.note.pitch.cmp(&b.note.pitch))
            .then_with(|| a.note.offset.total_cmp(&b.note.offset))
    });
}

fn validate_tracked(notes: &mut [TrackedNote]) {
    let mut plain: Vec<NoteEvent> = notes.iter().map(|tracked| tracked.note).collect();
    validate_notes(&mut plain);
    for (tracked, note) in notes.iter_mut().zip(plain) {
        tracked.note = note;
    }
}

/// Collects the tracker's actions into finished notes.
#[derive(Debug, Default)]
pub struct NoteAssembler {
    closed: Vec<TrackedNote>,
    open: VecDeque<TrackedNote>,
}

impl NoteAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.closed.clear();
        self.open.clear();
    }

    pub fn apply(&mut self, actions: &[NoteAction], chunk_index: u32) -> Result<(), String> {
        for action in actions {
            match action.kind {
                ActionKind::Start => {
                    // Resolved through the group, not copied from the token: a
                    // decoded program 96 names itself "drums" upstream and is
                    // routed as one here. It changes the trimming channel, so
                    // doing it at label time instead gives a different note list.
                    let is_drum =
                        GroupId::for_program(action.program as u16) == Some(GroupId::DRUMS);
                    let note = NoteEvent {
                        onset: action.time,
                        offset: action.time,
                        pitch: action.pitch.clamp(0, 127) as u8,
                        program: if is_drum {
                            DRUM_PROGRAM
                        } else {
                            action.program as u16
                        },
                        is_drum,
                    };
                    self.open.push_back(TrackedNote {
                        note,
                        chunk_index,
                        key: NoteKey::new(action.program, action.pitch),
                    });
                }

                ActionKind::End => {
                    let key = NoteKey::new(action.program, action.pitch);
                    let Some(index) = self.open.iter().position(|tracked| tracked.key == key)
                    else {
                        return Err(format!(
                            "note end for (program {}, pitch {}) with nothing open",
                            action.program, action.pitch
                        ));
                    };
                    let mut closed = self.open.remove(index).expect("index just found");
                    closed.note.offset = action.time;
                    closed.chunk_index = chunk_index;
                    self.closed.push(closed);
                }

                ActionKind::DrumHit => {
                    self.closed.push(TrackedNote {
                        note: NoteEvent {
                            onset: action.time,
                            offset: action.time + MIN_NOTE_DURATION_SECS,
                            pitch: action.pitch.clamp(0, 127) as u8,
                            program: DRUM_PROGRAM,
                            is_drum: true,
                        },
                        chunk_index,
                        key: NoteKey::new(i32::from(DRUM_PROGRAM), action.pitch),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn finalize(&self) -> Vec<NoteEvent> {
        let mut notes = self.closed.clone();
        validate_tracked(&mut notes);
        trim_tracked(&notes)
            .into_iter()
            .map(|tracked| tracked.note)
            .collect()
    }

    /// The notes this window owns. `chunk_index` and `chunk_index + 1` are
    /// trimmed together, then only `chunk_index` is reported -- a note that
    /// closed in the next chunk can still truncate one from this one.
    pub fn closed_in(&self, chunk_index: u32) -> Vec<NoteEvent> {
        let mut window: Vec<TrackedNote> = self
            .closed
            .iter()
            .filter(|tracked| {
                tracked.chunk_index == chunk_index || tracked.chunk_index == chunk_index + 1
            })
            .cloned()
            .collect();

        validate_tracked(&mut window);
        trim_tracked(&window)
            .into_iter()
            .filter(|tracked| tracked.chunk_index == chunk_index)
            .map(|tracked| tracked.note)
            .collect()
    }
}

#[allow(dead_code)]
fn sort_plain(notes: &mut [NoteEvent]) {
    sort_notes(notes);
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
            is_drum: false,
        }
    }

    #[test]
    fn trim_cuts_offsets_back_to_the_next_onset() {
        let notes = vec![
            note(0.0, 1.0, 60, 0),
            note(0.5, 0.8, 60, 0),
            note(1.0, 2.0, 60, 0),
        ];
        let trimmed = trim_overlapping_notes(&notes);
        assert_eq!(trimmed.len(), 3);
        assert_eq!(trimmed[0].offset, 0.5);
        assert_eq!(trimmed[1].offset, 0.8);
        assert_eq!(trimmed[2].offset, 2.0);
    }

    #[test]
    fn trim_drops_notes_left_empty() {
        let notes = vec![note(0.0, 1.0, 60, 0), note(0.0, 0.5, 60, 0)];
        let trimmed = trim_overlapping_notes(&notes);
        assert_eq!(trimmed.len(), 1);
        assert_eq!((trimmed[0].onset, trimmed[0].offset), (0.0, 0.5));
    }

    #[test]
    fn trim_separates_drums_from_melodic_notes_sharing_a_program() {
        // Program 128 with is_drum false is a legal melodic note, so it must
        // not be trimmed against a drum hit on the same pitch.
        let notes = vec![
            NoteEvent {
                onset: 0.0,
                offset: 1.0,
                pitch: 36,
                program: DRUM_PROGRAM,
                is_drum: true,
            },
            NoteEvent {
                onset: 0.2,
                offset: 0.4,
                pitch: 36,
                program: DRUM_PROGRAM,
                is_drum: false,
            },
        ];
        assert_eq!(trim_overlapping_notes(&notes).len(), 2);
    }

    #[test]
    fn trim_does_not_merge_different_pitches() {
        let notes = vec![note(0.0, 1.0, 60, 0), note(0.5, 0.9, 61, 0)];
        assert_eq!(trim_overlapping_notes(&notes).len(), 2);
    }
}
