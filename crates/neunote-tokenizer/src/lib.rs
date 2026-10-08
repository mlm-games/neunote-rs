#![forbid(unsafe_code)]

//! The MT3 vocabulary and its decode state machine.
//!
//! A literal port of `cpp/src/vocabulary.cpp`, `cpp/src/open_note_tracker.cpp`
//! and `cpp/src/note_assembler.cpp`. Integer logic with no learned parameters,
//! so this crate needs no model weights and no ML runtime.

use std::collections::VecDeque;

use neunote_types::{FRAME_RATE, MIN_NOTE_DURATION_SECS, NoteKey};

mod assembly;
mod instruments;
mod vocab;

pub use assembly::{NoteAssembler, trim_overlapping_notes};
pub use instruments::{conditioning_rows, forbidden_token_ids};
pub use vocab::{
    BOS_ID, DRUM_COUNT, DRUM_FIRST, EOS_ID, EventType, MAX_SHIFT_STEPS, NUM_TOKENS, PAD_ID,
    PITCH_COUNT, PITCH_FIRST, PROGRAM_COUNT, PROGRAM_FIRST, SHIFT_FIRST, TIE_FIRST, TIE_FIRST_ID,
    TokenEvent, UNK_ID, VELOCITY_COUNT, VELOCITY_FIRST, event_for, tie_section_tokens, token_for,
};

/// Where one chunk begins and ends. The last chunk has no `next_seek_time`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkBoundary {
    pub seek_time: f64,
    pub next_seek_time: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    Start,
    End,
    DrumHit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoteAction {
    pub kind: ActionKind,
    pub program: i32,
    pub pitch: i32,
    pub time: f64,
}

/// Turns a chunk's token ids into note starts, ends and drum hits.
///
/// The registers reset at every chunk boundary; the open-note map does not.
/// That asymmetry is the whole cross-chunk mechanism.
#[derive(Debug, Default)]
pub struct OpenNoteTracker {
    /// Insertion order matters: `finish` emits ends in it, and a hash map would
    /// reorder the output events.
    open: VecDeque<(NoteKey, f64)>,
    seek_time: f64,
    next_seek_time: Option<f64>,
    start_tick: i32,
    tick_state: i32,
    program: Option<i32>,
    velocity: Option<i32>,
    in_prologue: bool,
    skip_rest: bool,
    chunk_started: bool,
    tie_set: Vec<NoteKey>,
}

impl OpenNoteTracker {
    pub fn new() -> Self {
        Self {
            open: VecDeque::new(),
            seek_time: 0.0,
            next_seek_time: None,
            start_tick: 0,
            tick_state: 0,
            program: None,
            velocity: None,
            in_prologue: true,
            skip_rest: false,
            chunk_started: false,
            tie_set: Vec::new(),
        }
    }

    pub fn open_keys(&self) -> Vec<NoteKey> {
        let mut keys: Vec<NoteKey> = self.open.iter().map(|(key, _)| *key).collect();
        keys.sort_unstable();
        keys
    }

    fn end_all(&mut self, at: f64) -> Vec<NoteAction> {
        let actions = self
            .open
            .iter()
            .map(|(key, _)| NoteAction {
                kind: ActionKind::End,
                program: key.program,
                pitch: key.pitch,
                time: at,
            })
            .collect();
        self.open.clear();
        actions
    }

    /// Begin a chunk. A previous chunk that never closed its tie prologue is
    /// malformed: it declared nothing, so its open notes end at *its* boundary.
    pub fn feed_boundary(&mut self, boundary: ChunkBoundary) -> Vec<NoteAction> {
        let actions = if self.chunk_started && self.in_prologue {
            self.end_all(self.seek_time)
        } else {
            Vec::new()
        };

        self.seek_time = boundary.seek_time;
        self.next_seek_time = boundary.next_seek_time;
        self.start_tick = (boundary.seek_time * f64::from(FRAME_RATE)).round() as i32;
        self.tick_state = self.start_tick;
        self.program = None;
        self.velocity = None;
        self.in_prologue = true;
        self.skip_rest = false;
        self.chunk_started = true;
        self.tie_set.clear();
        actions
    }

    pub fn feed_token(&mut self, token: i32) -> Vec<NoteAction> {
        if self.skip_rest {
            return Vec::new();
        }
        let event = event_for(token);
        if self.in_prologue {
            self.feed_prologue(event)
        } else {
            self.feed_body(event)
        }
    }

    fn feed_prologue(&mut self, event: TokenEvent) -> Vec<NoteAction> {
        match event.r#type {
            EventType::Tie => {
                // End of the tie section: everything not re-declared stops here.
                self.in_prologue = false;
                self.velocity = None;

                let mut actions = Vec::new();
                let mut kept = VecDeque::with_capacity(self.open.len());
                for (key, onset) in self.open.drain(..) {
                    if self.tie_set.contains(&key) {
                        kept.push_back((key, onset));
                    } else {
                        actions.push(NoteAction {
                            kind: ActionKind::End,
                            program: key.program,
                            pitch: key.pitch,
                            time: self.seek_time,
                        });
                    }
                }
                self.open = kept;
                actions
            }
            EventType::Shift => {
                // No tie token: the chunk is malformed. Close everything and
                // discard the rest, including a tie that turns up later.
                self.in_prologue = false;
                self.skip_rest = true;
                self.end_all(self.seek_time)
            }
            EventType::Program => {
                self.program = Some(event.value);
                Vec::new()
            }
            EventType::Pitch => {
                if let Some(program) = self.program {
                    self.tie_set.push(NoteKey::new(program, event.value));
                }
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    fn feed_body(&mut self, event: TokenEvent) -> Vec<NoteAction> {
        match event.r#type {
            // Absolute within the chunk, and 0 is a no-op rather than a rewind.
            EventType::Shift => {
                if event.value > 0 {
                    self.tick_state = self.start_tick + event.value;
                }
                Vec::new()
            }
            EventType::Program => {
                self.program = Some(event.value);
                Vec::new()
            }
            EventType::Velocity => {
                self.velocity = Some(event.value);
                Vec::new()
            }
            EventType::Drum => {
                let time = f64::from(self.tick_state) / f64::from(FRAME_RATE);
                if self.next_seek_time.is_some_and(|next| time >= next) {
                    return Vec::new();
                }
                // Instantaneous, never enters the open set, reads no register.
                vec![NoteAction {
                    kind: ActionKind::DrumHit,
                    program: 0,
                    pitch: event.value,
                    time,
                }]
            }
            EventType::Pitch => {
                let (Some(program), Some(velocity)) = (self.program, self.velocity) else {
                    return Vec::new();
                };

                let time = f64::from(self.tick_state) / f64::from(FRAME_RATE);

                // The model routinely emits events past its own window; they
                // belong to the next chunk, which will decide for itself.
                if self.next_seek_time.is_some_and(|next| time >= next) {
                    return Vec::new();
                }

                let key = NoteKey::new(program, event.value);
                let mut actions = Vec::new();

                // Velocity is an on/off flag, not dynamics. An already-open
                // pitch therefore retriggers: closed here, reopened below, at
                // the same instant.
                if let Some(index) = self.open.iter().position(|(open, _)| *open == key) {
                    self.open.remove(index);
                    actions.push(NoteAction {
                        kind: ActionKind::End,
                        program: key.program,
                        pitch: key.pitch,
                        time,
                    });
                }

                if velocity > 0 {
                    self.open.push_back((key, time));
                    actions.push(NoteAction {
                        kind: ActionKind::Start,
                        program: key.program,
                        pitch: key.pitch,
                        time,
                    });
                }

                actions
            }
            _ => Vec::new(),
        }
    }

    /// End of stream. A stream that ran out mid-prologue never declared
    /// anything, so its open notes end at the boundary rather than getting the
    /// minimum duration.
    pub fn finish(&mut self) -> Vec<NoteAction> {
        if self.chunk_started && self.in_prologue {
            return self.end_all(self.seek_time);
        }

        let actions = self
            .open
            .iter()
            .map(|(key, onset)| NoteAction {
                kind: ActionKind::End,
                program: key.program,
                pitch: key.pitch,
                time: onset + MIN_NOTE_DURATION_SECS,
            })
            .collect();
        self.open.clear();
        actions
    }
}
