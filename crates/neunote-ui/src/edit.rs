//! The editable note list: what is selected, what changed, and how to get back.
//!
//! One [`Editor`] is the single owner of a transcription. It keeps two things:
//! the **raw** notes, which is what every edit and every undo step addresses,
//! and the **shown** notes, which is that list after whatever quantisation the
//! panel is asking for. Keeping them apart is what makes quantisation
//! non-destructive -- dragging the tempo slider re-derives the view instead of
//! pulling already-snapped notes a little further towards the grid every time --
//! and it is why a displayed row has to remember which raw note it came from.
//!
//! Every discrete change goes through [`Editor::replace`], which is what puts it
//! on the undo stack; a drag previews with [`Editor::preview`] and lands with
//! [`Editor::commit`], so holding the mouse down does not fill the history with
//! one entry per frame.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use neunote_types::NoteEvent;
use repose_core::{Signal, signal};

use crate::quantize::Quantize;
use crate::roll;

/// How many past versions are kept. A snapshot is the whole note list, so this
/// is the editor's memory ceiling as much as its history depth.
const HISTORY: usize = 64;

/// The shortest note the roll will create, matching what the model emits.
const MIN_DURATION: f64 = 0.01;

#[derive(Default)]
struct History {
    undo: Vec<Rc<Vec<NoteEvent>>>,
    redo: Vec<Rc<Vec<NoteEvent>>>,
}

#[derive(Clone)]
pub(crate) struct Editor {
    raw: Signal<Rc<Vec<NoteEvent>>>,
    shown: Signal<Rc<Vec<(usize, NoteEvent)>>>,
    selection: Signal<Rc<BTreeSet<usize>>>,
    quantize: Signal<Option<Quantize>>,
    hand: Rc<RefCell<Vec<bool>>>,
    history: Rc<RefCell<History>>,
}

impl Editor {
    pub(crate) fn new(
        raw: Signal<Rc<Vec<NoteEvent>>>,
        selection: Signal<Rc<BTreeSet<usize>>>,
    ) -> Self {
        let editor = Self {
            raw,
            shown: signal(Rc::new(Vec::new())),
            selection,
            quantize: signal(None),
            hand: Rc::new(RefCell::new(Vec::new())),
            history: Rc::new(RefCell::new(History::default())),
        };

        editor.refresh();
        editor
    }

    /// Re-derive what the roll draws from the raw list and the quantisation.
    fn refresh(&self) {
        let raw = self.raw.get();
        self.shown.set(Rc::new(match self.quantize.get() {
            Some(quantize) => quantize.apply_indexed(&roll::indexed(&raw), &self.hand.borrow()),
            None => roll::indexed(&raw),
        }));
    }

    /// The displayed notes, each with the raw note it came from.
    pub(crate) fn shown(&self) -> Rc<Vec<(usize, NoteEvent)>> {
        self.shown.get()
    }

    /// The notes as displayed, which is what the MIDI writer is given.
    pub(crate) fn notes(&self) -> Vec<NoteEvent> {
        self.shown.get().iter().map(|(_, note)| *note).collect()
    }

    pub(crate) fn is_quantised(&self) -> bool {
        self.quantize
            .get()
            .is_some_and(|quantize| !quantize.is_identity())
    }

    /// The grid the roll snaps to, if the panel is asking for one.
    pub(crate) fn grid(&self) -> Option<f64> {
        let quantize = self.quantize.get()?;
        quantize.times.then(|| quantize.step_secs())
    }

    /// Change what quantisation means. Not undoable, and it does not need to
    /// be: the raw list it is derived from never moved.
    pub(crate) fn set_quantize(&self, quantize: Option<Quantize>) {
        if self.quantize.get() == quantize {
            return;
        }

        self.quantize.set(quantize);
        self.refresh();
    }

    pub(crate) fn selection(&self) -> Rc<BTreeSet<usize>> {
        self.selection.get()
    }

    pub(crate) fn select(&self, indices: BTreeSet<usize>) {
        self.selection.set(Rc::new(indices));
    }

    pub(crate) fn toggle(&self, index: usize, on: bool) {
        let mut next = (*self.selection.get()).clone();
        if on {
            next.insert(index);
        } else {
            next.remove(&index);
        }
        self.select(next);
    }

    pub(crate) fn select_only(&self, index: usize) {
        self.select(BTreeSet::from([index]));
    }

    /// Forget the selection: the indices no longer mean anything once notes are
    /// added or removed.
    pub(crate) fn clear_selection(&self) {
        self.select(BTreeSet::new());
    }

    /// The raw list as it stands, for a drag to measure against.
    pub(crate) fn snapshot(&self) -> Rc<Vec<NoteEvent>> {
        self.raw.get()
    }

    /// Show a list without recording it. Used while a drag is still in flight.
    pub(crate) fn preview(&self, notes: Vec<NoteEvent>) {
        self.hand.borrow_mut().resize(notes.len(), false);
        self.raw.set(Rc::new(notes));
        self.refresh();
    }

    /// Record `before` as the state a drag started from, if the drag actually
    /// changed anything. Whatever it moved is the user's from here on:
    /// quantisation leaves it alone.
    pub(crate) fn commit(&self, before: Rc<Vec<NoteEvent>>) {
        let after = self.raw.get();
        if before.as_ref() == after.as_ref() {
            return;
        }

        let mut hand = self.hand.borrow_mut();
        for (index, old) in before.iter().enumerate() {
            if after.get(index) != Some(old)
                && let Some(slot) = hand.get_mut(index)
            {
                *slot = true;
            }
        }
        drop(hand);

        let mut history = self.history.borrow_mut();
        history.undo.push(before);
        if history.undo.len() > HISTORY {
            history.undo.remove(0);
        }
        history.redo.clear();
    }

    /// One discrete change: on the undo stack, and the new list on screen.
    pub(crate) fn replace(&self, notes: Vec<NoteEvent>) {
        let before = self.raw.get();
        if before.as_ref() == notes.as_slice() {
            return;
        }

        self.preview(notes);
        self.commit(before);
    }

    /// A fresh transcription replaces everything, history included: undoing
    /// back into the previous file's notes would be nonsense.
    pub(crate) fn reset(&self, notes: Vec<NoteEvent>) {
        self.clear_history();
        self.preview(notes);
        self.clear_selection();
        self.hand.borrow_mut().clear();
    }

    /// Notes arriving from a run in flight. The run owns these, so they are not
    /// undoable -- the next run replaces them anyway.
    pub(crate) fn stream(&self, notes: Vec<NoteEvent>) {
        self.clear_history();
        self.preview(notes);
        self.clear_selection();
        self.hand.borrow_mut().clear();
    }

    pub(crate) fn undo(&self) -> bool {
        let mut history = self.history.borrow_mut();
        let Some(previous) = history.undo.pop() else {
            return false;
        };

        history.redo.push(self.raw.get());
        drop(history);
        self.preview(previous.as_ref().clone());
        self.clear_selection();
        true
    }

    pub(crate) fn redo(&self) -> bool {
        let mut history = self.history.borrow_mut();
        let Some(next) = history.redo.pop() else {
            return false;
        };

        history.undo.push(self.raw.get());
        drop(history);
        self.preview(next.as_ref().clone());
        self.clear_selection();
        true
    }

    pub(crate) fn can_undo(&self) -> bool {
        !self.history.borrow().undo.is_empty()
    }

    pub(crate) fn can_redo(&self) -> bool {
        !self.history.borrow().redo.is_empty()
    }

    pub(crate) fn delete_selected(&self) -> bool {
        let selection = self.selection.get();
        if selection.is_empty() {
            return false;
        }

        let kept = self
            .raw
            .get()
            .iter()
            .enumerate()
            .filter(|(index, _)| !selection.contains(index))
            .map(|(_, note)| *note)
            .collect();

        let mut hand = self.hand.borrow_mut();
        let mut at = 0;
        hand.retain(|_| {
            let keep = !selection.contains(&at);
            at += 1;
            keep
        });
        drop(hand);

        self.replace(kept);
        self.clear_selection();
        true
    }

    /// Move every selected note by a whole number of grid steps and semitones.
    pub(crate) fn nudge(&self, steps: i32, semitones: i32, grid: Option<f64>) -> bool {
        let selection = self.selection.get();
        if selection.is_empty() {
            return false;
        }

        let shift = match grid {
            Some(step) if step > 0.0 => f64::from(steps) * step,
            _ => f64::from(steps) * 0.01,
        };

        let moved = self
            .raw
            .get()
            .iter()
            .enumerate()
            .map(|(index, note)| {
                if !selection.contains(&index) {
                    return *note;
                }

                let mut note = *note;
                let length = note.offset - note.onset;
                note.onset = (note.onset + shift).max(0.0);
                note.offset = (note.onset + length).max(note.onset + MIN_DURATION);
                note.pitch = (i32::from(note.pitch) + semitones).clamp(0, 127) as u8;
                note
            })
            .collect();

        {
            let mut hand = self.hand.borrow_mut();
            for index in selection.iter() {
                if let Some(slot) = hand.get_mut(*index) {
                    *slot = true;
                }
            }
        }

        self.replace(moved);
        true
    }

    /// Put a note in at a pitch and time, and select it so it can be dragged.
    pub(crate) fn add(&self, pitch: u8, onset: f64, length: f64, program: u16) {
        let mut notes = self.raw.get().as_ref().clone();
        let onset = onset.max(0.0);
        let index = notes.len();
        notes.push(NoteEvent {
            onset,
            offset: onset + length.max(MIN_DURATION),
            pitch: pitch.min(127),
            program,
            is_drum: program == neunote_types::DRUM_PROGRAM,
        });

        self.replace(notes);
        self.select_only(index);
        let mut hand = self.hand.borrow_mut();
        if let Some(slot) = hand.get_mut(index) {
            *slot = true;
        }
    }

    fn clear_history(&self) {
        let mut history = self.history.borrow_mut();
        history.undo.clear();
        history.redo.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repose_core::signal;

    fn note(onset: f64, offset: f64, pitch: u8) -> NoteEvent {
        NoteEvent {
            onset,
            offset,
            pitch,
            program: 0,
            is_drum: false,
        }
    }

    fn editor() -> Editor {
        Editor::new(
            signal(Rc::new(vec![note(0.0, 1.0, 60), note(2.0, 3.0, 62)])),
            signal(Rc::new(BTreeSet::new())),
        )
    }

    fn onsets(editor: &Editor) -> Vec<f64> {
        editor.snapshot().iter().map(|note| note.onset).collect()
    }

    fn shown(editor: &Editor) -> Vec<NoteEvent> {
        editor.notes()
    }

    #[test]
    fn an_undoable_change_comes_back() {
        let editor = editor();
        editor.replace(vec![note(0.0, 1.0, 61)]);

        assert!(editor.can_undo());
        assert!(editor.undo());
        assert_eq!(editor.shown().len(), 2);
        assert_eq!(shown(&editor)[0].pitch, 60);

        assert!(editor.can_redo());
        assert!(editor.redo());
        assert_eq!(shown(&editor)[0].pitch, 61);
    }

    #[test]
    fn undo_stops_at_the_end_instead_of_panicking() {
        let editor = editor();
        assert!(!editor.undo());
        assert!(!editor.redo());
    }

    #[test]
    fn a_drag_is_one_entry_however_many_frames_it_took() {
        let editor = editor();
        let before = editor.snapshot();

        for step in 1..10 {
            let moved = before
                .iter()
                .map(|note| NoteEvent {
                    onset: note.onset + f64::from(step) * 0.1,
                    ..*note
                })
                .collect();
            editor.preview(moved);
        }
        editor.commit(before);

        assert!(editor.undo(), "ten previews, one entry");
        assert_eq!(onsets(&editor), vec![0.0, 2.0]);
    }

    #[test]
    fn a_drag_that_changed_nothing_records_nothing() {
        let editor = editor();
        let before = editor.snapshot();
        editor.preview(before.as_ref().clone());
        editor.commit(before);

        assert!(!editor.can_undo());
    }

    #[test]
    fn a_new_edit_drops_the_redo_stack() {
        let editor = editor();
        editor.replace(vec![]);
        editor.undo();
        assert!(editor.can_redo());

        editor.replace(vec![note(5.0, 6.0, 70)]);
        assert!(!editor.can_redo());
    }

    #[test]
    fn a_new_transcription_forgets_the_old_one() {
        let editor = editor();
        editor.replace(vec![note(9.0, 9.5, 40)]);
        assert!(editor.can_undo());

        editor.reset(vec![note(0.0, 1.0, 30)]);
        assert!(!editor.can_undo(), "undoing into the last file is nonsense");
        assert_eq!(shown(&editor)[0].pitch, 30);
    }

    #[test]
    fn notes_streaming_in_from_a_run_are_not_undoable() {
        let editor = editor();
        editor.replace(vec![note(9.0, 9.5, 40)]);

        editor.stream(vec![note(0.0, 1.0, 30), note(1.0, 2.0, 31)]);
        assert_eq!(editor.shown().len(), 2);
        assert!(!editor.can_undo());
    }

    #[test]
    fn deleting_takes_the_selection_with_it() {
        let editor = editor();
        editor.select_only(0);

        assert!(editor.delete_selected());
        assert_eq!(editor.shown().len(), 1);
        assert_eq!(shown(&editor)[0].pitch, 62);
        assert!(editor.selection().is_empty());
        assert!(!editor.delete_selected(), "nothing left to delete");
    }

    #[test]
    fn nudging_moves_whole_steps_and_keeps_lengths() {
        let editor = editor();
        editor.select_only(1);

        assert!(editor.nudge(1, 1, Some(0.5)));
        let moved = shown(&editor)[1];
        assert!((moved.onset - 2.5).abs() < 1e-9);
        assert!((moved.offset - 3.5).abs() < 1e-9, "length kept");
        assert_eq!(moved.pitch, 63);

        assert!(editor.nudge(-1, -12, Some(0.5)), "and it stacks");
        assert_eq!(shown(&editor)[1].pitch, 51);
    }

    #[test]
    fn nudging_never_leaves_the_keyboard_or_the_start() {
        let editor = editor();
        editor.select(BTreeSet::from([0, 1]));
        editor.nudge(-1, 127, Some(1.0));

        assert_eq!(shown(&editor)[0].pitch, 127);
        assert_eq!(shown(&editor)[1].pitch, 127);
        editor.nudge(-1, -127, None);
        assert_eq!(shown(&editor)[0].pitch, 0);
        assert_eq!(shown(&editor)[0].onset, 0.0, "never before zero");
        assert!(shown(&editor)[0].offset > 0.0);
    }

    #[test]
    fn nudging_without_a_selection_does_nothing() {
        let editor = editor();
        assert!(!editor.nudge(1, 1, Some(0.5)));
        assert!(!editor.can_undo());
    }

    #[test]
    fn an_added_note_is_selected_and_undoable() {
        let editor = editor();
        editor.add(64, 1.25, 0.5, 0);

        assert_eq!(editor.shown().len(), 3);
        assert_eq!(editor.selection().as_ref(), &BTreeSet::from([2]));
        let added = shown(&editor)[2];
        assert!((added.onset - 1.25).abs() < 1e-9);
        assert!((added.offset - 1.75).abs() < 1e-9);

        assert!(editor.undo());
        assert_eq!(editor.shown().len(), 2);
    }

    #[test]
    fn an_added_note_gets_a_length_even_at_the_end_of_a_run() {
        let editor = editor();
        editor.add(64, 1.0, 0.0, 0);
        assert!(shown(&editor)[2].offset > shown(&editor)[2].onset);
    }

    /// One note, deliberately between grid points.
    fn off_grid() -> Editor {
        Editor::new(
            signal(Rc::new(vec![note(0.06, 0.5, 60)])),
            signal(Rc::new(BTreeSet::new())),
        )
    }

    #[test]
    fn quantisation_is_a_view_over_the_raw_list() {
        let editor = off_grid();
        editor.set_quantize(Some(Quantize {
            times: true,
            bpm: 120.0,
            strength: 0.5,
            ..Quantize::default()
        }));

        // Half of the way from 0.06 s to the sixteenth at 0.0.
        assert!(editor.is_quantised());
        assert!((shown(&editor)[0].onset - 0.03).abs() < 1e-9);
        assert!(
            (shown(&editor)[0].offset - 0.47).abs() < 1e-9,
            "length kept"
        );
        assert_eq!(
            editor.snapshot()[0].onset,
            0.06,
            "the raw list did not move"
        );

        editor.set_quantize(None);
        assert!(!editor.is_quantised());
        assert!((shown(&editor)[0].onset - 0.06).abs() < 1e-9);
        assert!(!editor.can_undo(), "changing a knob is not an edit");
    }

    #[test]
    fn a_partial_quantisation_does_not_drift_when_a_knob_moves() {
        let editor = off_grid();
        editor.set_quantize(Some(Quantize {
            times: true,
            bpm: 120.0,
            strength: 0.5,
            ..Quantize::default()
        }));
        let at_120 = shown(&editor)[0].onset;
        assert!((at_120 - 0.03).abs() < 1e-9);

        // At 60 bpm the grid is 0.25 s, so the nearest point is 0.0 again.
        editor.set_quantize(Some(Quantize {
            times: true,
            bpm: 60.0,
            strength: 0.5,
            ..Quantize::default()
        }));
        assert!((shown(&editor)[0].onset - 0.03).abs() < 1e-9);

        // Strength back to full: the view is derived from the raw onset every
        // time, so the note cannot creep towards the grid as the knobs move.
        editor.set_quantize(Some(Quantize {
            times: true,
            bpm: 120.0,
            strength: 1.0,
            ..Quantize::default()
        }));
        assert_eq!(shown(&editor)[0].onset, 0.0);
        editor.set_quantize(Some(Quantize {
            times: true,
            bpm: 120.0,
            strength: 0.5,
            ..Quantize::default()
        }));
        assert!((shown(&editor)[0].onset - 0.03).abs() < 1e-9, "not 0.015");
    }

    #[test]
    fn dropping_an_out_of_scale_note_leaves_a_gap_in_the_raw_indices() {
        let editor = Editor::new(
            signal(Rc::new(vec![
                note(0.0, 1.0, 60),
                note(1.0, 2.0, 61), // not in C major
                note(2.0, 3.0, 62),
            ])),
            signal(Rc::new(BTreeSet::new())),
        );

        editor.set_quantize(Some(Quantize {
            pitches: true,
            scale: crate::quantize::Scale::Major,
            snap: crate::quantize::Snap::Remove,
            ..Quantize::default()
        }));

        let rows = editor.shown();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].0, 2, "the dropped note is not renumbered");

        // Selecting and editing what is on screen edits the right raw note.
        editor.select_only(2);
        assert!(editor.nudge(0, 1, None));
        assert_eq!(editor.snapshot()[2].pitch, 63);
        assert_eq!(
            editor.snapshot()[1].pitch,
            61,
            "the dropped note is untouched"
        );
    }

    #[test]
    fn a_note_the_user_moved_is_not_quantised_over() {
        let editor = off_grid();
        editor.select_only(0);

        assert!(editor.nudge(1, 0, None));
        let by_hand = shown(&editor)[0].onset;

        editor.set_quantize(Some(Quantize {
            times: true,
            bpm: 120.0,
            division: crate::quantize::Division::Sixteenth,
            strength: 1.0,
            ..Quantize::default()
        }));

        assert_eq!(
            shown(&editor)[0].onset,
            by_hand,
            "the note stays where it was put"
        );
        assert_eq!(editor.snapshot()[0].onset, by_hand);
    }

    #[test]
    fn a_grid_only_exists_when_the_panel_asks_for_one() {
        let editor = editor();
        assert_eq!(editor.grid(), None);

        editor.set_quantize(Some(Quantize {
            times: true,
            bpm: 120.0,
            ..Quantize::default()
        }));
        assert!((editor.grid().expect("a grid") - 0.125).abs() < 1e-12);

        editor.set_quantize(Some(Quantize::default()));
        assert_eq!(editor.grid(), None);
    }

    #[test]
    fn the_notes_the_writer_gets_are_the_ones_on_screen() {
        let editor = editor();
        editor.set_quantize(Some(Quantize {
            pitches: true,
            scale: crate::quantize::Scale::Major,
            snap: crate::quantize::Snap::Nearest,
            ..Quantize::default()
        }));

        let written = editor.notes();
        assert_eq!(written.len(), editor.shown().len());
        assert_eq!(written[0].pitch, shown(&editor)[0].pitch);
    }

    #[test]
    fn selection_toggles() {
        let editor = editor();
        editor.select_only(0);
        editor.toggle(1, true);
        assert_eq!(editor.selection().as_ref(), &BTreeSet::from([0, 1]));
        editor.toggle(0, false);
        assert_eq!(editor.selection().as_ref(), &BTreeSet::from([1]));
    }

    #[test]
    fn the_history_stops_growing_past_its_ceiling() {
        let editor = editor();
        for step in 1..200 {
            editor.replace(vec![note(0.0, 1.0, (step % 100) as u8)]);
        }

        let mut undone = 0;
        while editor.undo() {
            undone += 1;
        }
        assert_eq!(undone, HISTORY);
    }
}
