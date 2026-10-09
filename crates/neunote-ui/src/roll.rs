//! Piano-roll geometry: pixels to musical time, and the arithmetic behind every
//! gesture on the roll.
//!
//! Pure, so the roll can be exercised without a window. The view owns the
//! [`Viewport`] and asks these questions; nothing here touches a signal, a
//! theme or a pointer.

use std::collections::{BTreeSet, HashSet};

use neunote_types::NoteEvent;
use repose_core::{Rect, Vec2};

/// Width of the pitch-name column down the left edge.
pub const KEYBED: f32 = 56.0;

/// How close to a note's right edge a press counts as grabbing it.
pub const EDGE: f32 = 6.0;

pub const MIN_PX_PER_SEC: f32 = 4.0;
pub const MAX_PX_PER_SEC: f32 = 2000.0;
pub const MIN_PX_PER_PITCH: f32 = 5.0;
pub const MAX_PX_PER_PITCH: f32 = 48.0;

/// What part of the score the roll is showing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub px_per_sec: f32,
    pub px_per_pitch: f32,
    /// Seconds at the left edge, before the key column.
    pub left_secs: f64,
    /// Pitch at the bottom edge.
    pub bottom_pitch: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            px_per_sec: 60.0,
            px_per_pitch: 12.0,
            left_secs: 0.0,
            bottom_pitch: 40.0,
        }
    }
}

impl Viewport {
    /// The whole transcription in view: every note inside the canvas, with a
    /// little room around the edges.
    pub fn fit(notes: &[NoteEvent], width: f32, height: f32) -> Self {
        let usable = (width - KEYBED).max(64.0);
        let Some(first) = notes.first() else {
            // Nothing to frame: an octave and a half around middle C.
            return Self {
                px_per_sec: usable / 10.0,
                px_per_pitch: 12.0,
                left_secs: 0.0,
                bottom_pitch: 48.0,
            }
            .clamped();
        };

        let mut lowest = f32::from(first.pitch);
        let mut highest = lowest;
        let mut end = first.offset.max(1.0);
        for note in notes {
            lowest = lowest.min(f32::from(note.pitch));
            highest = highest.max(f32::from(note.pitch));
            end = end.max(note.offset);
        }

        Self {
            px_per_sec: (usable as f64 / end) as f32,
            px_per_pitch: height.max(64.0) / (highest - lowest + 3.0).max(6.0),
            left_secs: 0.0,
            bottom_pitch: lowest - 1.0,
        }
        .clamped()
    }

    pub fn x_of(&self, onset: f64) -> f32 {
        KEYBED + ((onset - self.left_secs) as f32 * self.px_per_sec)
    }

    pub fn secs_at(&self, x: f32) -> f64 {
        self.left_secs + f64::from((x - KEYBED) / self.px_per_sec)
    }

    /// Higher pitches are drawn higher, so y counts down from the top.
    pub fn y_of(&self, pitch: f32, height: f32) -> f32 {
        height - (pitch - self.bottom_pitch) * self.px_per_pitch
    }

    pub fn pitch_at(&self, y: f32, height: f32) -> f32 {
        self.bottom_pitch + (height - y) / self.px_per_pitch
    }

    pub fn rect_of(&self, note: &NoteEvent, height: f32) -> Rect {
        Rect {
            x: self.x_of(note.onset),
            y: self.y_of(f32::from(note.pitch) + 1.0, height) + 1.0,
            w: ((note.offset - note.onset) as f32 * self.px_per_sec).max(2.0),
            h: (self.px_per_pitch - 2.0).max(2.0),
        }
    }

    /// Drag the content by a pixel delta.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.left_secs -= f64::from(dx / self.px_per_sec);
        self.bottom_pitch += dy / self.px_per_pitch;
        *self = self.clamped();
    }

    /// Zoom in time, keeping whatever is under `anchor` where it is.
    pub fn zoom_time(&mut self, factor: f32, anchor: f32) {
        let held = self.secs_at(anchor);
        self.px_per_sec = (self.px_per_sec * factor).clamp(MIN_PX_PER_SEC, MAX_PX_PER_SEC);
        self.left_secs = held - f64::from((anchor - KEYBED) / self.px_per_sec);
        *self = self.clamped();
    }

    /// Never scroll past the ends: time starts at zero, pitch stays on the
    /// keyboard. The view can still be scrolled past the notes themselves --
    /// there is no frame to stop it -- so `fit` is the way back.
    pub fn clamped(&self) -> Self {
        let mut out = *self;
        out.left_secs = out.left_secs.max(0.0);
        out.bottom_pitch = out.bottom_pitch.clamp(-2.0, 128.0);
        out
    }
}

/// A note the roll can show: where it sits in the *raw* transcription, and the
/// note as it is displayed. The two differ whenever quantisation is on, and a
/// selection has to mean the raw one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    pub index: usize,
    pub note: NoteEvent,
}

pub fn visible(notes: &[(usize, NoteEvent)], hidden: &HashSet<u16>) -> Vec<Row> {
    notes
        .iter()
        .filter(|(_, note)| !hidden.contains(&note.program))
        .map(|(index, note)| Row {
            index: *index,
            note: *note,
        })
        .collect()
}

/// Pair every note with its place in the raw list, which is what a selection and
/// every edit address.
pub fn indexed(notes: &[NoteEvent]) -> Vec<(usize, NoteEvent)> {
    notes.iter().copied().enumerate().collect()
}

/// The note under a point. Later notes are drawn over earlier ones, so the
/// search runs backwards.
pub fn hit(rows: &[Row], view: &Viewport, height: f32, at: Vec2) -> Option<usize> {
    if at.x < KEYBED {
        return None;
    }

    rows.iter().rev().find_map(|row| {
        let rect = view.rect_of(&row.note, height);
        if at.x >= rect.x && at.x <= rect.x + rect.w && at.y >= rect.y && at.y <= rect.y + rect.h {
            Some(row.index)
        } else {
            None
        }
    })
}

/// Whether a press landed on the right edge of a note, which is the resize
/// handle. Vertical touches do not resize: on a phone they would fight the
/// pitch drag.
pub fn on_right_edge(rows: &[Row], view: &Viewport, height: f32, at: Vec2, touch: bool) -> bool {
    if touch || at.x < KEYBED {
        return false;
    }

    rows.iter().rev().any(|row| {
        let rect = view.rect_of(&row.note, height);
        at.x > rect.x + rect.w - EDGE
            && at.x <= rect.x + rect.w + EDGE / 2.0
            && at.y >= rect.y
            && at.y <= rect.y + rect.h
    })
}

/// Every note a marquee touches, in transcription order.
pub fn box_select(
    rows: &[Row],
    view: &Viewport,
    height: f32,
    from: Vec2,
    to: Vec2,
) -> BTreeSet<usize> {
    let left = from.x.min(to.x).max(KEYBED);
    let right = from.x.max(to.x);
    let top = from.y.min(to.y);
    let bottom = from.y.max(to.y);

    rows.iter()
        .filter_map(|row| {
            let rect = view.rect_of(&row.note, height);
            let overlaps = rect.x <= right
                && rect.x + rect.w >= left
                && rect.y <= bottom
                && rect.y + rect.h >= top;
            overlaps.then_some(row.index)
        })
        .collect()
}

/// How far a drag from `from` to `to` moves a note, in seconds and semitones.
///
/// Measured from the anchor rather than accumulated per event, so the same
/// pointer position always gives the same result and a drag cannot drift. With
/// a grid the time part lands on multiples of it; the pitch part is always whole
/// semitones.
pub fn drag_delta(
    view: &Viewport,
    height: f32,
    from: Vec2,
    to: Vec2,
    grid: Option<f64>,
) -> (f64, i32) {
    let raw = view.secs_at(to.x) - view.secs_at(from.x);
    let secs = match grid {
        Some(step) if step > 0.0 => (raw / step).round() * step,
        _ => raw,
    };

    let semitones = (view.pitch_at(to.y, height) - view.pitch_at(from.y, height)).round() as i32;

    (secs, semitones)
}

/// The grid point at or before a time.
pub fn snap_time(time: f64, grid: Option<f64>) -> f64 {
    match grid {
        Some(step) if step > 0.0 => (time / step).floor() * step,
        _ => time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(onset: f64, offset: f64, pitch: u8) -> NoteEvent {
        NoteEvent {
            onset,
            offset,
            pitch,
            program: 0,
            is_drum: false,
        }
    }

    fn rows(notes: &[NoteEvent]) -> Vec<Row> {
        visible(&indexed(notes), &HashSet::new())
    }

    #[test]
    fn a_time_maps_to_a_pixel_and_back() {
        let view = Viewport {
            px_per_sec: 100.0,
            left_secs: 2.0,
            ..Viewport::default()
        };

        let x = view.x_of(5.0);
        assert!((x - (KEYBED + 300.0)).abs() < 1e-4);
        assert!((view.secs_at(x) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn the_bottom_edge_is_the_bottom_pitch_and_time_rises_upward() {
        let view = Viewport {
            px_per_pitch: 10.0,
            bottom_pitch: 60.0,
            ..Viewport::default()
        };
        let height = 400.0;

        assert!((view.y_of(60.0, height) - height).abs() < 1e-4);
        assert!((view.y_of(70.0, height) - (height - 100.0)).abs() < 1e-4);
        assert!((view.pitch_at(view.y_of(64.0, height), height) - 64.0).abs() < 1e-4);
    }

    #[test]
    fn fitting_a_transcription_shows_all_of_it() {
        let notes = vec![note(0.0, 1.0, 60), note(9.0, 10.0, 72)];
        let view = Viewport::fit(&notes, 900.0, 600.0);

        assert!(view.x_of(10.0) <= 900.0, "the last note is on screen");
        assert!(view.y_of(73.0, 600.0) >= 0.0, "the top note is on screen");
        assert!(
            view.y_of(59.0, 600.0) <= 600.0,
            "the bottom note is on screen"
        );
    }

    #[test]
    fn fitting_nothing_leaves_a_usable_viewport() {
        let view = Viewport::fit(&[], 900.0, 600.0);
        assert!((MIN_PX_PER_SEC..=MAX_PX_PER_SEC).contains(&view.px_per_sec));
        assert!((MIN_PX_PER_PITCH..=MAX_PX_PER_PITCH).contains(&view.px_per_pitch));
        assert_eq!(view.left_secs, 0.0);
    }

    #[test]
    fn panning_drags_the_content_with_the_pointer() {
        let mut view = Viewport {
            px_per_sec: 100.0,
            px_per_pitch: 10.0,
            left_secs: 5.0,
            bottom_pitch: 60.0,
        };

        view.pan(100.0, 50.0);
        assert!(
            (view.left_secs - 4.0).abs() < 1e-6,
            "drag right, see earlier"
        );
        assert!(
            (view.bottom_pitch - 65.0).abs() < 1e-4,
            "drag down, see higher"
        );

        // Back to where it was.
        view.pan(-100.0, -50.0);
        assert!((view.left_secs - 5.0).abs() < 1e-6);
        assert!((view.bottom_pitch - 60.0).abs() < 1e-4);
    }

    #[test]
    fn panning_never_goes_before_the_start_or_off_the_keyboard() {
        let mut view = Viewport {
            left_secs: 1.0,
            bottom_pitch: 60.0,
            px_per_pitch: 10.0,
            ..Viewport::default()
        };
        view.pan(10_000.0, -10_000.0);
        assert_eq!(view.left_secs, 0.0, "time starts at zero");
        assert!(view.bottom_pitch >= -2.0);

        view.pan(-10_000.0, 10_000.0);
        assert_eq!(view.bottom_pitch, 128.0, "the bottom stays on the keyboard");
    }

    #[test]
    fn zooming_keeps_the_time_under_the_cursor() {
        let mut view = Viewport {
            px_per_sec: 100.0,
            left_secs: 0.0,
            ..Viewport::default()
        };
        let anchor = KEYBED + 300.0;
        let held = view.secs_at(anchor);

        view.zoom_time(2.0, anchor);
        assert!((view.px_per_sec - 200.0).abs() < 1e-4);
        assert!(
            (view.secs_at(anchor) - held).abs() < 1e-4,
            "the cursor keeps its moment"
        );
    }

    #[test]
    fn zoom_stops_at_the_limits() {
        let mut view = Viewport::default();
        for _ in 0..64 {
            view.zoom_time(2.0, KEYBED + 10.0);
        }
        assert_eq!(view.px_per_sec, MAX_PX_PER_SEC);
        for _ in 0..64 {
            view.zoom_time(0.5, KEYBED + 10.0);
        }
        assert_eq!(view.px_per_sec, MIN_PX_PER_SEC);
    }

    #[test]
    fn rows_keep_the_place_of_their_note_in_the_raw_list() {
        // With quantisation on, the displayed list is not the raw one: a row has
        // to say which raw note it came from, or a selection would edit the
        // wrong one.
        let quantised = vec![(0usize, note(0.0, 1.0, 60)), (2usize, note(2.0, 3.0, 62))];

        let shown = visible(&quantised, &HashSet::new());
        assert_eq!(shown.len(), 2);
        assert_eq!(
            shown[1].index, 2,
            "a dropped note leaves a gap, not a renumbering"
        );
        assert_eq!(indexed(&[note(0.0, 1.0, 60)]).len(), 1);
    }

    #[test]
    fn a_press_finds_the_note_under_it_and_the_key_column_finds_none() {
        let notes = vec![note(0.0, 1.0, 60), note(2.0, 3.0, 62)];
        let rows = rows(&notes);
        let view = Viewport::fit(&notes, 900.0, 600.0);

        let rect = view.rect_of(&notes[1], 600.0);
        let inside = Vec2 {
            x: rect.x + 4.0,
            y: rect.y + rect.h / 2.0,
        };
        assert_eq!(hit(&rows, &view, 600.0, inside), Some(1));
        assert_eq!(hit(&rows, &view, 600.0, Vec2 { x: 10.0, y: 300.0 }), None);

        let above = Vec2 {
            x: inside.x,
            y: inside.y - 200.0,
        };
        assert_eq!(
            hit(&rows, &view, 600.0, above),
            None,
            "empty sky selects nothing"
        );
    }

    #[test]
    fn the_topmost_note_wins_where_two_overlap() {
        let notes = vec![note(0.0, 4.0, 60), note(1.0, 3.0, 60)];
        let rows = rows(&notes);
        let view = Viewport::fit(&notes, 900.0, 600.0);
        let rect = view.rect_of(&notes[1], 600.0);

        let at = Vec2 {
            x: rect.x + 4.0,
            y: rect.y + 1.0,
        };
        assert_eq!(hit(&rows, &view, 600.0, at), Some(1));
    }

    #[test]
    fn only_the_right_edge_is_the_resize_handle() {
        let notes = vec![note(0.0, 1.0, 60)];
        let rows = rows(&notes);
        let view = Viewport::fit(&notes, 900.0, 600.0);
        let rect = view.rect_of(&notes[0], 600.0);
        let middle_y = rect.y + rect.h / 2.0;

        assert!(on_right_edge(
            &rows,
            &view,
            600.0,
            Vec2 {
                x: rect.x + rect.w - 1.0,
                y: middle_y
            },
            false
        ));
        assert!(!on_right_edge(
            &rows,
            &view,
            600.0,
            Vec2 {
                x: rect.x + 2.0,
                y: middle_y
            },
            false
        ));
        assert!(!on_right_edge(
            &rows,
            &view,
            600.0,
            Vec2 {
                x: rect.x + rect.w - 1.0,
                y: middle_y
            },
            true
        ));
    }

    #[test]
    fn a_marquee_dragging_backwards_still_selects() {
        let notes = vec![note(0.0, 1.0, 60), note(0.5, 1.5, 60), note(9.0, 10.0, 60)];
        let rows = rows(&notes);
        let view = Viewport::fit(&notes, 900.0, 600.0);

        let top_left = view.rect_of(&notes[0], 600.0);
        let from = Vec2 {
            x: top_left.x - 5.0,
            y: top_left.y - 5.0,
        };
        let to = Vec2 {
            x: view.rect_of(&notes[1], 600.0).x + 10.0,
            y: top_left.y + 30.0,
        };

        let picked = box_select(&rows, &view, 600.0, from, to);
        assert_eq!(picked, BTreeSet::from([0, 1]));
    }

    #[test]
    fn a_drag_measured_from_its_anchor_never_drifts() {
        let view = Viewport {
            px_per_sec: 100.0,
            px_per_pitch: 10.0,
            bottom_pitch: 60.0,
            ..Viewport::default()
        };
        let from = Vec2 {
            x: KEYBED,
            y: 300.0,
        };
        let to = Vec2 {
            x: KEYBED + 250.0,
            y: 260.0,
        };

        let once = drag_delta(&view, 400.0, from, to, None);
        let twice = drag_delta(&view, 400.0, from, to, None);
        assert_eq!(once, twice);
        assert!((once.0 - 2.5).abs() < 1e-6);
        assert_eq!(once.1, 4);
    }

    #[test]
    fn a_grid_makes_dragged_times_land_on_it() {
        let view = Viewport {
            px_per_sec: 100.0,
            ..Viewport::default()
        };
        let from = Vec2 {
            x: KEYBED,
            y: 300.0,
        };

        // 0.62 s of travel on a 0.125 s grid is five sixteenths.
        let (secs, _) = drag_delta(
            &view,
            400.0,
            from,
            Vec2 {
                x: KEYBED + 62.0,
                y: 300.0,
            },
            Some(0.125),
        );
        assert!((secs - 0.625).abs() < 1e-9);

        // Without a grid the same drag keeps the 0.62.
        let (free, _) = drag_delta(
            &view,
            400.0,
            from,
            Vec2 {
                x: KEYBED + 62.0,
                y: 300.0,
            },
            None,
        );
        assert!((free - 0.62).abs() < 1e-6);
    }

    #[test]
    fn a_pitch_drag_is_always_whole_semitones() {
        let view = Viewport {
            px_per_pitch: 10.0,
            bottom_pitch: 60.0,
            ..Viewport::default()
        };
        let from = Vec2 {
            x: KEYBED + 10.0,
            y: 300.0,
        };

        // 4.4 rows up.
        let (_, semitones) = drag_delta(
            &view,
            400.0,
            from,
            Vec2 {
                x: KEYBED + 10.0,
                y: 256.0,
            },
            None,
        );
        assert_eq!(semitones, 4);
    }

    #[test]
    fn snapping_a_time_to_the_grid_floors_it() {
        assert_eq!(snap_time(0.62, Some(0.125)), 0.5);
        assert_eq!(snap_time(0.62, None), 0.62);
        assert_eq!(snap_time(0.62, Some(0.0)), 0.62, "a zero grid is no grid");
    }

    #[test]
    fn hidden_tracks_are_not_hit_or_marqueed() {
        let mut notes = vec![note(0.0, 1.0, 60)];
        notes.push(NoteEvent {
            onset: 5.0,
            offset: 6.0,
            pitch: 84,
            program: 40,
            is_drum: false,
        });

        let hidden = HashSet::from([40u16]);
        let shown = visible(&indexed(&notes), &hidden);
        assert_eq!(shown.len(), 1);
        assert_eq!(shown[0].index, 0);

        let view = Viewport::fit(&notes, 900.0, 600.0);
        let hidden_rect = view.rect_of(&notes[1], 600.0);
        let at_here = Vec2 {
            x: hidden_rect.x + hidden_rect.w / 2.0,
            y: hidden_rect.y + hidden_rect.h / 2.0,
        };

        assert_eq!(
            hit(&shown, &view, 600.0, at_here),
            None,
            "a hidden track cannot be grabbed"
        );
        assert!(
            box_select(&shown, &view, 600.0, at_here, at_here).is_empty(),
            "nor marqueed"
        );
    }
}
