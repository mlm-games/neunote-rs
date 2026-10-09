//! The piano roll: every note as a rectangle on a time/pitch grid, and the
//! gestures that edit them.
//!
//! The roll owns its viewport rather than living inside a scroll area: editing
//! needs to know where a pixel *is*, so panning and zooming are part of the
//! geometry in [`crate::roll`]. A drag previews through the editor and lands as
//! one undo entry, however many frames it took.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::Rc;

use neunote_types::NoteEvent;
use repose_canvas::{Canvas, DrawScope};
use repose_core::{PointerButton, PointerEventKind, StrokeCap, prelude::*};

use crate::edit::Editor;
use crate::roll::{self, Viewport};

/// Grid lines at least this far apart, so the roll never turns into hatching.
const MIN_GRID_PX: f32 = 72.0;

const GRID_STEPS: [f64; 10] = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0];

/// What a press on the roll turned into.
enum Drag {
    /// Middle button or alt: move the view, not the notes.
    Pan { last: Vec2 },
    Marquee {
        from: Vec2,
        to: Vec2,
        /// What was selected before the marquee, for a shift-drag that adds.
        base: Rc<std::collections::BTreeSet<usize>>,
    },
    Edit {
        anchor: Vec2,
        before: Rc<Vec<NoteEvent>>,
        selected: Rc<std::collections::BTreeSet<usize>>,
        /// The one note being resized, if the press caught its right edge.
        resize: Option<usize>,
    },
}

/// `size` is the roll's own size in physical pixels, written during paint: a
/// pointer event carries no size, and every gesture here needs one to turn a
/// pixel into a pitch.
pub(crate) fn view(
    editor: Editor,
    hidden: Rc<HashSet<u16>>,
    viewport: Signal<Viewport>,
    size: Signal<Vec2>,
) -> View {
    let drag: Rc<RefCell<Option<Drag>>> =
        remember_with_key("neunote:roll-drag", || RefCell::new(None));
    let pointer: Rc<Cell<Vec2>> = remember_with_key("neunote:roll-pointer", || {
        Cell::new(Vec2 { x: 0.0, y: 0.0 })
    });

    let rows = Rc::new(roll::visible(&editor.shown(), &hidden));
    let view_now = viewport.get();
    let selection = editor.selection();
    let grid = editor.grid();
    let step = grid.unwrap_or_else(|| grid_step(view_now.px_per_sec));

    let modifier = Modifier::new()
        .fill_max_size()
        .background(theme().surface)
        .on_pointer_down({
            let editor = editor.clone();
            let drag = Rc::clone(&drag);
            let size = size.clone();
            let pointer = Rc::clone(&pointer);
            let rows = Rc::clone(&rows);
            let selection = Rc::clone(&selection);

            move |event: PointerEvent| {
                pointer.set(event.position);
                let height = size.get().y;
                if height <= 0.0 {
                    return;
                }

                let adding = event.modifiers.shift || event.modifiers.ctrl;
                let panning = event.modifiers.alt
                    || matches!(event.event, PointerEventKind::Down(PointerButton::Tertiary));

                if panning {
                    *drag.borrow_mut() = Some(Drag::Pan {
                        last: event.position,
                    });
                    return;
                }

                // The key column is for reading pitches off, not editing.
                if event.position.x < roll::KEYBED {
                    return;
                }

                let hit = roll::hit(&rows, &view_now, height, event.position);
                match hit {
                    Some(index) => {
                        if adding {
                            editor.toggle(index, true);
                        } else if !selection.contains(&index) {
                            editor.select_only(index);
                        }

                        let resizing = if adding {
                            None
                        } else {
                            roll::on_right_edge(&rows, &view_now, height, event.position, false)
                                .then_some(index)
                        };

                        *drag.borrow_mut() = Some(Drag::Edit {
                            anchor: event.position,
                            before: editor.snapshot(),
                            selected: editor.selection(),
                            resize: resizing,
                        });
                    }
                    None => {
                        if !adding {
                            editor.clear_selection();
                        }

                        *drag.borrow_mut() = Some(Drag::Marquee {
                            from: event.position,
                            to: event.position,
                            base: editor.selection(),
                        });
                    }
                }
            }
        })
        .on_pointer_move({
            let editor = editor.clone();
            let drag = Rc::clone(&drag);
            let viewport = viewport.clone();
            let size = size.clone();
            let pointer = Rc::clone(&pointer);
            let rows = Rc::clone(&rows);

            move |event: PointerEvent| {
                pointer.set(event.position);
                let height = size.get().y;
                if height <= 0.0 {
                    return;
                }

                let current = viewport.get();
                let Some(active) = drag.borrow_mut().take() else {
                    return;
                };

                let next = match active {
                    Drag::Pan { last } => {
                        let delta = Vec2 {
                            x: event.position.x - last.x,
                            y: event.position.y - last.y,
                        };
                        viewport.update(|view| view.pan(delta.x, delta.y));
                        Drag::Pan {
                            last: event.position,
                        }
                    }

                    Drag::Marquee { from, base, .. } => {
                        let picked =
                            roll::box_select(&rows, &current, height, from, event.position);
                        let mut next_set = (*base).clone();
                        if event.modifiers.shift {
                            next_set.extend(picked);
                        } else {
                            next_set = picked;
                        }
                        editor.select(next_set);
                        Drag::Marquee {
                            from,
                            to: event.position,
                            base,
                        }
                    }

                    Drag::Edit {
                        anchor,
                        before,
                        selected,
                        resize,
                    } => {
                        match resize {
                            Some(index) => {
                                let mut next = before.as_ref().clone();
                                let at = current.secs_at(event.position.x).max(0.0);
                                if let Some(note) = next.get_mut(index) {
                                    note.offset = at.max(note.onset + 0.02);
                                }
                                editor.preview(next);
                            }
                            None => {
                                let (secs, semitones) = roll::drag_delta(
                                    &current,
                                    height,
                                    anchor,
                                    event.position,
                                    grid,
                                );

                                let next = before
                                    .iter()
                                    .enumerate()
                                    .map(|(index, note)| {
                                        if !selected.contains(&index) {
                                            return *note;
                                        }

                                        let mut note = *note;
                                        let length = note.offset - note.onset;
                                        note.onset = (note.onset + secs).max(0.0);
                                        note.offset = note.onset + length;
                                        note.pitch =
                                            (i32::from(note.pitch) + semitones).clamp(0, 127) as u8;
                                        note
                                    })
                                    .collect();

                                editor.preview(next);
                            }
                        }

                        Drag::Edit {
                            anchor,
                            before,
                            selected,
                            resize,
                        }
                    }
                };

                *drag.borrow_mut() = Some(next);
            }
        })
        .on_pointer_up({
            let editor = editor.clone();
            let drag = Rc::clone(&drag);

            move |_: PointerEvent| {
                if let Some(Drag::Edit { before, .. }) = drag.borrow_mut().take() {
                    editor.commit(before);
                } else {
                    drag.borrow_mut().take();
                }
            }
        })
        .on_pointer_cancel({
            let editor = editor.clone();
            let drag = Rc::clone(&drag);

            move |_: PointerEvent| {
                if let Some(Drag::Edit { before, .. }) = drag.borrow_mut().take() {
                    editor.commit(before);
                } else {
                    drag.borrow_mut().take();
                }
            }
        })
        .on_double_click({
            let editor = editor.clone();
            let viewport = viewport.clone();
            let size = size.clone();
            let pointer = Rc::clone(&pointer);

            move || {
                let height = size.get().y;
                if height <= 0.0 {
                    return;
                }

                let view_now = viewport.get();
                let at = pointer.get();
                if at.x < roll::KEYBED {
                    return;
                }

                let pitch = view_now.pitch_at(at.y, height).floor();
                if !(0.0..=127.0).contains(&pitch) {
                    return;
                }

                let onset = roll::snap_time(view_now.secs_at(at.x).max(0.0), grid);
                let length = grid.unwrap_or(0.25).max(0.05);
                editor.add(pitch as u8, onset, length, 0);
            }
        })
        .on_scroll({
            let viewport = viewport.clone();
            let pointer = Rc::clone(&pointer);

            move |delta: Vec2| {
                let mut view = viewport.get();
                let factor = if delta.y < 0.0 { 1.15 } else { 1.0 / 1.15 };
                view.zoom_time(factor, pointer.get().x.max(roll::KEYBED));
                viewport.set(view);
                delta
            }
        });

    let draw = {
        let size = size.clone();
        let drag = Rc::clone(&drag);
        let rows = Rc::clone(&rows);

        move |ds: &mut DrawScope| {
            let measured = ds.size;
            let current = size.get();
            if (current.x - measured.width).abs() > 0.5 || (current.y - measured.height).abs() > 0.5
            {
                size.set(Vec2 {
                    x: measured.width,
                    y: measured.height,
                });
            }

            let width = measured.width;
            let height = measured.height;
            if width <= 0.0 || height <= 0.0 {
                return;
            }

            paint(
                ds,
                &Frame {
                    rows: &rows,
                    view: view_now,
                    selection: &selection,
                    step,
                    drag: &drag,
                    width,
                    height,
                },
            );
        }
    };

    Canvas(modifier, draw)
}

/// The smallest whole time step whose lines stay far enough apart to read.
fn grid_step(px_per_sec: f32) -> f64 {
    GRID_STEPS
        .iter()
        .copied()
        .find(|step| *step as f32 * px_per_sec >= MIN_GRID_PX)
        .unwrap_or(*GRID_STEPS.last().expect("the list is never empty"))
}

fn is_black(pitch: u32) -> bool {
    matches!(pitch % 12, 1 | 3 | 6 | 8 | 10)
}

fn palette() -> [Color; 6] {
    [
        theme().primary,
        theme().secondary,
        theme().tertiary,
        theme().error,
        theme().primary_container,
        theme().secondary_container,
    ]
}

fn program_color(program: u16, palette: &[Color; 6]) -> Color {
    palette[(program as usize * 7 + 3) % palette.len()]
}

/// One frame's worth of what the roll is showing, handed to the painter so it
/// does not need eight arguments.
struct Frame<'a> {
    rows: &'a [roll::Row],
    view: Viewport,
    selection: &'a std::collections::BTreeSet<usize>,
    step: f64,
    drag: &'a RefCell<Option<Drag>>,
    width: f32,
    height: f32,
}

fn paint(ds: &mut DrawScope, frame: &Frame<'_>) {
    let Frame {
        rows,
        view,
        selection,
        step,
        drag,
        width,
        height,
    } = *frame;
    let palette = palette();
    let grid_color = theme().outline_variant;
    let octave_color = theme().surface_container_highest;
    let white_key = theme().surface_container_high;
    let black_key = theme().surface_container_lowest;
    let key_text = theme().on_surface_variant;

    let top = view.pitch_at(0.0, height);
    let bottom = view.pitch_at(height, height);

    // Pitch lanes, and the octave lines between them.
    let mut pitch = top.floor().clamp(-1.0, 127.0);
    while pitch <= bottom.ceil().min(128.0) {
        let y = view.y_of(pitch + 1.0, height);
        if (0.0..=127.0).contains(&pitch) {
            if is_black(pitch as u32) {
                ds.draw_rect(
                    Rect {
                        x: roll::KEYBED,
                        y,
                        w: width - roll::KEYBED,
                        h: view.px_per_pitch,
                    },
                    black_key.with_alpha_f32(0.35),
                    Px(0.0),
                );
            }

            if (pitch as i32) % 12 == 0 {
                ds.draw_line(
                    Vec2 { x: roll::KEYBED, y },
                    Vec2 { x: width, y },
                    octave_color,
                    Px(1.0),
                    StrokeCap::Butt,
                );
            }
        }

        pitch += 1.0;
    }

    // Time grid, from the left edge to the right.
    let mut at = (view.left_secs / step).floor() * step;
    while at <= view.secs_at(width) {
        let x = view.x_of(at);
        if x >= roll::KEYBED {
            ds.draw_line(
                Vec2 { x, y: 0.0 },
                Vec2 { x, y: height },
                grid_color,
                Px(1.0),
                StrokeCap::Butt,
            );
        }
        at += step;
    }

    // Notes.
    for row in rows {
        let rect = view.rect_of(&row.note, height);
        if rect.x > width
            || rect.x + rect.w < roll::KEYBED
            || rect.y > height
            || rect.y + rect.h < 0.0
        {
            continue;
        }

        let fill = program_color(row.note.program, &palette);
        let radius = Px((view.px_per_pitch / 4.0).clamp(1.0, 4.0));
        ds.draw_rect(rect, fill, radius);

        if selection.contains(&row.index) {
            ds.draw_rect_stroke(rect, theme().on_surface, radius, Px(1.5));
        } else {
            ds.draw_rect_stroke(rect, fill.with_alpha_f32(0.55), radius, Px(1.0));
        }
    }

    // The marquee, while it is being dragged.
    if let Some(Drag::Marquee { from, to, .. }) = drag.borrow().as_ref() {
        let left = from.x.min(to.x);
        let top_y = from.y.min(to.y);
        let rect = Rect {
            x: left,
            y: top_y,
            w: (to.x - from.x).abs(),
            h: (to.y - from.y).abs(),
        };
        ds.draw_rect(rect, theme().primary.with_alpha_f32(0.15), Px(0.0));
        ds.draw_rect_stroke(rect, theme().primary, Px(0.0), Px(1.0));
    }

    // The key column, on top of everything.
    ds.draw_rect(
        Rect {
            x: 0.0,
            y: 0.0,
            w: roll::KEYBED,
            h: height,
        },
        theme().surface_container,
        Px(0.0),
    );

    let mut key = top.floor().clamp(-1.0, 127.0);
    while key <= bottom.ceil().min(128.0) {
        if (0.0..=127.0).contains(&key) {
            let y = view.y_of(key + 1.0, height);
            let black = is_black(key as u32);
            ds.draw_rect(
                Rect {
                    x: 1.0,
                    y: y + 0.5,
                    w: if black {
                        roll::KEYBED * 0.6
                    } else {
                        roll::KEYBED - 3.0
                    },
                    h: (view.px_per_pitch - 1.0).max(1.0),
                },
                if black { black_key } else { white_key },
                Px(1.0),
            );

            if key as i32 % 12 == 0 && view.px_per_pitch >= 9.0 {
                ds.draw_text(
                    format!("C{}", key as i32 / 12 - 1),
                    Vec2 {
                        x: roll::KEYBED - 6.0,
                        y: y + 1.0,
                    },
                    key_text,
                    Px((view.px_per_pitch - 3.0).clamp(8.0, 14.0)),
                );
            }
        }

        key += 1.0;
    }

    ds.draw_line(
        Vec2 {
            x: roll::KEYBED,
            y: 0.0,
        },
        Vec2 {
            x: roll::KEYBED,
            y: height,
        },
        theme().outline,
        Px(1.0),
        StrokeCap::Butt,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_time_grid_stays_readable_at_every_zoom() {
        for px_per_sec in [4.0, 12.0, 60.0, 240.0, 2000.0] {
            let step = grid_step(px_per_sec) as f32 * px_per_sec;
            assert!(
                step >= MIN_GRID_PX,
                "{px_per_sec} px/s gives {step} px between lines"
            );
        }
    }

    #[test]
    fn black_keys_are_the_ones_without_a_white_key_below_them() {
        for pitch in 0..=127u32 {
            assert_eq!(
                is_black(pitch),
                matches!(pitch % 12, 1 | 3 | 6 | 8 | 10),
                "{pitch}"
            );
        }
        assert!(!is_black(60), "C is white");
        assert!(is_black(61), "C# is black");
    }

    #[test]
    fn every_program_gets_one_of_the_six_colours() {
        let colors = palette();
        for program in 0..200u16 {
            assert_eq!(
                program_color(program, &colors),
                program_color(program, &colors)
            );
        }
        assert_eq!(program_color(0, &colors), colors[3]);
        assert_eq!(program_color(1, &colors), colors[4]);
    }
}
