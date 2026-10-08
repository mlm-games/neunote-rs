//! The piano roll: every note as a rectangle on a time/pitch grid, with click
//! to select.

use std::collections::HashSet;
use std::rc::Rc;

use neunote_types::NoteEvent;
use repose_canvas::Canvas;
use repose_core::{StrokeCap, prelude::*};
use repose_ui::scroll::{ScrollAreaXY, remember_scroll_state_xy};
use repose_ui::*;

const ROW: f32 = 12.0;
const MIN_ROWS: f32 = 24.0;

/// Pixels per second that keeps grid lines at least this far apart.
const MIN_GRID_PX: f32 = 72.0;

struct Geometry {
    lowest: f32,
    rows: f32,
    width: f32,
    height: f32,
}

fn grid_step(zoom: f32) -> f32 {
    let mut step = 1.0;
    while step * zoom < MIN_GRID_PX {
        step *= 2.0;
        if step >= 16.0 {
            step *= 5.0;
        }
    }
    step
}

fn color_for(program: u16, palette: &[Color]) -> Color {
    palette[(program as usize * 7 + 3) % palette.len()]
}

pub(crate) fn view(
    notes: Rc<Vec<NoteEvent>>,
    hidden: Rc<HashSet<u16>>,
    zoom: f32,
    selected: Signal<Option<usize>>,
) -> View {
    let visible = Rc::new(
        notes
            .iter()
            .enumerate()
            .filter(|(_, note)| !hidden.contains(&note.program))
            .map(|(index, note)| (index, note.onset, note.offset, note.pitch, note.program))
            .collect::<Vec<_>>(),
    );

    let lowest = visible
        .iter()
        .map(|(_, _, _, pitch, _)| f32::from(*pitch))
        .fold(127.0, f32::min);
    let highest = visible
        .iter()
        .map(|(_, _, _, pitch, _)| f32::from(*pitch))
        .fold(0.0, f32::max);

    let rows = (highest - lowest + 1.0).max(MIN_ROWS);
    let seconds = visible
        .iter()
        .map(|(_, _, offset, _, _)| *offset)
        .fold(0.0f64, f64::max) as f32;
    let geometry = Rc::new(Geometry {
        lowest,
        rows,
        width: (seconds * zoom).max(64.0),
        height: rows * ROW,
    });

    let click_notes = visible.clone();
    let click_geometry = geometry.clone();
    let click_selected = selected.clone();

    let draw_notes = visible.clone();
    let draw_geometry = geometry.clone();
    let step = grid_step(zoom);

    let canvas = Canvas(
        Modifier::new()
            .fill_max_size()
            .on_pointer_down(move |event| {
                let point = event.position;
                let hit = click_notes
                    .iter()
                    .rev()
                    .find(|(_, onset, offset, pitch, program)| {
                        let rect = rect_of(*onset, *offset, *pitch, *program, &click_geometry, 0.0);
                        point.x >= rect.x
                            && point.x <= rect.x + rect.w
                            && point.y >= rect.y
                            && point.y <= rect.y + rect.h
                    });
                click_selected.set(hit.map(|(index, ..)| *index));
            }),
        move |ds| {
            let palette = [
                theme().primary,
                theme().secondary,
                theme().tertiary,
                theme().error,
                theme().primary_container,
                theme().secondary_container,
            ];
            let grid = theme().outline_variant;
            let c_line = theme().surface_container_highest;

            let width = draw_geometry.width;
            let height = draw_geometry.height;
            let mut at = 0.0;
            while at <= width {
                ds.draw_line(
                    Vec2 { x: at, y: 0.0 },
                    Vec2 { x: at, y: height },
                    grid,
                    Px(1.0),
                    StrokeCap::Butt,
                );
                at += step * zoom;
            }

            let mut row = 0.0;
            while row <= rows_limit(&draw_geometry) {
                let pitch = draw_geometry.lowest + row;
                let y = (rows_limit(&draw_geometry) - row) * ROW;
                if (pitch as i32) % 12 == 0 {
                    ds.draw_line(
                        Vec2 { x: 0.0, y },
                        Vec2 { x: width, y },
                        c_line,
                        Px(1.0),
                        StrokeCap::Butt,
                    );
                }
                row += 1.0;
            }

            for (_, onset, offset, pitch, program) in draw_notes.iter() {
                let rect = rect_of(*onset, *offset, *pitch, *program, &draw_geometry, zoom);
                ds.draw_rect(rect, color_for(*program, &palette), Px(2.0));
            }
        },
    );

    let scroll = remember_scroll_state_xy("neunote:piano-roll");

    ScrollAreaXY(
        Modifier::new().fill_max_size().background(theme().surface),
        scroll,
        Box(Modifier::new()
            .width(Dp(geometry.width))
            .height(Dp(geometry.height)))
        .child(canvas),
    )
}

fn rows_limit(geometry: &Geometry) -> f32 {
    geometry.rows - 1.0
}

fn rect_of(
    onset: f64,
    offset: f64,
    pitch: u8,
    _program: u16,
    geometry: &Geometry,
    zoom: f32,
) -> Rect {
    let x = onset as f32 * zoom;
    let w = ((offset - onset) as f32 * zoom).max(2.0);
    let row = f32::from(pitch) - geometry.lowest;
    let y = (geometry.rows - 1.0 - row) * ROW;
    Rect {
        x,
        y,
        w,
        h: ROW - 1.0,
    }
}
