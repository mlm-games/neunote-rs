//! The recording's own shape: min/max peaks along the top of the roll, with the
//! playhead over them and a press that moves it.
//!
//! Hearing the source is half of checking a transcription and seeing it is the
//! other half; it is also what puts the roll's grid in context, instead of
//! floating free in the middle of the window.

use std::cell::Cell;
use std::rc::Rc;

use repose_canvas::{Canvas, DrawScope};
use repose_core::{PointerEventKind, prelude::*};
use repose_ui::*;

use crate::view::Source;

pub(crate) fn view(
    source: Option<Rc<Source>>,
    head: f64,
    size: Signal<Vec2>,
    on_seek: impl Fn(f64) + 'static,
) -> View {
    // A pointer event carries no size, so the paint writes its own down and a
    // press reads it.
    let seek = Rc::new(on_seek);
    let dragging = Rc::new(Cell::new(false));

    let draw = {
        let source = source.clone();
        let size = size.clone();
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

            let middle = height * 0.5;
            let half = height * 0.44;

            ds.draw_rect(
                Rect {
                    x: 0.0,
                    y: height - 1.0,
                    w: width,
                    h: 1.0,
                },
                Color::from_rgba(0x8c, 0x8c, 0x99, 0x4d),
                Px(0.0),
            );

            if let Some(source) = source.as_ref() {
                let buckets = source.peaks.len().max(1);
                for x in 0..width as usize {
                    let bucket = x * buckets / width as usize;
                    let (low, high) = source.peaks[bucket];
                    let top = middle - high * half;
                    ds.draw_rect(
                        Rect {
                            x: x as f32,
                            y: top,
                            w: 1.0,
                            h: (middle - low * half - top).max(1.0),
                        },
                        Color::from_rgba(0x6b, 0x9e, 0xeb, 0xcc),
                        Px(0.0),
                    );
                }
            }

            let duration = source.as_ref().map_or(1.0, |source| source.duration as f32);
            let at = (head as f32 / duration).clamp(0.0, 1.0) * width;
            ds.draw_rect(
                Rect {
                    x: at - 0.5,
                    y: 0.0,
                    w: 1.0,
                    h: height,
                },
                Color::from_rgba(0xf2, 0x8c, 0x59, 0xf2),
                Px(0.0),
            );
        }
    };

    let seek_to = {
        let size = size.clone();
        let seek = seek.clone();
        move |x: f32| {
            let measured = size.get();
            if measured.x <= 0.0 {
                return;
            }

            seek(f64::from((x / measured.x).clamp(0.0, 1.0)));
        }
    };

    Box(Modifier::new()
        .fill_max_width()
        .height(Dp(56.0))
        .background(Color::from_rgba(0x1c, 0x1f, 0x26, 0xe6))
        .on_pointer_down({
            let dragging = dragging.clone();
            let seek_to = seek_to.clone();
            move |event| {
                if matches!(event.kind, repose_core::PointerKind::Mouse)
                    && matches!(event.event, PointerEventKind::Down(_))
                {
                    dragging.set(true);
                    seek_to(event.position.x);
                }
            }
        })
        .on_pointer_move({
            let dragging = dragging.clone();
            let seek_to = seek_to.clone();
            move |event| {
                if dragging.get() && matches!(event.event, PointerEventKind::Move) {
                    seek_to(event.position.x);
                }
            }
        })
        .on_pointer_up({
            let dragging = dragging.clone();
            move |event| {
                if matches!(event.event, PointerEventKind::Up(_)) {
                    dragging.set(false);
                }
            }
        }))
    .child(Canvas(Modifier::new().fill_max_size(), draw))
}
