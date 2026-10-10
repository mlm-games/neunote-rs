//! The track list: one row per instrument the transcription found, with what
//! the mix does with it.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use neunote_midi::group_by_program;
use neunote_types::NoteEvent;
use repose_core::prelude::*;
use repose_material::material3::{Slider, Switch, SwitchConfig};
use repose_ui::scroll::{ScrollArea, remember_scroll_state};
use repose_ui::*;

use crate::view::{Symbols, TrackMix, fixed_slider_width, icon_button};

/// Writes one track's mix and tells whoever is playing about it.
type TrackWriter = Rc<dyn Fn(u16, &dyn Fn(&mut TrackMix))>;

/// Group notes the way the MIDI writer does, so the list and the exported file
/// cannot disagree about what a track is.
pub(crate) fn rows(notes: &[NoteEvent]) -> Rc<Vec<(u16, String, usize, bool)>> {
    Rc::new(
        group_by_program(notes)
            .unwrap_or_default()
            .into_iter()
            .map(|track| (track.program, track.name, track.notes.len(), track.is_drum))
            .collect(),
    )
}

pub(crate) fn view(
    rows: Rc<Vec<(u16, String, usize, bool)>>,
    hidden: Signal<Rc<HashSet<u16>>>,
    mix: Signal<Rc<HashMap<u16, TrackMix>>>,
    publish: Rc<dyn Fn()>,
) -> View {
    let store = mix.clone();
    let announce = publish.clone();
    let write: TrackWriter = Rc::new(move |program, changed| {
        let mut next = (*store.get()).clone();
        changed(next.entry(program).or_default());
        store.set(Rc::new(next));
        announce();
    });

    let mut children = Vec::with_capacity(rows.len());
    for (program, name, count, is_drum) in rows.iter() {
        let program = *program;
        let label = if *is_drum {
            format!("{name} (drums)")
        } else {
            name.clone()
        };
        let shown = hidden.clone();
        let current = mix.get().get(&program).copied().unwrap_or_default();

        children.push(
            Column(
                Modifier::new()
                    .padding(Dp(4.0))
                    .gap(Dp(2.0))
                    .fill_max_width(),
            )
            .child((
                Row(Modifier::new()
                    .gap(Dp(8.0))
                    .align_items(AlignItems::CENTER)
                    .fill_max_width())
                .child((
                    Switch(
                        !shown.get().contains(&program),
                        {
                            let shown = shown.clone();
                            move |visible| {
                                let mut hidden = (*shown.get()).clone();
                                if visible {
                                    hidden.remove(&program);
                                } else {
                                    hidden.insert(program);
                                }
                                shown.set(Rc::new(hidden));
                            }
                        },
                        SwitchConfig::default(),
                    ),
                    Text(label).size(Sp(13.0)),
                    Spacer(),
                    Text(format!("{count}"))
                        .size(Sp(12.0))
                        .color(theme().on_surface_variant),
                )),
                Row(Modifier::new()
                    .gap(Dp(2.0))
                    .align_items(AlignItems::CENTER)
                    .fill_max_width())
                .child((
                    icon_button(Symbols::MUTE, "Mute", current.muted, {
                        {
                            let write = write.clone();
                            move || write(program, &|track| track.muted = !track.muted)
                        }
                    }),
                    icon_button(Symbols::SOLO, "Only this", current.solo, {
                        {
                            let write = write.clone();
                            move || write(program, &|track| track.solo = !track.solo)
                        }
                    }),
                    Slider(
                        current.gain,
                        (0.0, 1.5),
                        Some(0.05),
                        {
                            let write = write.clone();
                            move |value| write(program, &|track| track.gain = value)
                        },
                        fixed_slider_width(Dp(92.0)),
                    ),
                )),
            )),
        );
    }

    let scroll = remember_scroll_state("neunote:tracks");
    ScrollArea(
        Modifier::new().fill_max_height().fill_max_width(),
        scroll,
        Column(
            Modifier::new()
                .padding(Dp(4.0))
                .gap(Dp(6.0))
                .fill_max_width(),
        )
        .child(children),
    )
}
