//! The track list: one row per instrument the transcription found.

use std::collections::HashSet;
use std::rc::Rc;

use neunote_midi::group_by_program;
use neunote_types::NoteEvent;
use repose_core::prelude::*;
use repose_material::material3::{Switch, SwitchConfig};
use repose_ui::scroll::{ScrollArea, remember_scroll_state};
use repose_ui::*;

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
) -> View {
    let mut children = Vec::with_capacity(rows.len());
    for (program, name, count, is_drum) in rows.iter() {
        let program = *program;
        let label = if *is_drum {
            format!("{name} (drums)")
        } else {
            name.clone()
        };
        let toggle = hidden.clone();
        children.push(
            Row(Modifier::new().padding(Dp(4.0)).gap(Dp(8.0))).child((
                Switch(
                    !toggle.get().contains(&program),
                    {
                        let toggle = toggle.clone();
                        move |visible| {
                            let mut hidden = (*toggle.get()).clone();
                            if visible {
                                hidden.remove(&program);
                            } else {
                                hidden.insert(program);
                            }
                            toggle.set(Rc::new(hidden));
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
        );
    }

    let scroll = remember_scroll_state("neunote:tracks");

    ScrollArea(
        Modifier::new()
            .width(Dp(260.0))
            .fill_max_height()
            .background(theme().surface_container_low),
        scroll,
        Column(Modifier::new().padding(Dp(4.0)).gap(Dp(2.0))).child(children),
    )
}
