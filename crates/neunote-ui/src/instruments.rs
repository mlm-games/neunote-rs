//! The instrument picker: what to transcribe, before there is anything to show
//! a track list for.
//!
//! The names are the model's own instrument groups, and picking none of them
//! means Automatic -- the model decides. Typing group ids into a text field
//! asked the user to learn 35 names; this asks them to read them.

use neunote_types::GroupId;
use repose_core::prelude::*;
use repose_material::material3::{Checkbox, CheckboxConfig};
use repose_ui::scroll::{ScrollArea, remember_scroll_state};
use repose_ui::*;

/// `GroupId::ALL_NAMED` is in General MIDI order, so the families fall out of
/// the numbering: each entry covers a range of positions in it.
const FAMILIES: [(&str, usize, usize); 8] = [
    ("Keys", 0, 3),
    ("Guitar & bass", 4, 8),
    ("Strings", 9, 15),
    ("Voice & ensemble", 16, 18),
    ("Brass", 19, 23),
    ("Winds", 24, 31),
    ("Synth", 32, 33),
    ("Drums", 34, 34),
];

pub(crate) fn view(chosen: Signal<Vec<GroupId>>) -> View {
    let automatic = chosen.get().is_empty();
    let mut children = vec![
        Row(Modifier::new()
            .padding(Dp(6.0))
            .gap(Dp(8.0))
            .align_items(AlignItems::CENTER))
        .child((
            Checkbox(
                automatic,
                {
                    let chosen = chosen.clone();
                    move |wanted| {
                        if wanted {
                            chosen.set(Vec::new());
                        }
                    }
                },
                CheckboxConfig::default(),
            ),
            Text("Automatic").size(Sp(13.0)),
            Spacer(),
            Text("all instrument groups")
                .size(Sp(11.0))
                .color(theme().on_surface_variant),
        )),
    ];

    let all = GroupId::ALL_NAMED;
    for (family, first, last) in FAMILIES {
        let last = last.min(all.len() - 1);
        if first > last {
            continue;
        }

        children.push(
            Text((*family).to_owned())
                .size(Sp(11.0))
                .color(theme().on_surface_variant),
        );

        for group in &all[first..=last] {
            let group = *group;
            children.push(
                Row(Modifier::new()
                    .padding(Dp(2.0))
                    .gap(Dp(8.0))
                    .align_items(AlignItems::CENTER))
                .child((
                    Checkbox(
                        chosen.get().contains(&group),
                        {
                            let chosen = chosen.clone();
                            move |wanted| {
                                let mut next = chosen.get().clone();
                                if wanted {
                                    if !next.contains(&group) {
                                        next.push(group);
                                    }
                                } else {
                                    next.retain(|held| *held != group);
                                }
                                chosen.set(next);
                            }
                        },
                        CheckboxConfig::default(),
                    ),
                    Text(group.name().unwrap_or_default().replace('_', " ")).size(Sp(13.0)),
                )),
            );
        }
    }

    let scroll = remember_scroll_state("neunote:instruments");
    ScrollArea(
        Modifier::new().fill_max_height(),
        scroll,
        Column(Modifier::new().padding(Dp(4.0)).gap(Dp(2.0))).child(children),
    )
}
