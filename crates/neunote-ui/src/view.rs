//! The root view: state, the job, and the surfaces around it -- toolbar,
//! track list and piano roll, status line.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use neunote_audio::{decode_bytes, to_engine_input};
use neunote_midi::midi_bytes;
use neunote_types::{GroupId, ModelSize};
use repose_core::prelude::*;
use repose_core::shortcuts::ShortcutMap;
use repose_core::{RenderContext, shortcuts, timer};
use repose_material::material3::{
    Button, ButtonConfig, DropdownMenu, DropdownMenuConfig, DropdownMenuEntry, DropdownMenuItem,
    FilledTonalButton, LinearProgressIndicator, LinearProgressIndicatorConfig, MenuState,
    RadioButton, RadioButtonConfig, SegmentConfig, SegmentedButton, SegmentedButtonConfig, Slider,
    SliderConfig, Switch, SwitchConfig, TextButton, TextField, TextFieldConfig,
};
use repose_material::{Icon, Symbol, material_symbols};
use repose_ui::*;
use web_time::{Duration, Instant};

use crate::edit::Editor;
use crate::job::{Job, Message, Weights};
use crate::quantize::{Division, NOTE_NAMES, Quantize, Scale, Snap};
use crate::roll::{self, Viewport};
use crate::{piano_roll, tracks};

// Codepoints from the bundled Material Symbols Outlined face, for the actions
// whose glyph means the same thing everywhere.
material_symbols! {
    FOLDER: '\u{E2C7}',
    INBOX: '\u{E156}',
    CLOUD: '\u{F15C}',
    MUSIC_NOTE: '\u{E405}',
    CLOSE: '\u{E5CD}',
    TUNE: '\u{E429}',
}

/// A control's icon and its label. The icon takes its colour from whatever
/// content colour is in scope, which is what a button sets for its label.
fn with_icon(symbol: Symbol, label: impl Into<String>) -> View {
    Row(Modifier::new().gap(Dp(6.0)).align_items(AlignItems::CENTER))
        .child((Icon(symbol).size(Sp(16.0)), Text(label.into())))
}

/// An audio file the host has in hand, named for error messages.
pub struct LoadedAudio {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// A checkpoint the host has in hand.
pub struct LoadedWeights {
    pub name: String,
    pub bytes: Vec<u8>,
}

type Picked = Rc<dyn Fn(Option<LoadedAudio>)>;
type PickedWeights = Rc<dyn Fn(Option<LoadedWeights>)>;
type Saver = Rc<dyn Fn(&str, &[u8]) -> Result<String, String>>;
type Progress = Rc<dyn Fn(u64, u64)>;
type Bytes = Rc<dyn Fn(Result<Rc<Vec<u8>>, String>)>;

/// What the view needs from its platform. The desktop shell answers with
/// dialogs and the model cache; the web shell with a file input, a download,
/// and a browser store.
pub struct Shell {
    pub pick_audio: Rc<dyn Fn(Picked)>,
    pub pick_weights: Rc<dyn Fn(PickedWeights)>,
    /// A verified checkpoint already on this machine, if there is one.
    pub cached_weights: Rc<dyn Fn(ModelSize) -> Option<PathBuf>>,
    /// Checkpoint bytes this host already holds. Asynchronous because a browser
    /// reads them back out of its own store; an error says what to do about it.
    pub resolve_weights: Rc<dyn Fn(ModelSize, Bytes)>,
    /// Fetch a checkpoint into the host's own store. `accepted` is the user's
    /// answer on the weights' licence; this host enforces it, not the view.
    pub fetch_weights: Rc<dyn Fn(ModelSize, bool, Progress, Bytes)>,
    /// Whether this host already holds the user's answer on the weights'
    /// licence, so the switch starts from it instead of from nothing.
    pub licence_accepted: Rc<dyn Fn() -> bool>,
    /// Record that answer. `false` withdraws it where the host can forget.
    pub set_licence_accepted: Rc<dyn Fn(bool)>,
    pub save_midi: Saver,
}

struct Source {
    name: String,
    samples: Arc<Vec<f32>>,
    duration: f64,
}

#[derive(Clone, Default)]
enum Phase {
    #[default]
    Idle,
    Working {
        done: usize,
        total: usize,
    },
    Done,
    Cancelled,
    Failed,
}

/// The root view function the shells hand to their platform.
pub fn root(shell: Shell) -> impl FnMut(&mut Scheduler, &RenderContext) -> View + 'static {
    move |scheduler, _context| app(&shell, scheduler)
}

fn app(shell: &Shell, _scheduler: &mut Scheduler) -> View {
    let dark = remember(|| signal(true));
    let theme = if dark.get() {
        Theme::dark()
    } else {
        Theme::light()
    };

    // The theme is a composition local, so it has to be in place while the tree
    // is built rather than applied to a tree that already read the old one.
    with_theme(theme, || body(shell, (*dark).clone()))
}

fn body(shell: &Shell, dark: Signal<bool>) -> View {
    let phase = remember(|| signal(Phase::Idle));
    let status = remember(|| signal(String::from("Open an audio file to begin.")));
    let source = remember(|| signal(None::<Rc<Source>>));
    let raw = remember(|| signal(Rc::new(Vec::new())));
    let selection = remember(|| signal(Rc::new(BTreeSet::new())));
    let editor = Editor::new((*raw).clone(), (*selection).clone());
    let picked_weights = remember(|| signal(None::<Rc<Vec<u8>>>));
    let size = remember(|| signal(ModelSize::Small));
    let prelude = remember(|| signal(true));
    let instruments = remember(|| signal(String::new()));
    let hidden = remember(|| signal(Rc::new(HashSet::<u16>::new())));
    let licence = remember(|| signal((shell.licence_accepted)()));
    let viewport = remember(|| signal(Viewport::default()));
    let roll_size = remember_with_key("neunote:roll-size", || signal(Vec2 { x: 0.0, y: 0.0 }));
    let quantize = remember(|| signal(Quantize::default()));
    let quantising = remember(|| signal(false));
    let started: Rc<Cell<Option<Instant>>> =
        remember_with_key("neunote:started", || Cell::new(None));

    let job_slot: Rc<RefCell<Option<Job>>> =
        remember_with_key("neunote:job", || RefCell::new(None));

    // The worker reports on a channel; this drains it into the signals.
    scoped_effect_once({
        let job_slot = job_slot.clone();
        let phase = (*phase).clone();
        let status = (*status).clone();
        let editor = editor.clone();
        let viewport = (*viewport).clone();
        let roll_size = (*roll_size).clone();
        let started = Rc::clone(&started);

        move || {
            let mut handle = Some(timer::interval(Duration::from_millis(120), move || {
                let messages = match job_slot.borrow().as_ref() {
                    Some(job) => job.drain(),
                    None => Vec::new(),
                };
                if messages.is_empty() {
                    return;
                }

                let mut finished = None;
                for message in messages {
                    match message {
                        Message::Progress {
                            done,
                            total,
                            finalized,
                        } => {
                            phase.set(Phase::Working { done, total });
                            status.set(format!(
                                "chunk {done}/{total} · {:.0}s of audio final · ~{}s left",
                                finalized,
                                remaining(started.get(), done, total)
                            ));
                        }
                        Message::Notes(notes) => editor.stream(notes),
                        Message::Failed(error) => {
                            phase.set(Phase::Failed);
                            status.set(error);
                        }
                        Message::Cancelled => {
                            phase.set(Phase::Cancelled);
                            status.set("cancelled -- the notes it had are kept".to_owned());
                        }
                        Message::Finished(transcribed) => finished = Some(transcribed),
                    }
                }

                if let Some(transcribed) = finished {
                    let size = roll_size.get();
                    viewport.set(Viewport::fit(&transcribed, size.x, size.y));
                    editor.reset(transcribed);
                    started.set(None);
                    phase.set(Phase::Done);
                    status.set(String::from("done"));
                }
            }));

            Dispose::new(move || {
                if let Some(handle) = handle.take() {
                    handle.cancel();
                }
            })
        }
    });

    // --- actions -----------------------------------------------------------

    let on_open = {
        let phase = (*phase).clone();
        let status = (*status).clone();
        let source = (*source).clone();
        let editor = editor.clone();
        let viewport = (*viewport).clone();
        let picker = shell.pick_audio.clone();

        Rc::new(move || {
            // The picker may answer later, so the callback takes its own
            // handles and this closure stays callable more than once.
            let phase = phase.clone();
            let status = status.clone();
            let source = source.clone();
            let editor = editor.clone();
            let viewport = viewport.clone();

            picker(Rc::new(move |loaded| {
                let Some(loaded) = loaded else {
                    return;
                };

                phase.set(Phase::Working { done: 0, total: 0 });
                status.set(format!("decoding {}…", loaded.name));

                match decode_bytes(&loaded.bytes, &loaded.name) {
                    Ok(buffer) => {
                        let seconds = buffer.duration_secs();
                        match to_engine_input(&buffer) {
                            Ok(mono) => {
                                status.set(format!("{} · {:.1}s", loaded.name, seconds));
                                source.set(Some(Rc::new(Source {
                                    name: loaded.name,
                                    samples: Arc::new(mono),
                                    duration: seconds,
                                })));
                                editor.reset(Vec::new());
                                viewport.set(Viewport::default());
                                phase.set(Phase::Idle);
                            }
                            Err(error) => {
                                phase.set(Phase::Failed);
                                status.set(format!("cannot resample {}: {error}", loaded.name));
                            }
                        }
                    }
                    Err(error) => {
                        phase.set(Phase::Failed);
                        status.set(format!("cannot decode {}: {error}", loaded.name));
                    }
                }
            }));
        })
    };

    let on_choose_weights = {
        let status = (*status).clone();
        let picked_weights = (*picked_weights).clone();
        let picker = shell.pick_weights.clone();

        move || {
            let status = status.clone();
            let picked_weights = picked_weights.clone();

            picker(Rc::new(move |loaded| {
                let Some(loaded) = loaded else {
                    return;
                };
                status.set(format!("checkpoint: {}", loaded.name));
                picked_weights.set(Some(Rc::new(loaded.bytes)));
            }));
        }
    };

    let on_transcribe = {
        let phase = (*phase).clone();
        let status = (*status).clone();
        let source = (*source).clone();
        let editor = editor.clone();
        let picked_weights = (*picked_weights).clone();
        let size = (*size).clone();
        let prelude = (*prelude).clone();
        let instruments = (*instruments).clone();
        let job_slot = job_slot.clone();
        let started = Rc::clone(&started);
        let cached = shell.cached_weights.clone();
        let resolve = shell.resolve_weights.clone();

        move || {
            let Some(source) = source.get() else {
                status.set("open an audio file first".to_owned());
                return;
            };

            let groups = match parse_groups(&instruments.get()) {
                Ok(groups) => groups,
                Err(error) => {
                    status.set(error);
                    return;
                }
            };

            // Whatever the host hands over -- a path, or bytes from its own
            // store -- this is where the run starts.
            let ready = {
                let phase = phase.clone();
                let status = status.clone();
                let editor = editor.clone();
                let job_slot = job_slot.clone();
                let size = size.clone();
                let prelude = prelude.clone();
                let started = Rc::clone(&started);

                Rc::new(move |weights: Weights| {
                    editor.reset(Vec::new());
                    phase.set(Phase::Working { done: 0, total: 0 });
                    status.set(format!(
                        "transcribing {:.1}s with {}…",
                        source.duration,
                        weights.label()
                    ));

                    started.set(Some(Instant::now()));
                    *job_slot.borrow_mut() = Some(Job::start(
                        weights,
                        Arc::clone(&source.samples),
                        size.get(),
                        groups.clone(),
                        prelude.get(),
                    ));
                })
            };

            match cached(size.get()) {
                Some(path) => ready(Weights::Path(path)),
                None => match picked_weights.get() {
                    Some(bytes) => ready(Weights::Bytes(Arc::new(bytes.as_ref().clone()))),
                    None => {
                        phase.set(Phase::Working { done: 0, total: 0 });
                        status.set("looking for a checkpoint…".to_owned());

                        let ready = Rc::clone(&ready);
                        let failed_phase = phase.clone();
                        let failed_status = status.clone();
                        resolve(
                            size.get(),
                            Rc::new(move |result| match result {
                                Ok(bytes) => {
                                    ready(Weights::Bytes(Arc::new(bytes.as_ref().clone())))
                                }
                                Err(error) => {
                                    failed_phase.set(Phase::Idle);
                                    failed_status.set(error);
                                }
                            }),
                        );
                    }
                },
            }
        }
    };

    let on_download = {
        let status = (*status).clone();
        let phase = (*phase).clone();
        let size = (*size).clone();
        let accepted = licence.clone();
        let fetch = shell.fetch_weights.clone();

        move || {
            let progress_status = status.clone();
            let done_status = status.clone();
            let done_phase = phase.clone();
            let size_now = size.get();

            fetch(
                size_now,
                accepted.get(),
                Rc::new(move |done, total| {
                    progress_status.set(format!(
                        "fetching the {size_now} weights: {:.0} of {:.0} MB",
                        done as f64 / (1024.0 * 1024.0),
                        total as f64 / (1024.0 * 1024.0)
                    ));
                }),
                Rc::new(move |result| {
                    done_phase.set(Phase::Idle);
                    done_status.set(match result {
                        Ok(bytes) => {
                            format!("checkpoint ready ({} MB)", bytes.len() / (1024 * 1024))
                        }
                        Err(error) => error,
                    });
                }),
            );
        }
    };

    let on_cancel = {
        let status = (*status).clone();
        let job_slot = job_slot.clone();

        move || {
            if let Some(job) = job_slot.borrow().as_ref() {
                job.cancel();
                status.set("cancelling after the current chunk…".to_owned());
            }
        }
    };

    let on_save = {
        let status = (*status).clone();
        let source = (*source).clone();
        let bpm = quantize.clone();
        let editor = editor.clone();
        let saver = shell.save_midi.clone();

        Rc::new(move || {
            let notes = editor.notes();
            if notes.is_empty() {
                status.set("nothing to save yet".to_owned());
                return;
            }

            let name = source
                .get()
                .map(|source| format!("{}.mid", stem(&source.name)))
                .unwrap_or_else(|| String::from("neunote.mid"));

            match midi_bytes(&notes, bpm.get().bpm) {
                Ok(bytes) => match saver(&name, &bytes) {
                    Ok(written) => {
                        let count = notes.len();
                        status.set(format!("wrote {written} ({count} notes)"));
                    }
                    Err(error) => status.set(format!("cannot write MIDI: {error}")),
                },
                Err(error) => status.set(format!("cannot build MIDI: {error}")),
            }
        })
    };

    let on_fit = {
        let viewport = (*viewport).clone();
        let roll_size = (*roll_size).clone();
        let editor = editor.clone();

        Rc::new(move || {
            let size = roll_size.get();
            if size.y > 0.0 {
                viewport.set(Viewport::fit(&editor.notes(), size.x, size.y));
            }
        })
    };

    let on_toggle_theme = {
        let dark = dark.clone();
        move || dark.update(|dark| *dark = !*dark)
    };

    // Quantisation is a view over the raw notes, so turning it on or moving a
    // knob is one call: the editor re-derives what the roll draws.
    let apply_quantize: Rc<dyn Fn()> = Rc::new({
        let editor = editor.clone();
        let quantising = (*quantising).clone();
        let quantize = (*quantize).clone();

        move || {
            let params = quantize.get();
            editor.set_quantize(quantising.get().then_some(params));
        }
    });

    let on_undo = {
        let editor = editor.clone();
        move || {
            editor.undo();
        }
    };
    let on_redo = {
        let editor = editor.clone();
        move || {
            editor.redo();
        }
    };

    // --- keyboard ----------------------------------------------------------

    scoped_effect_once({
        let editor = editor.clone();
        let dark = dark.clone();
        let open = on_open.clone();
        let save = on_save.clone();
        let fit = on_fit.clone();

        move || {
            let command = Modifiers {
                command: true,
                ctrl: !cfg!(target_os = "macos"),
                ..Modifiers::default()
            };
            let shift = Modifiers {
                shift: true,
                ..Modifiers::default()
            };
            let nothing = Modifiers::default();

            let map = ShortcutMap::new()
                .bind(
                    Key::Character('o'),
                    command,
                    shortcuts::Action::Custom("open".into()),
                )
                .bind(
                    Key::Character('f'),
                    command,
                    shortcuts::Action::Custom("fit".into()),
                )
                .bind(
                    Key::Character('d'),
                    command,
                    shortcuts::Action::Custom("theme".into()),
                )
                .bind(
                    Key::Delete,
                    nothing,
                    shortcuts::Action::Custom("delete".into()),
                )
                .bind(
                    Key::Backspace,
                    nothing,
                    shortcuts::Action::Custom("delete".into()),
                )
                .bind(
                    Key::Escape,
                    nothing,
                    shortcuts::Action::Custom("clear".into()),
                )
                .bind(
                    Key::ArrowLeft,
                    nothing,
                    shortcuts::Action::Custom("earlier".into()),
                )
                .bind(
                    Key::ArrowRight,
                    nothing,
                    shortcuts::Action::Custom("later".into()),
                )
                .bind(
                    Key::ArrowLeft,
                    shift,
                    shortcuts::Action::Custom("earlier-far".into()),
                )
                .bind(
                    Key::ArrowRight,
                    shift,
                    shortcuts::Action::Custom("later-far".into()),
                )
                .bind(
                    Key::ArrowUp,
                    nothing,
                    shortcuts::Action::Custom("sharper".into()),
                )
                .bind(
                    Key::ArrowDown,
                    nothing,
                    shortcuts::Action::Custom("flatter".into()),
                )
                .bind(
                    Key::ArrowUp,
                    shift,
                    shortcuts::Action::Custom("sharper-far".into()),
                )
                .bind(
                    Key::ArrowDown,
                    shift,
                    shortcuts::Action::Custom("flatter-far".into()),
                );

            let actions = editor.clone();
            let theme_signal = dark.clone();

            let handler: shortcuts::Handler = Rc::new(move |action| match action {
                shortcuts::Action::Undo => actions.undo(),
                shortcuts::Action::Redo => actions.redo(),
                shortcuts::Action::Save => {
                    save();
                    true
                }
                shortcuts::Action::Custom(name) => match &*name {
                    "open" => {
                        open();
                        true
                    }
                    "fit" => {
                        fit();
                        true
                    }
                    "theme" => {
                        theme_signal.update(|dark| *dark = !*dark);
                        true
                    }
                    "delete" => actions.delete_selected(),
                    "clear" => {
                        actions.clear_selection();
                        true
                    }
                    "earlier" => actions.nudge(-1, 0, actions.grid()),
                    "later" => actions.nudge(1, 0, actions.grid()),
                    "earlier-far" => actions.nudge(-4, 0, actions.grid()),
                    "later-far" => actions.nudge(4, 0, actions.grid()),
                    "sharper" => actions.nudge(0, 1, actions.grid()),
                    "flatter" => actions.nudge(0, -1, actions.grid()),
                    "sharper-far" => actions.nudge(0, 12, actions.grid()),
                    "flatter-far" => actions.nudge(0, -12, actions.grid()),
                    _ => false,
                },
                _ => false,
            });

            let map_dispose = shortcuts::install_shortcut_map_with_key("neunote", map);
            let handler_dispose = shortcuts::install_shortcut_handler_with_key("neunote", handler);

            Dispose::new(move || {
                map_dispose.run();
                handler_dispose.run();
            })
        }
    });

    // --- chrome ------------------------------------------------------------

    let weights_label = match (shell.cached_weights)(size.get()) {
        Some(path) => format!("Checkpoint: {}", path.display()),
        None if picked_weights.get().is_some() => String::from("Checkpoint: picked file"),
        None => String::from("Checkpoint: none"),
    };

    let size_row = Row(Modifier::new().gap(Dp(2.0)).align_items(AlignItems::CENTER)).child(
        ModelSize::ALL
            .into_iter()
            .map(|candidate| {
                let chosen = (*size).clone();
                Row(Modifier::new().gap(Dp(4.0)).align_items(AlignItems::CENTER)).child((
                    RadioButton(
                        size.get() == candidate,
                        move || chosen.set(candidate),
                        RadioButtonConfig::default(),
                    ),
                    Text(candidate.as_str()).size(Sp(12.0)),
                ))
            })
            .collect::<Vec<_>>(),
    );

    // Three rows: the file and model controls, the run controls, and the
    // quantise panel with the editing commands. One row overflows at the
    // default window width.
    let toolbar = Column(Modifier::new().padding(Dp(8.0)).gap(Dp(6.0))).child(vec![
        Row(Modifier::new().gap(Dp(8.0)).align_items(AlignItems::CENTER)).child(vec![
            TextButton(
                Modifier::new(),
                click(on_open.clone()),
                ButtonConfig::default(),
                || with_icon(Symbols::FOLDER, "Open audio"),
            ),
            TextButton(
                Modifier::new(),
                on_choose_weights,
                ButtonConfig::default(),
                || with_icon(Symbols::INBOX, weights_label.clone()),
            ),
            size_row,
            Spacer(),
            Row(Modifier::new().gap(Dp(4.0)).align_items(AlignItems::CENTER)).child((
                Switch(
                    prelude.get(),
                    {
                        let prelude = prelude.clone();
                        move |on| prelude.set(on)
                    },
                    SwitchConfig::default(),
                ),
                Text("force ties").size(Sp(12.0)),
            )),
        ]),
        Row(Modifier::new().gap(Dp(8.0)).align_items(AlignItems::CENTER)).child(vec![
            TextField(
                Modifier::new().width(Dp(320.0)),
                instruments.get(),
                {
                    let instruments = instruments.clone();
                    move |value| instruments.set(value)
                },
                TextFieldConfig {
                    label: Some(String::from("instruments (comma separated, empty = all)")),
                    ..Default::default()
                },
            ),
            Spacer(),
            Row(Modifier::new().gap(Dp(4.0)).align_items(AlignItems::CENTER)).child((
                Switch(
                    licence.get(),
                    {
                        let licence = licence.clone();
                        let record = shell.set_licence_accepted.clone();
                        move |on| {
                            licence.set(on);
                            record(on);
                        }
                    },
                    SwitchConfig::default(),
                ),
                Text("accept CC BY-NC weights").size(Sp(12.0)),
            )),
            TextButton(
                Modifier::new(),
                on_download,
                ButtonConfig::default(),
                || with_icon(Symbols::CLOUD, "Download weights"),
            ),
            Button(
                Modifier::new(),
                on_transcribe,
                ButtonConfig::default(),
                || with_icon(Symbols::MUSIC_NOTE, "Transcribe"),
            ),
            TextButton(Modifier::new(), on_cancel, ButtonConfig::default(), || {
                with_icon(Symbols::CLOSE, "Cancel")
            }),
            FilledTonalButton(
                Modifier::new(),
                click(on_save.clone()),
                ButtonConfig::default(),
                || Text("Save MIDI"),
            ),
        ]),
        Row(Modifier::new().gap(Dp(8.0))).child(vec![
            quantise_panel(&quantising, &quantize, &apply_quantize),
            Spacer(),
            TextButton(
                Modifier::new(),
                on_undo,
                ButtonConfig {
                    enabled: editor.can_undo(),
                    ..Default::default()
                },
                || Text("Undo"),
            ),
            TextButton(
                Modifier::new(),
                on_redo,
                ButtonConfig {
                    enabled: editor.can_redo(),
                    ..Default::default()
                },
                || Text("Redo"),
            ),
            TextButton(
                Modifier::new(),
                click(on_fit.clone()),
                ButtonConfig::default(),
                || Text("Fit"),
            ),
            TextButton(
                Modifier::new(),
                on_toggle_theme,
                ButtonConfig::default(),
                || Text(if dark.get() { "Light" } else { "Dark" }),
            ),
        ]),
    ]);

    // --- body --------------------------------------------------------------

    let shown_notes = editor.notes();
    let track_panel = if shown_notes.is_empty() {
        Box(Modifier::new().width(Dp(260.0)).fill_max_height())
    } else {
        tracks::view(tracks::rows(&shown_notes), (*hidden).clone())
    };

    let roll = piano_roll::view(
        editor.clone(),
        (*hidden).get(),
        (*viewport).clone(),
        (*roll_size).clone(),
    );

    let selection = {
        let chosen = editor.selection();
        let notes = editor.notes();
        match chosen.first() {
            Some(index) => {
                let count = chosen.len();
                match notes.get(*index) {
                    Some(note) => format!(
                        "{} note{} · {} · pitch {} · {:.2}s → {:.2}s",
                        count,
                        if count == 1 { "" } else { "s" },
                        neunote_types::instrument_label(note.program),
                        note.pitch,
                        note.onset,
                        note.offset
                    ),
                    None => String::new(),
                }
            }
            None => String::new(),
        }
    };

    let (fraction, progress_label) = match phase.get() {
        Phase::Working { done, total } if total > 0 => {
            (Some(done as f32 / total as f32), format!("{done}/{total}"))
        }
        Phase::Working { .. } => (None, String::from("working…")),
        Phase::Done => (Some(1.0), String::from("done")),
        Phase::Cancelled => (Some(1.0), String::from("cancelled")),
        _ => (Some(0.0), String::new()),
    };

    let view_now = viewport.get();
    let footer = Row(Modifier::new().padding(Dp(8.0)).gap(Dp(12.0))).child((
        Box(Modifier::new().width(Dp(200.0))).child(LinearProgressIndicator(
            fraction,
            LinearProgressIndicatorConfig::default(),
        )),
        Text(format!(
            "{}  {progress_label}{}",
            status.get(),
            if editor.is_quantised() {
                "  · quantised"
            } else {
                ""
            }
        ))
        .size(Sp(13.0)),
        Spacer(),
        Text(selection)
            .size(Sp(12.0))
            .color(theme().on_surface_variant),
        labelled(
            "time",
            sized(
                Dp(200.0),
                Slider(
                    view_now.px_per_sec,
                    (roll::MIN_PX_PER_SEC, roll::MAX_PX_PER_SEC),
                    None,
                    {
                        let viewport = viewport.clone();
                        move |value| {
                            viewport.update(|view| view.px_per_sec = value);
                        }
                    },
                    SliderConfig::default(),
                ),
            ),
        ),
        labelled(
            "rows",
            sized(
                Dp(200.0),
                Slider(
                    view_now.px_per_pitch,
                    (roll::MIN_PX_PER_PITCH, roll::MAX_PX_PER_PITCH),
                    Some(1.0),
                    {
                        let viewport = viewport.clone();
                        move |value| {
                            viewport.update(|view| view.px_per_pitch = value);
                        }
                    },
                    SliderConfig::default(),
                ),
            ),
        ),
    ));

    Column(Modifier::new().fill_max_size()).child((
        toolbar,
        Row(Modifier::new().fill_max_size()).child((track_panel, roll)),
        footer,
    ))
}

/// Give a material widget a width of its own.
///
/// Not `View::modifier`: that replaces the widget's modifier, and a widget's
/// painter, size and focus all live on it -- a slider re-modified this way draws
/// nothing at all.
fn sized(width: Dp, control: View) -> View {
    Box(Modifier::new().width(width).align_self_center()).child(control)
}

/// Hand a shared action to a widget that wants a plain closure.
fn click(action: Rc<dyn Fn()>) -> impl Fn() + 'static {
    move || action()
}

/// A short caption in front of a control.
fn labelled(label: &str, control: View) -> View {
    Row(Modifier::new().gap(Dp(4.0)).align_items(AlignItems::CENTER)).child((
        Text(label.to_owned())
            .size(Sp(11.0))
            .color(theme().on_surface_variant),
        control,
    ))
}

/// The quantise panel. Off means the roll shows exactly what the model
/// produced; on means the roll shows the raw notes put through these settings,
/// and nothing about the raw list has moved.
fn quantise_panel(on: &Signal<bool>, params: &Signal<Quantize>, apply: &Rc<dyn Fn()>) -> View {
    let mut children = vec![
        Row(Modifier::new().gap(Dp(4.0)).align_items(AlignItems::CENTER)).child((
            Icon(Symbols::TUNE)
                .size(Sp(16.0))
                .color(theme().on_surface_variant),
            Switch(
                on.get(),
                {
                    let on = on.clone();
                    let apply = Rc::clone(apply);
                    move |value| {
                        on.set(value);
                        apply();
                    }
                },
                SwitchConfig::default(),
            ),
            Text("quantise").size(Sp(12.0)),
        )),
    ];

    if !on.get() {
        children.push(
            Text("off: the roll shows the transcription as produced")
                .size(Sp(11.0))
                .color(theme().on_surface_variant),
        );
        return Row(Modifier::new().gap(Dp(8.0)).align_items(AlignItems::CENTER)).child(children);
    }

    let current = params.get();

    children.push(labelled(
        "scale",
        Switch(
            current.pitches,
            {
                let params = params.clone();
                let apply = apply.clone();
                move |value| {
                    params.update(|params| params.pitches = value);
                    apply();
                }
            },
            SwitchConfig::default(),
        ),
    ));

    children.push(labelled(
        "root",
        menu(
            remember(MenuState::new),
            NOTE_NAMES[current.root as usize].to_owned(),
            NOTE_NAMES
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    let params = params.clone();
                    let apply = apply.clone();
                    (
                        (*name).to_owned(),
                        Rc::new(move || {
                            params.update(|params| params.root = index as u8);
                            apply();
                        }) as Rc<dyn Fn()>,
                    )
                })
                .collect(),
        ),
    ));

    children.push(labelled(
        "",
        menu(
            remember(MenuState::new),
            current.scale.name().to_owned(),
            Scale::ALL
                .iter()
                .map(|scale| {
                    let params = params.clone();
                    let apply = apply.clone();
                    (
                        scale.name().to_owned(),
                        Rc::new(move || {
                            params.update(|params| params.scale = *scale);
                            apply();
                        }) as Rc<dyn Fn()>,
                    )
                })
                .collect(),
        ),
    ));

    children.push(SegmentedButton(
        &[Snap::ALL
            .iter()
            .position(|snap| *snap == current.snap)
            .unwrap_or(1)],
        Snap::ALL
            .iter()
            .map(|snap| SegmentConfig {
                label: snap.name().into(),
                icon: None,
                on_click: Rc::new({
                    let params = params.clone();
                    let apply = apply.clone();
                    move || {
                        params.update(|params| params.snap = *snap);
                        apply();
                    }
                }),
                enabled: current.pitches,
                ..Default::default()
            })
            .collect(),
        SegmentedButtonConfig::default(),
    ));

    children.push(labelled(
        "grid",
        Switch(
            current.times,
            {
                let params = params.clone();
                let apply = apply.clone();
                move |value| {
                    params.update(|params| params.times = value);
                    apply();
                }
            },
            SwitchConfig::default(),
        ),
    ));

    children.push(menu(
        remember(MenuState::new),
        current.division.label().to_owned(),
        Division::ALL
            .iter()
            .map(|division| {
                let params = params.clone();
                let apply = apply.clone();
                (
                    division.label().to_owned(),
                    Rc::new(move || {
                        params.update(|params| params.division = *division);
                        apply();
                    }) as Rc<dyn Fn()>,
                )
            })
            .collect(),
    ));

    // Tempo and strength mean nothing until the grid is on, and they are wide
    // enough on their own to need a row of their own.
    if !current.times {
        return Row(Modifier::new().gap(Dp(8.0)).align_items(AlignItems::CENTER)).child(children);
    }

    children.push(labelled(
        "tempo",
        sized(
            Dp(200.0),
            Slider(
                current.bpm as f32,
                (40.0, 240.0),
                Some(1.0),
                {
                    let params = params.clone();
                    let apply = apply.clone();
                    move |value| {
                        params.update(|params| params.bpm = f64::from(value));
                        apply();
                    }
                },
                SliderConfig::default(),
            ),
        ),
    ));

    children.push(labelled(
        &format!("{}%", (current.strength * 100.0).round() as i32),
        sized(
            Dp(200.0),
            Slider(
                current.strength as f32,
                (0.0, 1.0),
                Some(0.05),
                {
                    let params = params.clone();
                    let apply = apply.clone();
                    move |value| {
                        params.update(|params| params.strength = f64::from(value));
                        apply();
                    }
                },
                SliderConfig::default(),
            ),
        ),
    ));

    Column(Modifier::new().gap(Dp(4.0))).child((
        Row(Modifier::new().gap(Dp(8.0)).align_items(AlignItems::CENTER)).child({
            // Everything before the sliders, so the first row stays narrow.
            children[..children.len() - 2].to_vec()
        }),
        Row(Modifier::new().gap(Dp(8.0)).align_items(AlignItems::CENTER))
            .child(children[children.len() - 2..].to_vec()),
    ))
}

/// A button that opens a menu of choices.
fn menu(state: Rc<MenuState>, label: String, choices: Vec<(String, Rc<dyn Fn()>)>) -> View {
    let items = choices
        .into_iter()
        .map(|(text, choose)| {
            let dismiss = state.clone();
            DropdownMenuEntry::Item(DropdownMenuItem::new(text, move || {
                choose();
                dismiss.dismiss();
            }))
        })
        .collect();

    DropdownMenu(
        state.clone(),
        Modifier::new(),
        Button(
            Modifier::new(),
            {
                let state = state.clone();
                move || state.open()
            },
            ButtonConfig::default(),
            || Text(label.clone()).size(Sp(12.0)),
        ),
        items,
        DropdownMenuConfig::default(),
    )
}

/// Roughly how long a run has left, from the chunks it has done so far. The
/// first chunk is the slowest and the estimate improves as it goes.
fn remaining(started: Option<Instant>, done: usize, total: usize) -> u64 {
    let (Some(started), true) = (started, done > 0 && total > done) else {
        return 0;
    };

    let elapsed = started.elapsed().as_secs_f64();
    ((elapsed / done as f64) * (total - done) as f64).round() as u64
}

fn parse_groups(text: &str) -> Result<Vec<GroupId>, String> {
    let mut groups = Vec::new();

    for name in text
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
        let group =
            GroupId::from_name(name).ok_or_else(|| format!("unknown instrument '{name}'"))?;
        if !groups.contains(&group) {
            groups.push(group);
        }
    }

    Ok(groups)
}

fn stem(name: &str) -> &str {
    match name.rsplit_once('.') {
        Some((stem, _)) => stem,
        None => name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_estimate_needs_a_started_run_and_some_progress() {
        assert_eq!(remaining(None, 3, 10), 0);
        assert_eq!(remaining(Some(Instant::now()), 0, 10), 0);
        assert_eq!(remaining(Some(Instant::now()), 10, 10), 0);
    }

    #[test]
    fn an_estimate_scales_with_what_is_left() {
        let started = Some(Instant::now() - Duration::from_secs(4));
        let quarter = remaining(started, 1, 4);
        let three_quarters = remaining(started, 3, 4);
        assert!(three_quarters < quarter, "less work left, less time left");
    }

    #[test]
    fn instrument_names_still_parse() {
        assert!(parse_groups("").unwrap().is_empty());
        assert_eq!(
            parse_groups("acoustic_piano, acoustic_bass").unwrap().len(),
            2
        );
        assert_eq!(
            parse_groups("acoustic_piano, acoustic_piano")
                .unwrap()
                .len(),
            1,
            "a repeated name is one group"
        );
        assert!(parse_groups("kazoo").is_err());
    }
}
