//! The root view: state, the job, and the three surfaces around it -- toolbar,
//! track list and piano roll, status line.

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use neunote_audio::{decode_bytes, to_engine_input};
use neunote_midi::midi_bytes;
use neunote_types::{GroupId, ModelSize, NoteEvent};
use repose_core::RenderContext;
use repose_core::prelude::*;
use repose_core::timer;
use repose_material::material3::{
    ButtonConfig, LinearProgressIndicator, LinearProgressIndicatorConfig, RadioButton,
    RadioButtonConfig, Slider, SliderConfig, Switch, SwitchConfig, TextButton, TextField,
    TextFieldConfig,
};
use repose_ui::*;
use web_time::Duration;

use crate::job::{Job, Message, Weights};
use crate::{piano_roll, tracks};

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

/// What the view needs from its platform. The desktop shell answers with
/// dialogs and the model cache; the web shell with a file input and a download.
pub struct Shell {
    pub pick_audio: Rc<dyn Fn(Picked)>,
    pub pick_weights: Rc<dyn Fn(PickedWeights)>,
    /// A verified checkpoint already on this machine, if there is one.
    pub cached_weights: Rc<dyn Fn(ModelSize) -> Option<PathBuf>>,
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
    Failed,
}

/// The root view function the shells hand to their platform.
pub fn root(shell: Shell) -> impl FnMut(&mut Scheduler, &RenderContext) -> View + 'static {
    move |scheduler, _context| app(&shell, scheduler)
}

fn app(shell: &Shell, _scheduler: &mut Scheduler) -> View {
    let phase = remember(|| signal(Phase::Idle));
    let status = remember(|| signal(String::from("Open an audio file to begin.")));
    let source = remember(|| signal(None::<Rc<Source>>));
    let notes = remember(|| signal(None::<Rc<Vec<NoteEvent>>>));
    let picked_weights = remember(|| signal(None::<Rc<Vec<u8>>>));
    let size = remember(|| signal(ModelSize::DEFAULT));
    let prelude = remember(|| signal(true));
    let instruments = remember(|| signal(String::new()));
    let hidden = remember(|| signal(Rc::new(HashSet::<u16>::new())));
    let zoom = remember(|| signal(24.0f32));
    let selected = remember(|| signal(None::<usize>));

    let job_slot: Rc<RefCell<Option<Job>>> =
        remember_with_key("neunote:job", || RefCell::new(None));

    // The worker reports on a channel; this drains it into the signals.
    scoped_effect_once({
        let job_slot = job_slot.clone();
        let phase = phase.clone();
        let status = status.clone();
        let notes = notes.clone();
        let selected = selected.clone();

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
                                "chunk {done}/{total} · {:.0}s of audio final",
                                finalized
                            ));
                        }
                        Message::Failed(error) => {
                            phase.set(Phase::Failed);
                            status.set(error);
                        }
                        Message::Finished(transcribed) => finished = Some(transcribed),
                    }
                }

                if let Some(transcribed) = finished {
                    let count = transcribed.len();
                    *job_slot.borrow_mut() = None;
                    notes.set(Some(Rc::new(transcribed)));
                    selected.set(None);
                    phase.set(Phase::Done);
                    status.set(format!("{count} notes"));
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
        let phase = phase.clone();
        let status = status.clone();
        let source = source.clone();
        let notes = notes.clone();
        let selected = selected.clone();
        let picker = shell.pick_audio.clone();

        move || {
            // The picker may answer later, so the callback takes its own
            // handles and this closure stays callable more than once.
            let phase = phase.clone();
            let status = status.clone();
            let source = source.clone();
            let notes = notes.clone();
            let selected = selected.clone();

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
                                notes.set(None);
                                selected.set(None);
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
        }
    };

    let on_choose_weights = {
        let status = status.clone();
        let picked_weights = picked_weights.clone();
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
        let phase = phase.clone();
        let status = status.clone();
        let source = source.clone();
        let notes = notes.clone();
        let picked_weights = picked_weights.clone();
        let size = size.clone();
        let prelude = prelude.clone();
        let instruments = instruments.clone();
        let selected = selected.clone();
        let job_slot = job_slot.clone();
        let cached = shell.cached_weights.clone();

        move || {
            let Some(source) = source.get() else {
                status.set("open an audio file first".to_owned());
                return;
            };

            let weights = match cached(size.get()) {
                Some(path) => Weights::Path(path),
                None => {
                    match picked_weights.get() {
                        Some(bytes) => Weights::Bytes(Arc::new(bytes.as_ref().clone())),
                        None => {
                            status.set("no checkpoint: fetch one with `neunote models fetch`, or pick a .gguf".to_owned());
                            return;
                        }
                    }
                }
            };

            let groups = match parse_groups(&instruments.get()) {
                Ok(groups) => groups,
                Err(error) => {
                    status.set(error);
                    return;
                }
            };

            notes.set(None);
            selected.set(None);
            phase.set(Phase::Working { done: 0, total: 0 });
            status.set(format!(
                "transcribing {:.1}s with {}…",
                source.duration,
                weights.label()
            ));

            *job_slot.borrow_mut() = Some(Job::start(
                weights,
                source.samples.clone(),
                size.get(),
                groups,
                prelude.get(),
            ));
        }
    };

    let on_cancel = {
        let status = status.clone();
        let job_slot = job_slot.clone();

        move || {
            if let Some(job) = job_slot.borrow().as_ref() {
                job.cancel();
                status.set("cancelling after the current chunk…".to_owned());
            }
        }
    };

    let on_save = {
        let status = status.clone();
        let notes = notes.clone();
        let source = source.clone();
        let saver = shell.save_midi.clone();

        move || {
            let Some(notes) = notes.get() else {
                status.set("nothing to save yet".to_owned());
                return;
            };
            let name = source
                .get()
                .map(|source| format!("{}.mid", stem(&source.name)))
                .unwrap_or_else(|| String::from("neunote.mid"));

            match midi_bytes(&notes, 120.0) {
                Ok(bytes) => match saver(&name, &bytes) {
                    Ok(written) => {
                        let count = notes.len();
                        status.set(format!("wrote {written} ({count} notes)"));
                    }
                    Err(error) => status.set(format!("cannot write MIDI: {error}")),
                },
                Err(error) => status.set(format!("cannot build MIDI: {error}")),
            }
        }
    };

    // --- chrome ------------------------------------------------------------

    let weights_label = match (shell.cached_weights)(size.get()) {
        Some(path) => format!("Checkpoint: {}", path.display()),
        None if picked_weights.get().is_some() => String::from("Checkpoint: picked file"),
        None => String::from("Checkpoint: none"),
    };

    let size_row = Row(Modifier::new().gap(Dp(2.0))).child(
        ModelSize::ALL
            .into_iter()
            .map(|candidate| {
                let chosen = size.clone();
                Row(Modifier::new().gap(Dp(4.0))).child((
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

    // Two rows: the file and model controls, then the filter and the run
    // controls. One row overflows at the default window width.
    let toolbar = Column(Modifier::new().padding(Dp(8.0)).gap(Dp(6.0))).child(vec![
        Row(Modifier::new().gap(Dp(8.0))).child(vec![
            TextButton(Modifier::new(), on_open, ButtonConfig::default(), || {
                Text("Open audio")
            }),
            TextButton(
                Modifier::new(),
                on_choose_weights,
                ButtonConfig::default(),
                || Text(weights_label.clone()),
            ),
            size_row,
            Spacer(),
            Row(Modifier::new().gap(Dp(4.0))).child((
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
        Row(Modifier::new().gap(Dp(8.0))).child(vec![
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
            TextButton(
                Modifier::new(),
                on_transcribe,
                ButtonConfig::default(),
                || Text("Transcribe"),
            ),
            TextButton(Modifier::new(), on_cancel, ButtonConfig::default(), || {
                Text("Cancel")
            }),
            TextButton(Modifier::new(), on_save, ButtonConfig::default(), || {
                Text("Save MIDI")
            }),
        ]),
    ]);

    // --- body --------------------------------------------------------------

    let track_panel = match notes.get() {
        Some(notes) => tracks::view(tracks::rows(&notes), (*hidden).clone()),
        None => Box(Modifier::new().width(Dp(260.0)).fill_max_height()),
    };

    let roll = match notes.get() {
        Some(notes) => piano_roll::view(notes, hidden.get(), zoom.get(), (*selected).clone()),
        None => Box(Modifier::new().fill_max_size().padding(Dp(24.0))).child(
            Text("Transcribe to see the notes.")
                .size(Sp(14.0))
                .color(theme().on_surface_variant),
        ),
    };

    let selection = match (selected.get(), notes.get()) {
        (Some(index), Some(notes)) => match notes.get(index) {
            Some(note) => format!(
                "{} · pitch {} · {:.2}s → {:.2}s",
                neunote_types::instrument_label(note.program),
                note.pitch,
                note.onset,
                note.offset
            ),
            None => String::new(),
        },
        _ => String::new(),
    };

    let (fraction, progress_label) = match phase.get() {
        Phase::Working { done, total } if total > 0 => {
            (Some(done as f32 / total as f32), format!("{done}/{total}"))
        }
        Phase::Working { .. } => (None, String::from("working…")),
        Phase::Done => (Some(1.0), String::from("done")),
        _ => (Some(0.0), String::new()),
    };

    let footer = Row(Modifier::new().padding(Dp(8.0)).gap(Dp(12.0))).child((
        Box(Modifier::new().width(Dp(200.0))).child(LinearProgressIndicator(
            fraction,
            LinearProgressIndicatorConfig::default(),
        )),
        Text(format!("{}  {progress_label}", status.get())).size(Sp(13.0)),
        Spacer(),
        Text(selection)
            .size(Sp(12.0))
            .color(theme().on_surface_variant),
        Slider(
            zoom.get(),
            (4.0, 160.0),
            Some(1.0),
            {
                let zoom = zoom.clone();
                move |value| zoom.set(value)
            },
            SliderConfig::default(),
        ),
    ));

    Column(Modifier::new().fill_max_size()).child((
        toolbar,
        Row(Modifier::new().fill_max_size()).child((track_panel, roll)),
        footer,
    ))
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
