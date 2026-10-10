//! The root view: state, the job, and the surfaces around it -- toolbar,
//! track list and piano roll, status line.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use neunote_audio::{compute_peaks, decode_bytes, to_engine_input};
use neunote_midi::midi_bytes;
use neunote_types::{ModelSize, NoteEvent};
use repose_core::prelude::*;
use repose_core::shortcuts::ShortcutMap;
use repose_core::{RenderContext, shortcuts, timer};
use repose_material::material3::{
    Button, ButtonConfig, DropdownMenu, DropdownMenuConfig, DropdownMenuEntry, DropdownMenuItem,
    FilledTonalButton, IconButton, IconButtonConfig, LinearProgressIndicator,
    LinearProgressIndicatorConfig, MenuState, SegmentConfig, SegmentedButton,
    SegmentedButtonConfig, Slider, SliderConfig, Switch, SwitchConfig, TextButton, TooltipBox,
    TooltipConfig, TooltipState,
};
use repose_material::{Icon, Symbol, material_symbols};
use repose_ui::*;
use web_time::{Duration, Instant};

use crate::edit::Editor;
use crate::instruments;
use crate::job::{Job, Message, Weights};
use crate::quantize::{Division, NOTE_NAMES, Quantize, Scale, Snap};
use crate::roll::{self, Viewport};
use crate::waveform;
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
    UNDO: '\u{e166}',
    REDO: '\u{e15a}',
    FIT: '\u{E5D0}',
    DARK: '\u{e51c}',
    LIGHT: '\u{e518}',
    MORE: '\u{e5d4}',
    SAVE: '\u{e161}',
    PLAY: '\u{e037}',
    PAUSE: '\u{e034}',
    STOP: '\u{e5cd}',
    WAVE: '\u{E1B8}',
    MUTE: '\u{e04f}',
    SOLO: '\u{E050}',
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
type Saved = Rc<dyn Fn(Result<String, String>)>;
type Saver = Rc<dyn Fn(&str, &[u8], Saved)>;
type Copier = Rc<dyn Fn(&str, &[u8]) -> Result<String, String>>;
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
    /// Put the MIDI where another program can take it from: the system
    /// clipboard, under the MIME types a DAW looks for. A browser's clipboard
    /// cannot hand a file to a desktop program, so the web host says so.
    pub copy_midi: Copier,
    /// Playback, where the host has a device to play through. A browser has
    /// none, and the view offers no transport without it.
    pub transport: Option<Rc<dyn Transport>>,
}

/// What a host that can make sound answers with.
pub trait Transport: Send + Sync + 'static {
    fn play(&self, samples: Arc<Vec<f32>>, notes: Arc<Vec<NoteEvent>>, duration: f64, mix: Mix);
    fn set_mix(&self, mix: Mix);
    fn pause(&self);
    fn resume(&self);
    /// Move the sound to `seconds` into the file.
    fn seek(&self, seconds: f64);
    fn stop(&self);
    fn playing(&self) -> bool;
    /// Seconds into the file the sound has reached.
    fn position(&self) -> f64;
}

/// What is playing, and which instruments are in it.
#[derive(Clone)]
pub struct Mix {
    pub mode: Option<Mode>,
    pub volume: f32,
    /// By General MIDI program.
    pub tracks: Vec<(u16, TrackMix)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Audio,
    Notes,
}

#[derive(Clone, Copy)]
pub struct TrackMix {
    pub gain: f32,
    pub muted: bool,
    pub solo: bool,
}

impl Default for TrackMix {
    fn default() -> Self {
        Self {
            gain: 1.0,
            muted: false,
            solo: false,
        }
    }
}

pub(crate) struct Source {
    name: String,
    samples: Arc<Vec<f32>>,
    pub(crate) duration: f64,
    pub(crate) peaks: Arc<Vec<(f32, f32)>>,
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
    let instruments = remember(|| signal(Vec::new()));
    let hidden = remember(|| signal(Rc::new(HashSet::<u16>::new())));
    let licence = remember(|| signal((shell.licence_accepted)()));
    let viewport = remember(|| signal(Viewport::default()));
    let strip_size = remember(|| signal(Vec2 { x: 0.0, y: 0.0 }));
    let roll_size = remember_with_key("neunote:roll-size", || signal(Vec2 { x: 0.0, y: 0.0 }));
    let quantize = remember(|| signal(Quantize::default()));
    let quantising = remember(|| signal(false));
    let head = remember(|| signal(0.0f64));
    let playing = remember(|| signal(false));
    let mode = remember(|| signal(Mode::Audio));
    let mix = remember(|| signal(Rc::new(HashMap::<u16, TrackMix>::new())));
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

    // A handle of its own, so the picker's callback can own one without
    // taking the signal away from the transport below.
    let head_reset = head.clone();
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
            let head = head_reset.clone();

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
                                let peaks = Arc::new(compute_peaks(&mono, 1600));
                                source.set(Some(Rc::new(Source {
                                    name: loaded.name,
                                    samples: Arc::new(mono),
                                    duration: seconds,
                                    peaks,
                                })));
                                head.set(0.0);
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

            let groups = instruments.get();

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

    // --- playback ---------------------------------------------------------
    // Nothing to play without a host that can make sound, and nothing to play
    // without the recording to play.

    let notes_for_play = editor.notes();
    let mix_now: Rc<dyn Fn() -> Mix> = Rc::new({
        let mode = (*mode).clone();
        let mix = (*mix).clone();
        let shown = hidden.clone();
        let source = (*source).clone();
        let tracks = tracks::rows(&notes_for_play);

        move || Mix {
            mode: (mode.get() == Mode::Audio || source.get().is_some()).then_some(mode.get()),
            volume: 0.9,
            tracks: tracks
                .iter()
                .map(|(program, _, _, _)| {
                    let track = (*mix.get()).get(program).copied().unwrap_or_default();
                    let shown = !shown.get().contains(program);
                    let mut track = track;
                    if !shown {
                        track.muted = true;
                    }
                    (*program, track)
                })
                .collect(),
        }
    });

    let on_play = {
        let source = (*source).clone();
        let transport = shell.transport.clone();
        let notes = Arc::new(notes_for_play.clone());
        let mix_now = mix_now.clone();
        let playing = (*playing).clone();
        let status = (*status).clone();

        Rc::new(move || {
            let Some(transport) = transport.as_ref() else {
                status.set("this build cannot make sound".to_owned());
                return;
            };
            let Some(source) = source.get() else {
                status.set("open an audio file first".to_owned());
                return;
            };

            if playing.get() {
                transport.pause();
                playing.set(false);
                return;
            }

            if transport.playing() {
                transport.resume();
            } else {
                transport.play(
                    Arc::clone(&source.samples),
                    Arc::clone(&notes),
                    source.duration,
                    mix_now(),
                );
            }
            playing.set(true);
        })
    };

    let on_stop = {
        let transport = shell.transport.clone();
        let playing = (*playing).clone();
        let head = (*head).clone();

        Rc::new(move || {
            if let Some(transport) = transport.as_ref() {
                transport.stop();
            }
            playing.set(false);
            head.set(0.0);
        })
    };

    let on_seek = {
        let transport = shell.transport.clone();
        let head = (*head).clone();
        let source = (*source).clone();

        Rc::new(move |fraction: f64| {
            let Some(transport) = transport.as_ref() else {
                return;
            };
            let Some(source) = source.get() else {
                return;
            };

            let seconds = fraction * source.duration;
            head.set(seconds);
            transport.seek(seconds);
        })
    };

    let on_mode = {
        let mode = (*mode).clone();
        let transport = shell.transport.clone();
        let mix_now = mix_now.clone();
        let playing = (*playing).clone();

        Rc::new(move |wanted: Mode| {
            mode.set(wanted);
            playing.set(true);
            if let Some(transport) = transport.as_ref() {
                transport.resume();
                transport.set_mix(mix_now());
            }
        })
    };

    // The host keeps the clock: the view only reads where the sound has got to.
    scoped_effect_once({
        let transport = shell.transport.clone();
        let head = (*head).clone();
        let playing = (*playing).clone();

        move || {
            let handle = transport.as_ref().map(|transport| {
                let transport = Rc::clone(transport);
                timer::interval(Duration::from_millis(60), move || {
                    if playing.get() {
                        head.set(transport.position());
                    }
                })
            });

            Dispose::new(move || {
                if let Some(handle) = handle {
                    handle.cancel();
                }
            })
        }
    });

    let publish: Rc<dyn Fn()> = {
        let transport = shell.transport.clone();
        let mix_now = mix_now.clone();
        Rc::new(move || {
            if let Some(transport) = transport.as_ref() {
                transport.set_mix(mix_now());
            }
        })
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
                Ok(bytes) => {
                    let count = notes.len();
                    let status = status.clone();
                    status.set(format!("saving {name}…"));
                    saver(
                        &name,
                        &bytes,
                        Rc::new(move |result| match result {
                            Ok(written) => status.set(format!("wrote {written} ({count} notes)")),
                            Err(error) => status.set(format!("cannot write MIDI: {error}")),
                        }),
                    );
                }
                Err(error) => status.set(format!("cannot build MIDI: {error}")),
            }
        })
    };

    let on_copy = {
        let source = (*source).clone();
        let bpm = quantize.clone();
        let editor = editor.clone();
        let copy = shell.copy_midi.clone();
        let status = (*status).clone();

        Rc::new(move || {
            let notes = editor.notes();
            if notes.is_empty() {
                status.set("nothing to copy yet".to_owned());
                return;
            }

            let name = source
                .get()
                .map(|source| format!("{}.mid", stem(&source.name)))
                .unwrap_or_else(|| String::from("neunote.mid"));

            match midi_bytes(&notes, bpm.get().bpm) {
                Ok(bytes) => status.set(match copy(&name, &bytes) {
                    Ok(answer) => answer,
                    Err(error) => error,
                }),
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

    let on_toggle_theme = Rc::new({
        let dark = dark.clone();
        move || dark.update(|dark| *dark = !*dark)
    });

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

    // The file's name, not its path: the button says what is loaded, and a
    // 75-character home directory does not fit a toolbar.
    let weights_label = match (shell.cached_weights)(size.get()) {
        Some(path) => path
            .file_name()
            .map(|name| name.to_string_lossy())
            .map(|name| {
                name.strip_prefix("muscriptor-")
                    .unwrap_or(&name)
                    .trim_end_matches(".gguf")
                    .to_owned()
            })
            .unwrap_or_else(|| path.display().to_string()),
        None if picked_weights.get().is_some() => String::from("a picked file"),
        None => String::from("no checkpoint"),
    };

    let overflow = remember(MenuState::new);
    // The three checkpoints as one control: Small, Medium, Large.
    let model = SegmentedButton(
        &[ModelSize::ALL
            .into_iter()
            .position(|candidate| candidate == size.get())
            .unwrap_or(0)],
        ModelSize::ALL
            .into_iter()
            .map(|candidate| SegmentConfig {
                label: candidate.display_name().into(),
                icon: None,
                on_click: {
                    let size = (*size).clone();
                    Rc::new(move || size.set(candidate))
                },
                enabled: true,
                ..Default::default()
            })
            .collect(),
        SegmentedButtonConfig::default(),
    );

    let overflow_menu = menu(
        overflow.clone(),
        IconButton(
            Icon(Symbols::MORE).size(Sp(19.0)),
            {
                let overflow = overflow.clone();
                move || overflow.open()
            },
            IconButtonConfig::default(),
        ),
        vec![
            (
                "Cancel the run".to_owned(),
                Rc::new({
                    let cancel = on_cancel.clone();
                    move || cancel()
                }) as Rc<dyn Fn()>,
            ),
            (
                "Copy MIDI for a DAW".to_owned(),
                Rc::new({
                    let copy = on_copy.clone();
                    move || copy()
                }) as Rc<dyn Fn()>,
            ),
            (
                "Download weights".to_owned(),
                Rc::new({
                    let download = on_download.clone();
                    move || download()
                }) as Rc<dyn Fn()>,
            ),
            (
                format!(
                    "{} the CC BY-NC weights licence",
                    if licence.get() {
                        "✓ accepted"
                    } else {
                        "Accept"
                    }
                ),
                Rc::new({
                    let licence = (*licence).clone();
                    let record = shell.set_licence_accepted.clone();
                    move || {
                        let accepted = !licence.get();
                        licence.set(accepted);
                        record(accepted);
                    }
                }) as Rc<dyn Fn()>,
            ),
            (
                format!("{} force ties", if prelude.get() { "✓" } else { "" }),
                Rc::new({
                    let prelude = prelude.clone();
                    move || prelude.update(|on| *on = !*on)
                }) as Rc<dyn Fn()>,
            ),
            (
                format!("{} theme", if dark.get() { "Light" } else { "Dark" }),
                Rc::new({
                    let toggle = on_toggle_theme.clone();
                    move || toggle()
                }) as Rc<dyn Fn()>,
            ),
        ],
    );

    let toolbar = Row(Modifier::new()
        .padding(Dp(8.0))
        .gap(Dp(6.0))
        .align_items(AlignItems::CENTER))
    .child(vec![
        icon_button(
            Symbols::FOLDER,
            "Open an audio file",
            true,
            click(on_open.clone()),
        ),
        TextButton(
            Modifier::new(),
            on_choose_weights,
            ButtonConfig::default(),
            || with_icon(Symbols::INBOX, weights_label.clone()),
        ),
        model,
        Button(
            Modifier::new(),
            on_transcribe,
            ButtonConfig::default(),
            || with_icon(Symbols::MUSIC_NOTE, "Transcribe"),
        ),
        FilledTonalButton(
            Modifier::new(),
            click(on_save.clone()),
            ButtonConfig::default(),
            || with_icon(Symbols::SAVE, "Save MIDI"),
        ),
        Spacer(),
        icon_button(Symbols::UNDO, "Undo", editor.can_undo(), {
            let undo = on_undo.clone();
            move || undo()
        }),
        icon_button(Symbols::REDO, "Redo", editor.can_redo(), {
            let redo = on_redo.clone();
            move || redo()
        }),
        icon_button(Symbols::FIT, "Fit the notes", true, {
            let fit = on_fit.clone();
            move || fit()
        }),
        overflow_menu,
    ]);

    // --- body --------------------------------------------------------------

    let shown_notes = editor.notes();

    // The rail is the inspector: the instruments to transcribe until there are
    // notes, the tracks the run produced once there are, and the quantise
    // settings under both. None of that was ever worth a toolbar row.
    let list = if shown_notes.is_empty() {
        instruments::view((*instruments).clone())
    } else {
        tracks::view(
            tracks::rows(&shown_notes),
            (*hidden).clone(),
            (*mix).clone(),
            publish.clone(),
        )
    };

    let inspector = quantise_panel(&quantising, &quantize, &apply_quantize);
    let rail = Column(
        Modifier::new()
            .width(Dp(268.0))
            .fill_max_height()
            .background(theme().surface_container_low),
    )
    .child((
        Box(Modifier::new().fill_max_height().flex_grow(1.0)).child(list),
        inspector,
    ));
    // Hearing what the model made of the recording, next to the recording.
    let transport_bar = Row(Modifier::new()
        .fill_max_width()
        .padding(Dp(6.0))
        .gap(Dp(6.0))
        .align_items(AlignItems::CENTER))
    .child(vec![
        if playing.get() {
            Button(
                Modifier::new(),
                click(on_play.clone()),
                ButtonConfig::default(),
                || with_icon(Symbols::PAUSE, "Pause"),
            )
        } else {
            Button(
                Modifier::new(),
                click(on_play.clone()),
                ButtonConfig::default(),
                || with_icon(Symbols::PLAY, "Play"),
            )
        },
        icon_button(
            Symbols::STOP,
            "Back to the start",
            true,
            click(on_stop.clone()),
        ),
        Row(Modifier::new().gap(Dp(4.0)).align_items(AlignItems::CENTER)).child((
            Text("play")
                .size(Sp(11.0))
                .color(theme().on_surface_variant),
            SegmentedButton(
                &[if mode.get() == Mode::Audio { 0 } else { 1 }],
                [
                    ("the recording".to_owned(), Mode::Audio),
                    ("the notes".to_owned(), Mode::Notes),
                ]
                .into_iter()
                .enumerate()
                .map(|(index, (label, wanted))| SegmentConfig {
                    label,
                    icon: None,
                    on_click: {
                        let on_mode = Rc::clone(&on_mode);
                        Rc::new(move || on_mode(wanted))
                    },
                    enabled: index == (if mode.get() == Mode::Audio { 0 } else { 1 }),
                    ..Default::default()
                })
                .collect(),
                SegmentedButtonConfig::default(),
            ),
        )),
        Spacer(),
        Text(format!(
            "{} / {}",
            clock(head.get()),
            clock(source.get().as_ref().map_or(0.0, |source| source.duration))
        ))
        .size(Sp(12.0))
        .color(theme().on_surface_variant),
    ]);

    let stage = Column(Modifier::new().fill_max_size()).child((
        waveform::view(source.get(), head.get(), (*strip_size).clone(), {
            let on_seek = on_seek.clone();
            move |fraction| on_seek(fraction)
        }),
        transport_bar,
        piano_roll::view(
            editor.clone(),
            (*hidden).get(),
            (*viewport).clone(),
            (*roll_size).clone(),
        ),
    ));

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
            "width",
            sized(
                Dp(150.0),
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
                Dp(120.0),
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
        Row(Modifier::new().fill_max_size()).child((rail, stage)),
        footer,
    ))
}

/// Give a material widget a width of its own.
///
/// Not `View::modifier`: that replaces the widget's modifier, and a widget's
/// painter, size and focus all live on it -- a slider re-modified this way draws
/// nothing at all.
pub(crate) fn sized(width: Dp, control: View) -> View {
    Box(Modifier::new().width(width).align_self_center()).child(control)
}

/// Hand a shared action to a widget that wants a plain closure.
fn click(action: Rc<dyn Fn()>) -> impl Fn() + 'static {
    move || action()
}

/// A quiet square button for an action its glyph already says, with a tooltip
/// for what it says.
pub(crate) fn icon_button(
    symbol: Symbol,
    label: &'static str,
    enabled: bool,
    on_click: impl Fn() + 'static,
) -> View {
    let state = remember(TooltipState::new);
    TooltipBox(
        label,
        state,
        Modifier::new(),
        IconButton(
            Icon(symbol).size(Sp(19.0)),
            on_click,
            IconButtonConfig {
                enabled,
                container_size: Some(Dp(34.0)),
                ..Default::default()
            },
        ),
        TooltipConfig::default(),
    )
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
/// The quantise panel, now the rail's lower half. Off means the roll shows
/// exactly what the model produced; on means the roll shows the raw notes put
/// through these settings, and nothing about the raw list has moved.
fn quantise_panel(on: &Signal<bool>, params: &Signal<Quantize>, apply: &Rc<dyn Fn()>) -> View {
    if !on.get() {
        return Row(Modifier::new()
            .padding(Dp(10.0))
            .gap(Dp(6.0))
            .align_items(AlignItems::CENTER))
        .child((
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
            Spacer(),
            Text("off").size(Sp(11.0)).color(theme().on_surface_variant),
        ));
    }

    let current = params.get();
    let toggle = |label: &str, value: bool, write: Rc<dyn Fn(bool)>| {
        Row(Modifier::new().gap(Dp(4.0)).align_items(AlignItems::CENTER)).child((
            Switch(
                value,
                {
                    let apply = Rc::clone(apply);
                    move |on| {
                        write(on);
                        apply();
                    }
                },
                SwitchConfig::default(),
            ),
            Text(label.to_owned()).size(Sp(12.0)),
        ))
    };

    let root_state = remember(MenuState::new);
    let scale_state = remember(MenuState::new);
    let snap_state = remember(MenuState::new);
    let division_state = remember(MenuState::new);

    Column(Modifier::new().padding(Dp(10.0)).gap(Dp(8.0))).child(vec![
        Row(Modifier::new().gap(Dp(6.0)).align_items(AlignItems::CENTER)).child((
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
        toggle(
            "pitches into a scale",
            current.pitches,
            Rc::new({
                let params = params.clone();
                move |value| params.update(|params| params.pitches = value)
            }),
        ),
        rail_field(
            "root",
            menu(
                root_state.clone(),
                menu_anchor(&root_state, NOTE_NAMES[current.root as usize].to_owned()),
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
        ),
        rail_field(
            "scale",
            menu(
                scale_state.clone(),
                menu_anchor(&scale_state, current.scale.name().to_owned()),
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
        ),
        rail_field(
            "out-of-scale notes",
            menu(
                snap_state.clone(),
                menu_anchor(&snap_state, current.snap.name().to_owned()),
                Snap::ALL
                    .iter()
                    .map(|snap| {
                        let params = params.clone();
                        let apply = apply.clone();
                        (
                            snap.name().to_owned(),
                            Rc::new(move || {
                                params.update(|params| params.snap = *snap);
                                apply();
                            }) as Rc<dyn Fn()>,
                        )
                    })
                    .collect(),
            ),
        ),
        toggle(
            "onsets onto a grid",
            current.times,
            Rc::new({
                let params = params.clone();
                move |value| params.update(|params| params.times = value)
            }),
        ),
        rail_field(
            "grid",
            menu(
                division_state.clone(),
                menu_anchor(&division_state, current.division.label().to_owned()),
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
            ),
        ),
        rail_field(
            &format!("tempo · {} bpm", current.bpm.round() as i32),
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
        rail_field(
            &format!("strength · {}%", (current.strength * 100.0).round() as i32),
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
    ])
}

/// A button that opens a menu of choices.
fn menu(state: Rc<MenuState>, anchor: View, choices: Vec<(String, Rc<dyn Fn()>)>) -> View {
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
        anchor,
        items,
        DropdownMenuConfig::default(),
    )
}

/// The labelled button a menu is anchored to.
fn menu_anchor(state: &Rc<MenuState>, label: String) -> View {
    let open = state.clone();
    Button(
        Modifier::new(),
        move || open.open(),
        ButtonConfig::default(),
        || Text(label).size(Sp(12.0)),
    )
}

/// A control with a caption above it, the width of the rail.
fn rail_field(caption: &str, control: View) -> View {
    Column(Modifier::new().gap(Dp(2.0))).child((
        Text(caption.to_owned())
            .size(Sp(11.0))
            .color(theme().on_surface_variant),
        control,
    ))
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

/// Seconds as minutes and seconds, the way a recording's duration reads.
fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    format!("{}:{:02}", total / 60, total % 60)
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
}
