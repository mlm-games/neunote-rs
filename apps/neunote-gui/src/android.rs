use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use neunote_types::ModelSize;
use neunote_ui::{LoadedAudio, LoadedWeights, Shell, Transport};

const AUDIO: &[&str] = &["wav", "flac", "mp3", "ogg", "oga", "opus", "aac", "m4a"];

type AudioDone = Rc<dyn Fn(Option<LoadedAudio>)>;
type WeightsDone = Rc<dyn Fn(Option<LoadedWeights>)>;
type SaveDone = Rc<dyn Fn(Result<String, String>)>;

#[derive(Default)]
struct Answers {
    audio: Option<(String, Vec<u8>)>,
    weights: Option<(String, Vec<u8>)>,
    save: Option<Result<String, String>>,
}

#[derive(Default)]
struct Handlers {
    audio: Option<AudioDone>,
    weights: Option<WeightsDone>,
    save: Option<SaveDone>,
}

pub(crate) struct Pending {
    answers: Arc<Mutex<Answers>>,
    handlers: Rc<RefCell<Handlers>>,
    picking: Arc<AtomicBool>,
    saving: Arc<AtomicBool>,
}

/// Android delivers a picker's answer to the Java main thread, which is the
/// thread the view runs on, so a picker cannot be awaited from here. Every pick
/// blocks a worker instead, and `pump` hands the answer over on this thread.
pub(crate) fn shell(pending: &Pending) -> Shell {
    let cached = {
        let cache = neunote_models::cache();
        let mut found: HashMap<ModelSize, PathBuf> = HashMap::new();
        for size in ModelSize::ALL {
            let path = cache.model_path(size);
            if path.is_file() {
                found.insert(size, path);
            }
        }
        found
    };

    let transport = crate::audio::Device::open().map(|device| Rc::new(device) as Rc<dyn Transport>);

    Shell {
        cached_weights: Rc::new(move |size| cached.get(&size).cloned()),

        resolve_weights: Rc::new(|_size, done| {
            done(Err(String::from(
                "no cached checkpoint for that size -- pick a .gguf",
            )));
        }),

        fetch_weights: Rc::new(|size, _accepted, _progress, done| {
            done(Err(format!(
                "fetch the weights outside the app, then pick the .gguf (--size {})",
                size.as_str()
            )));
        }),

        licence_accepted: Rc::new(|| neunote_models::cache().licence_accepted()),

        set_licence_accepted: Rc::new(|accepted| {
            if accepted {
                let _ = neunote_models::cache().accept_licence();
            }
        }),

        pick_audio: {
            let pending = Clone::clone(pending);
            Rc::new(move |done| {
                pending.handlers.borrow_mut().audio = Some(done);
                pending.open(
                    AUDIO.iter().map(|e| (*e).to_owned()).collect(),
                    String::from("Audio"),
                    |answers, picked| answers.audio = Some(picked),
                );
            })
        },

        pick_weights: {
            let pending = Clone::clone(pending);
            Rc::new(move |done| {
                pending.handlers.borrow_mut().weights = Some(done);
                pending.open(
                    vec![String::from("gguf")],
                    String::from("Checkpoint"),
                    |answers, picked| answers.weights = Some(picked),
                );
            })
        },

        save_midi: {
            let pending = Clone::clone(pending);
            Rc::new(move |name, bytes, done| {
                pending.handlers.borrow_mut().save = Some(done);
                let name = name.to_owned();
                let bytes = bytes.to_vec();
                pending.save(name, bytes);
            })
        },

        copy_midi: Rc::new(|_name, _bytes| {
            Err(String::from(
                "the Android clipboard cannot hand a file to a DAW, use Save",
            ))
        }),

        transport,
    }
}

impl Clone for Pending {
    fn clone(&self) -> Self {
        Self {
            answers: Arc::clone(&self.answers),
            handlers: Rc::clone(&self.handlers),
            picking: Arc::clone(&self.picking),
            saving: Arc::clone(&self.saving),
        }
    }
}

impl Pending {
    pub(crate) fn new() -> Self {
        Self {
            answers: Arc::new(Mutex::new(Answers::default())),
            handlers: Rc::new(RefCell::new(Handlers::default())),
            picking: Arc::new(AtomicBool::new(false)),
            saving: Arc::new(AtomicBool::new(false)),
        }
    }

    fn open(
        &self,
        extensions: Vec<String>,
        title: String,
        keep: impl Fn(&mut Answers, (String, Vec<u8>)) + Send + 'static,
    ) {
        if self.picking.swap(true, Ordering::AcqRel) {
            return;
        }

        let answers = Arc::clone(&self.answers);
        let picking = Arc::clone(&self.picking);

        std::thread::spawn(move || {
            let picked = futures_lite::future::block_on(
                rlobkit_dialogs::RlobKit::open_file_picker(rlobkit_dialogs::OpenFileOptions {
                    file_type: rlobkit_dialogs::RlobKitType::Custom {
                        extensions,
                        mime_types: vec![String::from("*/*")],
                    },
                    mode: rlobkit_dialogs::RlobKitMode::Single,
                    title: Some(title),
                    ..Default::default()
                }),
            );

            if let Ok(Some(mut files)) = picked
                && let Some(file) = files.pop()
            {
                let name = file.name().to_owned();
                if let Ok(bytes) = file.read_bytes() {
                    let mut slot = answers.lock().unwrap_or_else(|e| e.into_inner());
                    keep(&mut slot, (name, bytes.to_vec()));
                }
            }

            picking.store(false, Ordering::Release);
            repose_platform::wake_event_loop();
        });
    }

    fn save(&self, name: String, bytes: Vec<u8>) {
        if self.saving.swap(true, Ordering::AcqRel) {
            return;
        }

        let answers = Arc::clone(&self.answers);
        let saving = Arc::clone(&self.saving);

        std::thread::spawn(move || {
            let saved = futures_lite::future::block_on(rlobkit_dialogs::RlobKit::save_bytes(
                rlobkit_dialogs::SaveFileOptions {
                    suggested_name: Some(name),
                    ..Default::default()
                },
                &bytes,
            ));

            let answer = match saved {
                Ok(Some(file)) => Ok(file.name().to_owned()),
                Ok(None) => Err(String::from("save cancelled")),
                Err(error) => Err(error.to_string()),
            };
            answers.lock().unwrap_or_else(|e| e.into_inner()).save = Some(answer);

            saving.store(false, Ordering::Release);
            repose_platform::wake_event_loop();
        });
    }

    /// Hands whatever the workers brought back to the view, on this thread.
    pub(crate) fn pump(&self) {
        let (audio, weights, save) = {
            let mut answers = self.answers.lock().unwrap_or_else(|e| e.into_inner());
            (
                answers.audio.take(),
                answers.weights.take(),
                answers.save.take(),
            )
        };

        if let Some((name, bytes)) = audio {
            let done = self.handlers.borrow_mut().audio.take();
            if let Some(done) = done {
                done(Some(LoadedAudio { name, bytes }));
            }
        }

        if let Some((name, bytes)) = weights {
            let done = self.handlers.borrow_mut().weights.take();
            if let Some(done) = done {
                done(Some(LoadedWeights { name, bytes }));
            }
        }

        if let Some(answer) = save {
            let done = self.handlers.borrow_mut().save.take();
            if let Some(done) = done {
                done(answer);
            }
        }
    }
}
