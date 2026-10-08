//! The transcription job: one run of the pipeline, off the UI thread where
//! there is one.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};

use neunote_engine::Model;
use neunote_pipeline::muscriptor::Muscriptor;
use neunote_pipeline::{Outcome, transcribe_with};
use neunote_types::{GroupId, ModelSize, NoteEvent};

/// Where the checkpoint comes from. A native host has a path and lets the
/// engine read it; a browser host holds the bytes.
pub(crate) enum Weights {
    Path(PathBuf),
    Bytes(Arc<Vec<u8>>),
}

impl Weights {
    pub(crate) fn label(&self) -> String {
        match self {
            Weights::Path(path) => path.display().to_string(),
            Weights::Bytes(_) => "selected checkpoint".to_owned(),
        }
    }
}

pub(crate) enum Message {
    Progress {
        done: usize,
        total: usize,
        finalized: f64,
    },
    Finished(Vec<NoteEvent>),
    Failed(String),
}

pub(crate) struct Job {
    inbox: Receiver<Message>,
    cancel: Arc<AtomicBool>,
}

impl Job {
    pub(crate) fn start(
        weights: Weights,
        mono: Arc<Vec<f32>>,
        size: ModelSize,
        instruments: Vec<GroupId>,
        prelude_forcing: bool,
    ) -> Self {
        let (outbox, inbox) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let handle = cancel.clone();

        let run = move || {
            let mut engine = match load_engine(&weights) {
                Ok(engine) => engine,
                Err(error) => {
                    let _ = outbox.send(Message::Failed(error));
                    return;
                }
            };

            let progress_out = outbox.clone();
            let result = transcribe_with(
                &mut engine,
                &mono,
                size,
                &instruments,
                prelude_forcing,
                &cancel,
                move |progress| {
                    let _ = progress_out.send(Message::Progress {
                        done: progress.chunks_done,
                        total: progress.chunks_total,
                        finalized: progress.finalized_through,
                    });
                },
            );

            let message = match result {
                Ok(Outcome::Finished(notes)) => Message::Finished(notes),
                Ok(Outcome::Cancelled) => Message::Failed("cancelled".to_owned()),
                Err(error) => Message::Failed(error),
            };
            let _ = outbox.send(message);
        };

        // A browser tab has no second thread to hand this to: wasm32 only grows
        // threads with atomics and shared memory, and a blocking wait on one
        // starves the event loop that would pump it. So the web build runs the
        // job inline -- the page is unresponsive until the notes are ready.
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(run);
        #[cfg(target_arch = "wasm32")]
        run();

        Job {
            inbox,
            cancel: handle,
        }
    }

    pub(crate) fn drain(&self) -> Vec<Message> {
        self.inbox.try_iter().collect()
    }

    pub(crate) fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
}

fn load_engine(weights: &Weights) -> Result<Muscriptor, String> {
    match weights {
        Weights::Path(path) => Muscriptor::load(path),
        Weights::Bytes(bytes) => Model::from_bytes(bytes.as_ref().clone(), "checkpoint")
            .map(Muscriptor::from_model)
            .map_err(|error| format!("cannot load the checkpoint: {error}")),
    }
}
