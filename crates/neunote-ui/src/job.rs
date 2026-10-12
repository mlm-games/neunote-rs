//! The transcription job: one run of the pipeline, off the UI thread where
//! there is one.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use neunote_engine::Model;
use neunote_pipeline::Run;
use neunote_pipeline::muscriptor::Muscriptor;
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
    /// How many chunks the run has, sent before the first one so a caller can
    /// show something before the first (slow) chunk lands.
    Total(usize),
    Progress {
        done: usize,
        total: usize,
        finalized: f64,
    },
    /// Notes a chunk finalized. A preview of the finished list, not a piece of
    /// it: the run replaces this with the real answer.
    Notes(Vec<NoteEvent>),
    Finished(Vec<NoteEvent>),
    Cancelled,
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

        // A browser tab has no second thread to hand this to: `std::thread::spawn`
        // panics on this target. So the web build keeps the engine on the UI
        // thread and does one chunk per frame, which is what leaves the page
        // painting and the cancel button answering.
        #[cfg(target_arch = "wasm32")]
        Stepped::start(
            weights,
            mono,
            size,
            instruments,
            prelude_forcing,
            Arc::clone(&cancel),
            outbox,
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            let out = outbox.clone();
            let cancel = Arc::clone(&cancel);
            std::thread::spawn(move || {
                if let Err(error) = blocking(
                    weights,
                    mono,
                    size,
                    instruments,
                    prelude_forcing,
                    &cancel,
                    &out,
                ) {
                    let _ = out.send(Message::Failed(error));
                }
            });
        }

        Job { inbox, cancel }
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
        Weights::Bytes(bytes) => Model::from_bytes(Arc::clone(bytes), "checkpoint")
            .map(Muscriptor::from_model)
            .map_err(|error| format!("cannot load the checkpoint: {error}")),
    }
}

/// One run, driven to the end on whatever thread owns it.
#[cfg(not(target_arch = "wasm32"))]
fn blocking(
    weights: Weights,
    mono: Arc<Vec<f32>>,
    size: ModelSize,
    instruments: Vec<GroupId>,
    prelude_forcing: bool,
    cancel: &AtomicBool,
    out: &Sender<Message>,
) -> Result<(), String> {
    let mut engine = load_engine(&weights)?;
    let mut run = Run::for_engine(&engine, &mono, size, &instruments, prelude_forcing)?;

    let _ = out.send(Message::Total(run.total()));

    loop {
        if cancel.load(Ordering::SeqCst) {
            let _ = out.send(Message::Cancelled);
            return Ok(());
        }

        let Some(step) = run.step(&mut engine, &mono)? else {
            break;
        };

        let _ = out.send(Message::Progress {
            done: step.progress.chunks_done,
            total: step.progress.chunks_total,
            finalized: step.progress.finalized_through,
        });
        if let Some(notes) = step.notes {
            let _ = out.send(Message::Notes(notes));
        }
    }

    match run.finish() {
        Ok(notes) => {
            let _ = out.send(Message::Finished(notes));
        }
        Err(error) => {
            let _ = out.send(Message::Failed(error));
        }
    }

    Ok(())
}

/// The web build's driver: one chunk per frame, the engine alive across frames.
///
/// Both halves have to outlive the frame that starts the job -- loading the
/// checkpoint and running the first chunk each take long enough to look like a
/// hang -- so they sit in one `Rc` the frame closure owns.
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;

#[cfg(target_arch = "wasm32")]
use neunote_pipeline::Engine;

#[cfg(target_arch = "wasm32")]
use repose_core::timer::{TimerHandle, delay_frames};

#[cfg(target_arch = "wasm32")]
struct Stepped {
    weights: Weights,
    engine: RefCell<Option<Muscriptor>>,
    run: RefCell<Option<Run>>,
    mono: Arc<Vec<f32>>,
    instruments: Vec<GroupId>,
    prelude_forcing: bool,
    out: Sender<Message>,
    cancel: Arc<AtomicBool>,
    /// Held so the scheduled frames can be called at all.
    ticker: RefCell<Option<TimerHandle>>,
}

#[cfg(target_arch = "wasm32")]
impl Stepped {
    fn start(
        weights: Weights,
        mono: Arc<Vec<f32>>,
        size: ModelSize,
        instruments: Vec<GroupId>,
        prelude_forcing: bool,
        cancel: Arc<AtomicBool>,
        out: Sender<Message>,
    ) {
        let state = Rc::new(Stepped {
            weights,
            engine: RefCell::new(None),
            run: RefCell::new(None),
            mono,
            instruments,
            prelude_forcing,
            out,
            cancel,
            ticker: RefCell::new(None),
        });

        let begin = Rc::clone(&state);
        // The first frame only loads the engine and schedules the second, so a
        // slow load does not read as a dead page. The handle is dropped here on
        // purpose: the job outlives any owner of it, so it must not cancel.
        delay_frames(1, move || begin.open_and_schedule(size)).detach();
    }

    fn open_and_schedule(self: &Rc<Self>, size: ModelSize) {
        match load_engine(&self.weights) {
            Ok(engine) => {
                let segment = engine.segment_samples();
                match Run::new(
                    segment,
                    self.mono.len(),
                    size,
                    &self.instruments,
                    self.prelude_forcing,
                ) {
                    Ok(run) => {
                        let _ = self.out.send(Message::Total(run.total()));
                        *self.run.borrow_mut() = Some(run);
                        *self.engine.borrow_mut() = Some(engine);
                        self.schedule_next();
                    }
                    Err(error) => {
                        let _ = self.out.send(Message::Failed(error));
                    }
                }
            }
            Err(error) => {
                let _ = self.out.send(Message::Failed(error));
            }
        }
    }

    fn schedule_next(self: &Rc<Self>) {
        let next = Rc::clone(self);
        *self.ticker.borrow_mut() = Some(delay_frames(1, move || {
            next.tick();
        }));
    }

    fn tick(self: &Rc<Self>) {
        if self.cancel.load(Ordering::SeqCst) {
            let _ = self.out.send(Message::Cancelled);
            self.release();
            return;
        }

        // Scoped so neither RefCell is held while this reschedules itself.
        enum Stepped {
            More,
            Done,
            Failed(String),
        }

        let stepped = {
            let mut engine = self.engine.borrow_mut();
            let Some(engine) = engine.as_mut() else {
                return;
            };

            match self
                .run
                .borrow_mut()
                .as_mut()
                .map(|run| run.step(engine, &self.mono))
            {
                Some(Ok(Some(step))) => {
                    let _ = self.out.send(Message::Progress {
                        done: step.progress.chunks_done,
                        total: step.progress.chunks_total,
                        finalized: step.progress.finalized_through,
                    });
                    if let Some(notes) = step.notes {
                        let _ = self.out.send(Message::Notes(notes));
                    }
                    Stepped::More
                }
                Some(Ok(None)) | None => Stepped::Done,
                Some(Err(error)) => Stepped::Failed(error),
            }
        };

        match stepped {
            Stepped::More => self.schedule_next(),
            Stepped::Done => self.finish(),
            Stepped::Failed(error) => {
                let _ = self.out.send(Message::Failed(error));
                self.release();
            }
        }
    }

    fn finish(&self) {
        let finished = self
            .run
            .borrow_mut()
            .take()
            .map(|run| run.finish())
            .transpose();

        match finished {
            Ok(Some(notes)) => {
                let _ = self.out.send(Message::Finished(notes));
            }
            Ok(None) => {}
            Err(error) => {
                let _ = self.out.send(Message::Failed(error));
            }
        }

        self.release();
    }

    /// Drop the engine. The run is already off the queue, so nothing else holds
    /// it.
    fn release(&self) {
        self.engine.borrow_mut().take();
        self.run.borrow_mut().take();
        self.ticker.borrow_mut().take();
    }
}
