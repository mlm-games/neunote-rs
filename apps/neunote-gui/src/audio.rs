//! Audio out, for the desktop host only: playback of the recording and of what
//! the transcription of it says.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use neunote_types::NoteEvent;

use neunote_ui::{Mix, Mode, Transport};

/// The recording is 16 kHz; the device is not.
const RECORDING_RATE: f64 = 16_000.0;

struct Voice {
    offset: f64,
    frequency: f32,
    program: u16,
}

struct Job {
    samples: Arc<Vec<f32>>,
    mix: Mix,
    duration: f64,
    cursor: f64,
    pending: Vec<NoteEvent>,
    voices: Vec<Voice>,
}

struct Shared {
    job: Mutex<Option<Job>>,
    position: AtomicU64,
    playing: AtomicBool,
}

pub struct Device {
    shared: Arc<Shared>,
    _stream: Stream,
}

/// The wasm host's stream holds a `web_sys::AudioContext`, which is neither
/// Send nor Sync, and `Transport` requires both. It is only ever touched by
/// the thread that opened it, so the wrapper is sound. The stream itself is
/// held for its Drop: dropping it closes the device.
#[allow(dead_code)]
struct Stream(cpal::Stream);

#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
unsafe impl Send for Stream {}

#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
unsafe impl Sync for Stream {}

impl Device {
    /// Open the default output device, if there is one.
    pub fn open() -> Option<Self> {
        // The browser only offers the audioworklet host, and `default_host` is
        // not guaranteed to pick it.
        #[cfg(target_arch = "wasm32")]
        let host = cpal::available_hosts()
            .iter()
            .find(|id| **id == cpal::HostId::AudioWorklet)
            .and_then(|id| cpal::host_from_id(*id).ok())
            .unwrap_or_else(cpal::default_host);

        #[cfg(not(target_arch = "wasm32"))]
        let host = cpal::default_host();

        let device = host.default_output_device()?;
        let supported = device.default_output_config().ok()?;
        let channels = usize::from(supported.channels());
        let rate = f64::from(supported.sample_rate());
        let config = supported.config();

        let shared = Arc::new(Shared {
            job: Mutex::new(None),
            position: AtomicU64::new(0.0f64.to_bits()),
            playing: AtomicBool::new(false),
        });

        let stream = match supported.sample_format() {
            SampleFormat::F32 => build::<f32>(&device, config, channels, rate, Arc::clone(&shared)),
            SampleFormat::F64 => build::<f64>(&device, config, channels, rate, Arc::clone(&shared)),
            SampleFormat::I16 => build::<i16>(&device, config, channels, rate, Arc::clone(&shared)),
            SampleFormat::U16 => build::<u16>(&device, config, channels, rate, Arc::clone(&shared)),
            SampleFormat::I32 => build::<i32>(&device, config, channels, rate, Arc::clone(&shared)),
            SampleFormat::I64 => build::<i64>(&device, config, channels, rate, Arc::clone(&shared)),
            SampleFormat::U64 => build::<u64>(&device, config, channels, rate, Arc::clone(&shared)),
            _ => return None,
        }
        .ok()?;

        // cpal hands the stream back paused; without this the callback never runs
        // and play() writes into a stream nobody is reading.
        stream.play().ok()?;

        Some(Self {
            shared,
            _stream: Stream(stream),
        })
    }
}

impl Transport for Device {
    fn play(&self, samples: Arc<Vec<f32>>, notes: Arc<Vec<NoteEvent>>, duration: f64, mix: Mix) {
        let mut pending = (*notes).clone();
        pending.sort_by(|a, b| a.onset.total_cmp(&b.onset));

        *self.shared.job.lock().unwrap() = Some(Job {
            samples,
            mix,
            duration,
            cursor: 0.0,
            pending,
            voices: Vec::new(),
        });

        self.shared
            .position
            .store(0.0f64.to_bits(), Ordering::Relaxed);
        self.shared.playing.store(true, Ordering::Relaxed);
    }

    fn set_mix(&self, mix: Mix) {
        if let Some(job) = self.shared.job.lock().unwrap().as_mut() {
            job.mix = mix;
        }
    }

    fn pause(&self) {
        self.shared.playing.store(false, Ordering::Relaxed);
    }

    fn resume(&self) {
        if self.shared.job.lock().unwrap().is_some() {
            self.shared.playing.store(true, Ordering::Relaxed);
        }
    }

    fn stop(&self) {
        self.shared.playing.store(false, Ordering::Relaxed);
        self.shared.job.lock().unwrap().take();
        self.shared
            .position
            .store(0.0f64.to_bits(), Ordering::Relaxed);
    }

    fn playing(&self) -> bool {
        self.shared.playing.load(Ordering::Relaxed)
    }

    fn seek(&self, seconds: f64) {
        let mut guard = self.shared.job.lock().unwrap();
        let Some(job) = guard.as_mut() else {
            return;
        };

        let at = seconds.clamp(0.0, job.duration.max(0.0));
        job.cursor = at;
        job.voices.clear();
        job.pending.retain(|note| note.offset > at);
        self.shared.position.store(at.to_bits(), Ordering::Relaxed);
    }

    fn position(&self) -> f64 {
        f64::from_bits(self.shared.position.load(Ordering::Relaxed))
    }
}

fn build<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    channels: usize,
    rate: f64,
    shared: Arc<Shared>,
) -> Result<cpal::Stream, cpal::Error>
where
    T: FromSample<f32> + SizedSample,
{
    device.build_output_stream(
        config,
        move |out: &mut [T], _| render(out, channels, rate, &shared),
        |_| {},
        None,
    )
}

fn render<T: FromSample<f32>>(out: &mut [T], channels: usize, rate: f64, shared: &Shared) {
    let silence = |out: &mut [T]| {
        for sample in out.iter_mut() {
            *sample = T::from_sample_(0.0);
        }
    };

    let mut guard = shared.job.lock().unwrap();
    let Some(job) = guard.as_mut() else {
        silence(out);
        return;
    };

    if !shared.playing.load(Ordering::Relaxed) {
        silence(out);
        return;
    }

    let step = 1.0 / rate;
    let solo = job.mix.tracks.iter().any(|(_, track)| track.solo);

    for frame in out.chunks_mut(channels) {
        let value = match job.mix.mode {
            Some(Mode::Audio) => source_sample(job, job.cursor),
            Some(Mode::Notes) => note_sample(job, job.cursor, solo) * job.mix.volume,
            None => 0.0,
        };

        let value = if value.is_finite() {
            value.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        for channel in frame.iter_mut() {
            *channel = T::from_sample_(value);
        }

        job.cursor += step;
        if job.cursor >= job.duration {
            job.cursor = 0.0;
            job.voices.clear();
            job.pending.sort_by(|a, b| a.onset.total_cmp(&b.onset));
        }
    }

    shared
        .position
        .store(job.cursor.to_bits(), Ordering::Relaxed);
}

fn source_sample(job: &Job, at: f64) -> f32 {
    let exact = at * RECORDING_RATE;
    let index = exact as usize;
    if index >= job.samples.len() {
        return 0.0;
    }

    // The recording is 16 kHz and the device is not. Holding each sample until
    // the next arrives is a staircase, but it is the honest one and every step
    // is a single sample long.
    let a = job.samples[index];
    let b = job.samples.get(index + 1).copied().unwrap_or(a);
    a + (b - a) * (exact - index as f64) as f32
}

fn note_sample(job: &mut Job, at: f64, solo: bool) -> f32 {
    while job
        .pending
        .first()
        .is_some_and(|note| note.onset <= at && at < note.offset)
    {
        let note = job.pending.remove(0);
        job.voices.push(Voice {
            offset: note.offset,
            frequency: midi_frequency(note.pitch),
            program: note.program,
        });
    }

    job.voices.retain(|voice| voice.offset > at);

    let mut sum = 0.0;
    for voice in &job.voices {
        let track = job
            .mix
            .tracks
            .iter()
            .find_map(|(program, track)| (*program == voice.program).then_some(track));

        let audible =
            track.is_some_and(|track| !track.muted && (!solo || track.solo) && track.gain > 0.0);
        if !audible {
            continue;
        }

        let gain = track.map_or(0.0, |track| track.gain);
        sum += (at * f64::from(voice.frequency) * std::f64::consts::TAU).sin() as f32 * gain;
    }

    (sum * 0.25).clamp(-1.0, 1.0)
}

fn midi_frequency(pitch: u8) -> f32 {
    (440.0 * 2.0f64.powf((f64::from(pitch) - 69.0) / 12.0)) as f32
}
