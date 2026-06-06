use std::path::Path;
use std::time::Instant;

use clap::Parser;

use neunote_core::audio::resampler::Resampler;
use neunote_core::midi::writer::write_midi_file_from_tracks;
use neunote_core::ml::pipeline::BasicPitch;
use neunote_core::ml::weights::load_cnn_weights;
use neunote_core::{PitchRangeAssigner, TrackAssigner};

#[derive(Parser)]
#[command(
    name = "neunote",
    about = "Transcribes polyphonic audio to MIDI using the Basic Pitch model"
)]
struct Cli {
    /// Input WAV file
    input: String,

    /// Output MIDI file (default: <input>.mid)
    output: Option<String>,

    /// Note sensitivity 0.0–1.0 (default: 0.5). Higher = more notes detected.
    #[arg(long, default_value = "0.5")]
    note_sensitivity: f32,

    /// Split sensitivity 0.0–1.0 (default: 0.7). Higher = fewer note splits.
    #[arg(long, default_value = "0.7")]
    split_sensitivity: f32,

    /// Minimum note duration in milliseconds (default: 128)
    #[arg(long, default_value = "128")]
    min_note_duration: f32,
}

fn main() {
    let cli = Cli::parse();

    if !(0.0..=1.0).contains(&cli.note_sensitivity) {
        eprintln!("Error: --note-sensitivity must be between 0.0 and 1.0");
        std::process::exit(1);
    }
    if !(0.0..=1.0).contains(&cli.split_sensitivity) {
        eprintln!("Error: --split-sensitivity must be between 0.0 and 1.0");
        std::process::exit(1);
    }
    if cli.min_note_duration < 0.0 {
        eprintln!("Error: --min-note-duration must be non-negative");
        std::process::exit(1);
    }

    let output_path = cli.output.unwrap_or_else(|| {
        let stem = Path::new(&cli.input)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        format!("{}.mid", stem)
    });

    // Determine model directory
    let model_dir = find_model_dir().unwrap_or_else(|| {
        eprintln!("Error: Cannot find CNN model JSON files.");
        eprintln!("Place cnn_contour_model.json, cnn_note_model.json,");
        eprintln!("       cnn_onset_1_model.json, cnn_onset_2_model.json");
        eprintln!("in ./models/ or /usr/share/neunote/models/ or next to the binary.");
        std::process::exit(1);
    });
    eprintln!("Using models from: {}", model_dir.display());

    // Load CNN weights
    eprint!("Loading CNN weights... ");
    let start = Instant::now();
    let w = load_cnn_weights(&model_dir).unwrap_or_else(|e| {
        eprintln!("Failed: {}", e);
        std::process::exit(1);
    });
    eprintln!("done ({:.2}s)", start.elapsed().as_secs_f64());

    // Create pipeline and set parameters
    let mut bp = BasicPitch::new(&w, &model_dir);
    bp.set_parameters(
        cli.note_sensitivity,
        cli.split_sensitivity,
        cli.min_note_duration,
    );

    // Read WAV file
    eprint!("Reading WAV: {}... ", cli.input);
    let (samples, sample_rate) = read_wav(&cli.input).unwrap_or_else(|e| {
        eprintln!("Failed: {}", e);
        std::process::exit(1);
    });
    eprintln!("{} samples @ {} Hz", samples.len(), sample_rate);

    // Resample to 22050 Hz
    eprint!("Resampling to 22050 Hz... ");
    let audio_22050 = if (sample_rate - 22050.0).abs() > 0.5 {
        let mut resampler = Resampler::new();
        resampler.prepare(sample_rate, 22050.0);
        resampler.process(&samples, sample_rate, 22050.0)
    } else {
        samples
    };
    eprintln!("{} samples", audio_22050.len());

    // Transcribe
    eprint!("Transcribing... ");
    let start = Instant::now();
    bp.transcribe(&audio_22050);
    let elapsed = start.elapsed().as_secs_f64();
    let events = bp.note_events();
    eprintln!("done ({:.2}s) — {} notes found", elapsed, events.len());

    if events.is_empty() {
        eprintln!("Warning: No notes detected. Try adjusting --note-sensitivity.");
        return;
    }

    // Assign notes to tracks by pitch range
    let assigner = PitchRangeAssigner::default();
    let tracks = assigner.assign_tracks(events);

    // Print note summary
    let total_duration = tracks
        .iter()
        .flat_map(|t| t.events.iter())
        .map(|e| e.end_time)
        .fold(0.0, f64::max);
    eprintln!("\nTranscription Summary:");
    eprintln!("  Duration: {:.1}s", total_duration);
    eprintln!(
        "  Notes: {}",
        tracks.iter().map(|t| t.events.len()).sum::<usize>()
    );
    for track in &tracks {
        let min_pitch = track.events.iter().map(|e| e.pitch).min().unwrap_or(0);
        let max_pitch = track.events.iter().map(|e| e.pitch).max().unwrap_or(0);
        eprintln!(
            "  {}: {} notes ({}–{})",
            track.name,
            track.events.len(),
            neunote_core::midi::events::midi_note_to_str(min_pitch),
            neunote_core::midi::events::midi_note_to_str(max_pitch),
        );
    }

    // Write MIDI
    eprint!("Writing MIDI: {}... ", output_path);
    let file = std::fs::File::create(&output_path).unwrap_or_else(|e| {
        eprintln!("Failed to create {}: {}", output_path, e);
        std::process::exit(1);
    });
    let mut writer = std::io::BufWriter::new(file);
    write_midi_file_from_tracks(&mut writer, &tracks, 120.0).unwrap_or_else(|e| {
        eprintln!("Failed to write MIDI: {}", e);
        std::process::exit(1);
    });
    eprintln!("done");

    eprintln!("\nSuccess! MIDI written to: {}", output_path);
}

fn find_model_dir() -> Option<std::path::PathBuf> {
    let candidates = [
        Path::new("./models").to_path_buf(),
        Path::new("/usr/share/neunote/models").to_path_buf(),
        Path::new("/usr/local/share/neunote/models").to_path_buf(),
    ];
    for dir in &candidates {
        if dir.join("cnn_contour_model.json").exists() {
            return Some(dir.clone());
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let sibling = exe.parent().unwrap().join("models");
        if sibling.join("cnn_contour_model.json").exists() {
            return Some(sibling);
        }
    }
    None
}

fn read_wav(path: &str) -> Result<(Vec<f32>, f64), String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| format!("Cannot open WAV: {}", e))?;
    let spec = reader.spec();
    if spec.bits_per_sample != 16 && spec.bits_per_sample != 24 && spec.bits_per_sample != 32 {
        return Err(format!("Unsupported bit depth: {}", spec.bits_per_sample));
    }
    if spec.sample_format != hound::SampleFormat::Int
        && spec.sample_format != hound::SampleFormat::Float
    {
        return Err(format!(
            "Unsupported sample format: {:?}",
            spec.sample_format
        ));
    }

    let sample_rate = spec.sample_rate as f64;
    let channels = spec.channels as usize;

    let total_samples = reader.len() as usize / (spec.bits_per_sample as usize / 8 * channels);
    let mut mono = Vec::with_capacity(total_samples / channels);
    let mut frame_sum = 0.0f64;
    let mut frame_count = 0usize;

    match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => {
                for sample in reader.samples::<i16>() {
                    let s = sample.map_err(|e| format!("WAV read error: {}", e))?;
                    frame_sum += s as f64 / i16::MAX as f64;
                    frame_count += 1;
                    if frame_count >= channels {
                        mono.push((frame_sum / channels as f64) as f32);
                        frame_sum = 0.0;
                        frame_count = 0;
                    }
                }
            }
            24 => {
                for sample in reader.samples::<i32>() {
                    let s = sample.map_err(|e| format!("WAV read error: {}", e))?;
                    frame_sum += s as f64 / 8388607.0;
                    frame_count += 1;
                    if frame_count >= channels {
                        mono.push((frame_sum / channels as f64) as f32);
                        frame_sum = 0.0;
                        frame_count = 0;
                    }
                }
            }
            _ => {
                for sample in reader.samples::<i32>() {
                    let s = sample.map_err(|e| format!("WAV read error: {}", e))?;
                    frame_sum += s as f64 / i32::MAX as f64;
                    frame_count += 1;
                    if frame_count >= channels {
                        mono.push((frame_sum / channels as f64) as f32);
                        frame_sum = 0.0;
                        frame_count = 0;
                    }
                }
            }
        },
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                let s = sample.map_err(|e| format!("WAV read error: {}", e))?;
                frame_sum += s as f64;
                frame_count += 1;
                if frame_count >= channels {
                    mono.push((frame_sum / channels as f64) as f32);
                    frame_sum = 0.0;
                    frame_count = 0;
                }
            }
        }
    }

    let max_val = mono.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if max_val > 1.0 {
        for s in mono.iter_mut() {
            *s /= max_val;
        }
    }

    Ok((mono, sample_rate))
}
