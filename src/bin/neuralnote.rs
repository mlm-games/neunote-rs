use std::path::Path;
use std::time::Instant;

use neuralnote_core::audio::resampler::Resampler;
use neuralnote_core::midi::writer::write_midi_file;
use neuralnote_core::ml::pipeline::BasicPitch;
use neuralnote_core::ml::weights::load_cnn_weights;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: neuralnote <input.wav> [output.mid]");
        eprintln!();
        eprintln!("Transcribes polyphonic audio to MIDI using the Basic Pitch model.");
        eprintln!("  input.wav   - Audio file (mono or stereo, any sample rate)");
        eprintln!("  output.mid  - Optional MIDI output path (default: input name)");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = args.get(2).cloned().unwrap_or_else(|| {
        let stem = Path::new(input_path).file_stem().unwrap().to_string_lossy().to_string();
        format!("{}.mid", stem)
    });

    // Determine model directory (look next to the binary, then in typical locations)
    let model_dir = find_model_dir().unwrap_or_else(|| {
        eprintln!("Error: Cannot find CNN model JSON files.");
        eprintln!("Place cnn_contour_model.json, cnn_note_model.json,");
        eprintln!("       cnn_onset_1_model.json, cnn_onset_2_model.json");
        eprintln!("in ./models/ or /usr/share/neuralnote/models/ or next to the binary.");
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

    // Create pipeline
    let mut bp = BasicPitch::new(&w, &model_dir);

    // Read WAV file
    eprint!("Reading WAV: {}... ", input_path);
    let (samples, sample_rate) = read_wav(input_path).unwrap_or_else(|e| {
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
    eprintln!(
        "done ({:.2}s) — {} notes found",
        elapsed,
        events.len()
    );

    if events.is_empty() {
        eprintln!("Warning: No notes detected. Try adjusting parameters.");
        return;
    }

    // Print note summary
    let total_duration = events
        .iter()
        .map(|e| e.end_time)
        .fold(0.0, f64::max);
    eprintln!("\nTranscription Summary:");
    eprintln!("  Duration: {:.1}s", total_duration);
    eprintln!("  Notes: {}", events.len());
    let note_range_min = events.iter().map(|e| e.pitch).min().unwrap();
    let note_range_max = events.iter().map(|e| e.pitch).max().unwrap();
    eprintln!(
        "  Range: {} ({}) - {} ({})",
        note_range_min,
        neuralnote_core::midi::events::midi_note_to_str(note_range_min),
        note_range_max,
        neuralnote_core::midi::events::midi_note_to_str(note_range_max),
    );

    // Write MIDI
    eprint!("Writing MIDI: {}... ", output_path);
    let file = std::fs::File::create(&output_path).unwrap_or_else(|e| {
        eprintln!("Failed to create {}: {}", output_path, e);
        std::process::exit(1);
    });
    let mut writer = std::io::BufWriter::new(file);
    write_midi_file(&mut writer, events, 120.0).unwrap_or_else(|e| {
        eprintln!("Failed to write MIDI: {}", e);
        std::process::exit(1);
    });
    eprintln!("done");

    eprintln!("\nSuccess! MIDI written to: {}", output_path);
}

fn find_model_dir() -> Option<std::path::PathBuf> {
    // Check several locations
    let candidates = [
        Path::new("./models").to_path_buf(),
        Path::new("/usr/share/neuralnote/models").to_path_buf(),
        Path::new("/usr/local/share/neuralnote/models").to_path_buf(),
    ];
    for dir in &candidates {
        if dir.join("cnn_contour_model.json").exists() {
            return Some(dir.clone());
        }
    }
    // Check next to the binary
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
        return Err(format!("Unsupported sample format: {:?}", spec.sample_format));
    }

    let sample_rate = spec.sample_rate as f64;
    let channels = spec.channels as usize;

    // Read all samples and convert to f32, mix down to mono
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
                    frame_sum += s as f64 / 8388607.0; // i24 max
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

    // Normalize to [-1, 1]
    let max_val = mono.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if max_val > 1.0 {
        for s in mono.iter_mut() {
            *s /= max_val;
        }
    }

    Ok((mono, sample_rate))
}
