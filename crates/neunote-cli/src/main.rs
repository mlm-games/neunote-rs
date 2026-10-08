#![forbid(unsafe_code)]

//! The command line interface.
//!
//! Three commands do real work: `devices`, `models` and `transcribe`.
//! Transcription needs the inference engine, which is not built yet, so
//! `transcribe` reports that plainly rather than pretending or silently
//! falling back to the old Basic Pitch pipeline. Everything up to and after
//! the engine call -- decoding, resampling, the chunk loop, note assembly,
//! MIDI export -- is wired up and exercised by the tests.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use neunote_types::{GroupId, ModelSize, NoteEvent};

use neunote_cli::pipeline;

#[derive(Parser)]
#[command(
    name = "neunote",
    about = "Transcribes audio to MIDI with MuScriptor",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List the compute devices transcription can run on.
    Devices,

    /// Download, verify or inspect the model weights.
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },

    /// Transcribe an audio file to MIDI.
    Transcribe(TranscribeArgs),

    /// Record that you accept the model weights' non-commercial licence.
    ///
    /// Required once, before the first download. The weights are CC BY-NC 4.0
    /// and this app cannot redistribute them.
    AcceptLicence,
}

#[derive(Debug, Subcommand)]
enum ModelsAction {
    /// Show where each model is and whether it is installed.
    List,

    /// Download a model.
    Fetch {
        /// Which size to download.
        #[arg(long, value_enum, default_value_t = ModelChoice::Medium)]
        size: ModelChoice,

        /// Delete the partial file first and start over.
        #[arg(long)]
        restart: bool,
    },

    /// Re-check an installed model against its compiled-in digest.
    Verify {
        #[arg(long, value_enum, default_value_t = ModelChoice::All)]
        size: ModelChoice,
    },

    /// Print the cache directory.
    Path,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum ModelChoice {
    Small,
    Medium,
    Large,
    All,
}

impl From<ModelChoice> for ModelSize {
    fn from(choice: ModelChoice) -> Self {
        match choice {
            ModelChoice::Small => ModelSize::Small,
            ModelChoice::Medium | ModelChoice::All => ModelSize::Medium,
            ModelChoice::Large => ModelSize::Large,
        }
    }
}

#[derive(Debug, clap::Args)]
struct TranscribeArgs {
    /// The audio file to transcribe.
    input: PathBuf,

    /// Where to write the MIDI. Defaults to the input path with a .mid suffix.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Which model to use.
    #[arg(long, value_enum, default_value_t = ModelChoice::Medium)]
    model: ModelChoice,

    /// Transcribe only these instruments, comma separated. Omit for all.
    #[arg(long, value_delimiter = ',')]
    instruments: Option<Vec<String>>,

    /// Let the model predict each chunk's tie prologue instead of forcing the
    /// notes that are still sounding. Prelude forcing is on by default and is
    /// what carries a sustained note across a chunk boundary.
    #[arg(long)]
    no_prelude_forcing: bool,

    /// Tempo of the written MIDI, in beats per minute.
    #[arg(long, default_value_t = neunote_midi::DEFAULT_BPM)]
    bpm: f64,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let cache = neunote_models::cache();

    match run(cli.command, &cache) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(command: Command, cache: &neunote_models::Cache) -> Result<(), String> {
    match command {
        Command::Devices => devices(),
        Command::Models { action } => models(action, cache),
        Command::AcceptLicence => {
            let path = cache
                .accept_licence()
                .map_err(|error| format!("recording acceptance: {error}"))?;
            println!("recorded at {}", path.display());
            println!("the weights may only be used non-commercially.");
            Ok(())
        }

        Command::Transcribe(args) => tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| error.to_string())?
            .block_on(transcribe(args, cache)),
    }
}

/// Device enumeration belongs to the engine, which does not exist yet. Rather
/// than print a fake CPU-only list that would have to be thrown away, this says
/// what is missing.
fn devices() -> Result<(), String> {
    eprintln!(
        "device enumeration is part of the inference engine, which has not been built yet.\n\
         Nothing to list: transcription runs on the CPU once the engine lands."
    );
    Ok(())
}

fn models(action: ModelsAction, cache: &neunote_models::Cache) -> Result<(), String> {
    match action {
        ModelsAction::Path => {
            println!("{}", cache.dir().display());
            return Ok(());
        }

        ModelsAction::List => {
            println!("cache: {}", cache.dir().display());
            println!();
            for size in ModelSize::ALL {
                let entry = neunote_models::entry(size);
                let state = if cache.is_installed(size) {
                    "installed"
                } else {
                    "not downloaded"
                };
                println!(
                    "{:<7} {:<28} {:>13}  {}",
                    size.as_str(),
                    entry.file_name,
                    human_bytes(entry.num_bytes),
                    state
                );
            }
            println!();
            println!(
                "weights are {} and are never bundled; see {}",
                neunote_models::WEIGHTS_LICENSE,
                neunote_models::WEIGHTS_LICENSE_URL
            );
        }

        ModelsAction::Fetch {
            size: choice,
            restart,
        } => {
            if choice == ModelChoice::All {
                return Err("--size must name one model, not 'all'".to_owned());
            }
            let size = ModelSize::from(choice);
            if restart {
                cache
                    .discard_partial_for(neunote_models::entry(size).file_name)
                    .map_err(|error| error.to_string())?;
            }
            fetch(size, cache)?;
            return Ok(());
        }

        ModelsAction::Verify { size: choice } => {
            let sizes: Vec<ModelSize> = if choice == ModelChoice::All {
                ModelSize::ALL.to_vec()
            } else {
                vec![ModelSize::from(choice)]
            };

            for size in sizes {
                match cache.is_installed(size) {
                    true => {
                        tokio_runtime()?
                            .block_on(cache.verify(size))
                            .map_err(|error| format!("{}: {error}", size.as_str()))?;
                        println!("{}: ok", size.as_str());
                    }
                    false => println!("{}: not installed", size.as_str()),
                }
            }
            return Ok(());
        }
    }

    Ok(())
}

fn fetch(size: ModelSize, cache: &neunote_models::Cache) -> Result<(), String> {
    if !cache.licence_accepted() {
        return Err(format!(
            "the MuScriptor weights are {} and may only be used non-commercially.\n\
             Run `neunote accept-licence` to record your acceptance, then fetch again.\n\
             {}",
            neunote_models::WEIGHTS_LICENSE,
            neunote_models::WEIGHTS_LICENSE_URL
        ));
    }

    let entry = neunote_models::entry(size);
    println!(
        "fetching {} ({})",
        entry.file_name,
        human_bytes(entry.num_bytes)
    );

    let registry = neunote_models::Registry::new();
    let mut last_percent = -1i64;

    let result = {
        let cache = cache.clone();
        let registry = &registry;
        tokio_runtime()?.block_on(neunote_models::fetch(&cache, size, registry, |status| {
            let percent = (status.fraction() * 100.0) as i64;
            if percent != last_percent {
                last_percent = percent;
                eprint!(
                    "\r  {percent:>3}%  {:>10} / {}",
                    human_bytes(status.downloaded_bytes),
                    human_bytes(status.total_bytes)
                );
            }
        }))
    };

    eprintln!();

    match result {
        Ok(path) => {
            println!("installed at {}", path.display());
            Ok(())
        }
        Err(error) => Err(match error {
            neunote_models::ModelError::Cancelled => {
                "cancelled; the partial file was kept, run fetch again to resume".to_owned()
            }
            other => other.to_string(),
        }),
    }
}

fn tokio_runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())
}

async fn transcribe(args: TranscribeArgs, cache: &neunote_models::Cache) -> Result<(), String> {
    let instruments = match &args.instruments {
        Some(names) => parse_instruments(names)?,
        None => Vec::new(),
    };

    // Decode and resample first: both are engine-independent, so a failure here
    // is a real error rather than a missing engine.
    eprintln!("reading {}", args.input.display());
    let decoded = neunote_audio::decode_file(&args.input).map_err(|error| error.to_string())?;
    let mono = neunote_audio::to_engine_input(&decoded).map_err(|error| error.to_string())?;

    eprintln!(
        "  {:.1}s, {} samples at {} Hz",
        mono.len() as f64 / f64::from(neunote_types::TRANSCRIPTION_SAMPLE_RATE),
        mono.len(),
        neunote_types::TRANSCRIPTION_SAMPLE_RATE
    );

    if neunote_audio::is_clipped(&mono) {
        eprintln!("  warning: samples outside [-1, 1]; passed through unchanged");
    }

    let size = ModelSize::from(args.model);
    let model_path = cache.model_path(size);
    let notes = pipeline::transcribe(
        &mono,
        &model_path,
        size,
        &instruments,
        !args.no_prelude_forcing,
        cache,
    )
    .await?;

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| args.input.with_extension("mid"));

    neunote_midi::write_midi_file(&output, &notes, args.bpm)
        .map_err(|error| format!("writing {}: {error}", output.display()))?;

    summarise(&notes, &output);
    Ok(())
}

/// The instruments the run is restricted to.
fn parse_instruments(names: &[String]) -> Result<Vec<GroupId>, String> {
    let mut groups = Vec::with_capacity(names.len());

    for name in names {
        let group =
            GroupId::from_name(name).ok_or_else(|| format!("unknown instrument '{name}'"))?;
        if !groups.contains(&group) {
            groups.push(group);
        }
    }

    if groups.is_empty() {
        return Err("--instruments was given but names nothing".to_owned());
    }

    Ok(groups)
}

/// Group notes by instrument and report what was found, so a silent or
/// one-instrument result is visible without opening the file.
fn summarise(notes: &[NoteEvent], output: &Path) {
    let tracks = match neunote_midi::group_by_program(notes) {
        Ok(tracks) => tracks,
        Err(error) => {
            eprintln!();
            eprintln!("cannot lay the notes out as tracks: {error}");
            return;
        }
    };

    eprintln!();
    if tracks.is_empty() {
        eprintln!("no notes found");
    } else {
        for track in &tracks {
            eprintln!("  {:<24} {:>6} notes", track.name, track.note_count());
        }
    }
    eprintln!();
    eprintln!(
        "wrote {} ({} notes, {:.1}s)",
        output.display(),
        notes.len(),
        neunote_midi::duration_secs(&tracks)
    );
}

fn human_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    let mib = bytes as f64 / MIB;
    if mib >= 1024.0 {
        format!("{:.2} GiB", mib / 1024.0)
    } else {
        format!("{mib:.0} MiB")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_are_rendered_at_a_useful_scale() {
        assert_eq!(human_bytes(209_425_152), "200 MiB");
        assert_eq!(human_bytes(618_442_496), "590 MiB");
        assert_eq!(human_bytes(2_739_142_176), "2.55 GiB");
    }

    #[test]
    fn instrument_names_resolve_to_groups() {
        let groups = parse_instruments(&["acoustic_piano".into(), "drums".into()]).unwrap();
        assert_eq!(groups, vec![GroupId(0), GroupId::DRUMS]);

        // Duplicates collapse rather than repeating a conditioning row.
        let groups = parse_instruments(&["violin".into(), "violin".into()]).unwrap();
        assert_eq!(groups, vec![GroupId(9)]);
    }

    #[test]
    fn an_unknown_instrument_is_rejected_with_its_name() {
        let error = parse_instruments(&["kazoo".into()]).unwrap_err();
        assert!(error.contains("kazoo"), "got {error}");
    }

    #[test]
    fn an_empty_instrument_list_is_an_error_not_a_no_op() {
        // The reference forbids every program and every drum for an empty
        // selection, so silently meaning "all" here would be a lie.
        assert!(parse_instruments(&[]).is_err());
        assert!(parse_instruments(&["".into()]).is_err());
    }

    #[test]
    fn the_model_choice_maps_onto_sizes() {
        assert_eq!(ModelSize::from(ModelChoice::Small), ModelSize::Small);
        assert_eq!(ModelSize::from(ModelChoice::Medium), ModelSize::Medium);
        assert_eq!(ModelSize::from(ModelChoice::Large), ModelSize::Large);
    }

    #[test]
    fn the_command_line_parses() {
        let cli = Cli::try_parse_from(["neunote", "models", "list"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Models {
                action: ModelsAction::List
            }
        ));

        let cli = Cli::try_parse_from([
            "neunote",
            "transcribe",
            "song.wav",
            "-o",
            "out.mid",
            "--model",
            "small",
            "--instruments",
            "acoustic_piano,drums",
            "--bpm",
            "90",
        ])
        .unwrap();

        match cli.command {
            Command::Transcribe(args) => {
                assert_eq!(args.input, PathBuf::from("song.wav"));
                assert_eq!(args.output, Some(PathBuf::from("out.mid")));
                assert_eq!(args.model, ModelChoice::Small);
                assert_eq!(
                    args.instruments,
                    Some(vec!["acoustic_piano".to_owned(), "drums".to_owned()])
                );
                assert!((args.bpm - 90.0).abs() < 1e-9);
            }
            other => panic!("expected transcribe, got {other:?}"),
        }
    }

    #[test]
    fn the_output_defaults_to_the_input_with_a_mid_suffix() {
        let cli = Cli::try_parse_from(["neunote", "transcribe", "song.wav"]).unwrap();
        match cli.command {
            Command::Transcribe(args) => {
                let output = args
                    .output
                    .unwrap_or_else(|| args.input.with_extension("mid"));
                assert_eq!(output, PathBuf::from("song.mid"));
            }
            other => panic!("expected transcribe, got {other:?}"),
        }
    }

    #[test]
    fn an_unknown_instrument_flag_is_rejected_at_parse_time() {
        assert!(
            Cli::try_parse_from(["neunote", "transcribe", "song.wav", "--model", "enormous"])
                .is_err()
        );
    }
}
