//! The desktop shell: a dialog for files, a digest-verified cache for weights.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use neunote_types::ModelSize;
use neunote_ui::{LoadedAudio, LoadedWeights, Shell};

const AUDIO: &[&str] = &["wav", "flac", "mp3", "ogg", "oga", "opus", "aac", "m4a"];

pub(crate) fn shell() -> Shell {
    // Resolved once: nothing installs a checkpoint while the window is open.
    let cache = neunote_models::cache();
    let cached: HashMap<ModelSize, PathBuf> = ModelSize::ALL
        .into_iter()
        .filter(|size| cache.model_path(*size).is_file())
        .map(|size| (size, cache.model_path(size)))
        .collect();

    Shell {
        cached_weights: Rc::new(move |size| cached.get(&size).cloned()),

        // The desktop reads weights from the cache above, or from a file the
        // user picks; there is nothing for the view to resolve or fetch.
        resolve_weights: Rc::new(|_size, done| {
            done(Err(String::from(
                "no cached checkpoint for that size -- pick a .gguf, or run `neunote models fetch`",
            )));
        }),

        fetch_weights: Rc::new(|size, _accepted, _progress, done| {
            done(Err(format!(
                "run `neunote models fetch --size {}` and it will be here",
                size.as_str()
            )));
        }),

        // The same marker file the CLI writes, so one answer covers both.
        licence_accepted: Rc::new(|| neunote_models::cache().licence_accepted()),

        set_licence_accepted: Rc::new(|accepted| {
            if accepted {
                let _ = neunote_models::cache().accept_licence();
            }
        }),

        pick_audio: Rc::new(|done| {
            let picked = rfd::FileDialog::new()
                .add_filter("Audio", AUDIO)
                .pick_file();
            done(read_audio(picked));
        }),

        pick_weights: Rc::new(|done| {
            let picked = rfd::FileDialog::new()
                .add_filter("Checkpoint", &["gguf"])
                .pick_file();
            done(picked.and_then(|path| {
                Some(LoadedWeights {
                    name: file_name(&path),
                    bytes: std::fs::read(&path).ok()?,
                })
            }));
        }),

        save_midi: Rc::new(|name, bytes| {
            let path = rfd::FileDialog::new()
                .set_file_name(name)
                .add_filter("MIDI", &["mid"])
                .save_file()
                .ok_or_else(|| String::from("save cancelled"))?;
            std::fs::write(&path, bytes).map_err(|error| error.to_string())?;
            Ok(path.display().to_string())
        }),
    }
}

fn read_audio(path: Option<PathBuf>) -> Option<LoadedAudio> {
    let path = path?;
    Some(LoadedAudio {
        name: file_name(&path),
        bytes: std::fs::read(&path).ok()?,
    })
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
