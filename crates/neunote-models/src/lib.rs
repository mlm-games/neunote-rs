#![forbid(unsafe_code)]

//! Model weights: where they live, how they are fetched, how they are checked.
//!
//! The weights are CC BY-NC 4.0 and are never committed. They are downloaded at
//! runtime into a cache directory and verified against digests compiled into
//! this crate -- never against a checksum fetched from the same server, which
//! would defeat the point of checking.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use neunote_types::{CHECKPOINT_FORMAT_VERSION, ModelSize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::io::AsyncWriteExt;

pub mod manifest;

pub use manifest::{MODEL_DIRECTORY, MODEL_REPO, MODEL_REVISION, ModelEntry, entry, resolve_url};

/// The weights are not commercial. Anything that downloads them has to make
/// that visible, and record that the user accepted it.
pub const WEIGHTS_LICENSE: &str = "CC BY-NC 4.0";
pub const WEIGHTS_LICENSE_URL: &str = "https://creativecommons.org/licenses/by-nc/4.0/";

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("model already present: {0}")]
    AlreadyPresent(String),

    #[error("download failed: {0}")]
    Download(String),

    #[error("the server returned {status}, which cannot be resumed")]
    NotResumable { status: u16 },

    #[error("{name} is {actual} bytes, expected {expected}")]
    SizeMismatch {
        name: String,
        actual: u64,
        expected: u64,
    },

    #[error("{name} has sha256 {actual}, expected {expected}")]
    Checksum {
        name: String,
        actual: String,
        expected: String,
    },

    #[error("no {0} model is installed")]
    NotInstalled(String),

    #[error(
        "model format version {found} is not supported, this build reads {CHECKPOINT_FORMAT_VERSION}"
    )]
    FormatVersion { found: u32 },

    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),

    #[error("cancelled")]
    Cancelled,
}

/// Where a download has got to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Connecting,
    Downloading,
    Verifying,
    Ready,
    Failed,
}

/// Progress for one model size.
#[derive(Debug, Clone, PartialEq)]
pub struct Status {
    pub size: ModelSize,
    pub phase: Phase,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

impl Status {
    fn new(size: ModelSize) -> Self {
        Self {
            size,
            phase: Phase::Idle,
            downloaded_bytes: 0,
            total_bytes: entry(size).num_bytes,
        }
    }

    pub fn is_busy(&self) -> bool {
        matches!(
            self.phase,
            Phase::Connecting | Phase::Downloading | Phase::Verifying
        )
    }

    pub fn is_ready(&self) -> bool {
        self.phase == Phase::Ready
    }

    /// 0.0 to 1.0, or 0.0 before the size is known.
    pub fn fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        (self.downloaded_bytes as f64 / self.total_bytes as f64).clamp(0.0, 1.0)
    }
}

/// Tracks progress and cancellation for downloads in flight.
#[derive(Debug, Default)]
pub struct Registry {
    statuses: Mutex<Vec<Status>>,
    cancel: Mutex<Vec<Arc<AtomicBool>>>,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            statuses: Mutex::new(ModelSize::ALL.into_iter().map(Status::new).collect()),
            cancel: Mutex::new(Vec::new()),
        }
    }

    pub fn status(&self, size: ModelSize) -> Status {
        self.statuses
            .lock()
            .expect("status lock")
            .iter()
            .find(|status| status.size == size)
            .cloned()
            .expect("every size has a status")
    }

    pub fn statuses(&self) -> Vec<Status> {
        self.statuses.lock().expect("status lock").clone()
    }

    fn update(&self, size: ModelSize, phase: Phase, downloaded: u64) {
        let mut statuses = self.statuses.lock().expect("status lock");
        if let Some(status) = statuses.iter_mut().find(|status| status.size == size) {
            status.phase = phase;
            status.downloaded_bytes = downloaded;
        }
    }

    /// Ask an in-flight download for this size to stop. The partial file is
    /// kept, so resuming it later is cheaper than starting over.
    pub fn cancel(&self, size: ModelSize) {
        for flag in self.cancel.lock().expect("cancel lock").iter() {
            flag.store(true, Ordering::SeqCst);
        }
        let _ = size;
    }

    pub fn reset_cancellation(&self) {
        for flag in self.cancel.lock().expect("cancel lock").iter() {
            flag.store(false, Ordering::SeqCst);
        }
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        let mut flags = self.cancel.lock().expect("cancel lock");
        flags.clear();
        let flag = Arc::new(AtomicBool::new(false));
        flags.push(flag.clone());
        flag
    }
}

/// A cache directory holding the weights.
///
/// Passed around as a value rather than read from the environment on every call:
/// two jobs fetching different sizes, or a test alongside a running app, would
/// otherwise share one ambient directory and stomp on each other's part files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cache {
    dir: PathBuf,
}

/// The cache the process uses by default, honouring `NEUNOTE_MODELS_DIR`.
pub fn cache() -> Cache {
    match std::env::var_os("NEUNOTE_MODELS_DIR") {
        Some(dir) => Cache::new(dir),
        None => Cache::platform_default(),
    }
}

impl Cache {
    /// The platform's data directory: XDG on Linux, Application Support on
    /// macOS, AppData on Windows.
    pub fn platform_default() -> Self {
        let base = if cfg!(target_os = "macos") {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join("Library").join("Application Support"))
        } else if cfg!(target_os = "windows") {
            std::env::var_os("APPDATA").map(PathBuf::from)
        } else {
            std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
                })
        };

        Self {
            dir: base
                .unwrap_or_else(|| PathBuf::from("."))
                .join("neunote")
                .join("models"),
        }
    }

    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &std::path::Path {
        &self.dir
    }

    /// Where a size's model lives once it is installed.
    pub fn model_path(&self, size: ModelSize) -> PathBuf {
        self.model_path_for(entry(size).file_name)
    }

    /// Where a download of `file_name` accumulates before it is verified.
    pub fn part_path_for(&self, file_name: &str) -> PathBuf {
        self.dir.join(format!("{file_name}.part"))
    }

    /// Where a file named `file_name` is installed.
    pub fn model_path_for(&self, file_name: &str) -> PathBuf {
        self.dir.join(file_name)
    }

    fn acceptance_path(&self) -> PathBuf {
        self.dir.join("LICENCE-ACCEPTED")
    }

    /// The length of an installed file, if it matches the expected length.
    ///
    /// Cheap on purpose: this runs on every tick of a UI, and hashing a 2.7 GB
    /// file per tick would make the app unusable. The digest is checked by
    /// [`Cache::verify`] and by [`fetch_entry`].
    pub fn installed_bytes(&self, size: ModelSize) -> Option<u64> {
        let metadata = std::fs::metadata(self.model_path(size)).ok()?;
        (metadata.len() == entry(size).num_bytes).then_some(metadata.len())
    }

    pub fn is_installed(&self, size: ModelSize) -> bool {
        self.installed_bytes(size).is_some()
    }

    /// Hash an installed file against the compiled-in digest.
    pub async fn verify(&self, size: ModelSize) -> Result<(), ModelError> {
        verify_against(&self.model_path(size), entry(size)).await
    }

    /// Bytes already downloaded for a file, whether or not they are usable.
    pub fn partial_bytes_for(&self, file_name: &str) -> u64 {
        std::fs::metadata(self.part_path_for(file_name))
            .map(|metadata| metadata.len())
            .unwrap_or(0)
    }

    /// Drop a partially downloaded file.
    pub fn discard_partial_for(&self, file_name: &str) -> std::io::Result<()> {
        let path = self.part_path_for(file_name);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    /// Remove a file from the cache.
    pub fn remove_file(&self, file_name: &str) -> std::io::Result<()> {
        let path = self.model_path_for(file_name);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    /// Open the cache directory in the platform's file browser.
    pub fn open(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;

        let command = if cfg!(target_os = "macos") {
            "open"
        } else if cfg!(target_os = "windows") {
            "explorer"
        } else {
            "xdg-open"
        };

        std::process::Command::new(command)
            .arg(&self.dir)
            .spawn()
            .map(|_| ())
    }

    /// Whether the user has acknowledged the weights' non-commercial licence.
    pub fn licence_accepted(&self) -> bool {
        self.acceptance_path().exists()
    }

    /// Record acceptance of the weights' licence.
    pub fn accept_licence(&self) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.acceptance_path();
        std::fs::write(
            &path,
            format!(
                "MuScriptor weights accepted under {WEIGHTS_LICENSE}\n{WEIGHTS_LICENSE_URL}\n\
                 Checkpoint format version {CHECKPOINT_FORMAT_VERSION}\n{}",
                entry(ModelSize::Medium).sha256
            ),
        )?;
        Ok(path)
    }
}

/// Check a file's length and digest against an expected entry.
///
/// Streamed in fixed-size reads: the large model is 2.7 GB and slurping it
/// would need a buffer to match.
pub async fn verify_against(
    path: &std::path::Path,
    expected: &ModelEntry,
) -> Result<(), ModelError> {
    let actual_len = tokio::fs::metadata(path).await?.len();
    if actual_len != expected.num_bytes {
        return Err(ModelError::SizeMismatch {
            name: expected.file_name.to_owned(),
            actual: actual_len,
            expected: expected.num_bytes,
        });
    }

    let actual = hash_file(path).await?;
    if actual != expected.sha256 {
        return Err(ModelError::Checksum {
            name: expected.file_name.to_owned(),
            actual,
            expected: expected.sha256.to_owned(),
        });
    }

    Ok(())
}

const HASH_CHUNK: usize = 1 << 20;

async fn hash_file(path: &std::path::Path) -> Result<String, ModelError> {
    use tokio::io::AsyncReadExt;

    let mut file = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; HASH_CHUNK];

    loop {
        let read = file.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hex::encode(hasher.finalize()))
}

/// The base the downloader builds URLs from.
///
/// Split out so the download rules can be tested against a local server without
/// an environment override that could also be set in production.
pub fn base_url() -> String {
    format!("https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_DIRECTORY}")
}

/// Download a model unless a valid copy is already there.
pub async fn fetch(
    cache: &Cache,
    size: ModelSize,
    registry: &Registry,
    on_progress: impl FnMut(&Status),
) -> Result<PathBuf, ModelError> {
    fetch_entry(cache, size, entry(size), registry, &base_url(), on_progress).await
}

/// Mark a size's job as over, successfully or not.
///
/// Without this a failed download leaves the registry showing `Verifying` for
/// ever, and every caller polling it would wait on a job that is not running.
fn settle(registry: &Registry, size: ModelSize, downloaded: u64, ok: bool) {
    registry.update(
        size,
        if ok { Phase::Ready } else { Phase::Failed },
        downloaded,
    );
}

/// Download a model from an arbitrary base URL, applying the same rules.
///
/// Resumable: an interrupted download leaves a `.part` file that the next call
/// continues from with an HTTP range request. The digest is computed over the
/// finished file and compared against the expected value, then the file is
/// renamed into place, so a partial file is never mistaken for a good one.
pub async fn fetch_entry(
    cache: &Cache,
    size: ModelSize,
    expected: &ModelEntry,
    registry: &Registry,
    base: &str,
    mut on_progress: impl FnMut(&Status),
) -> Result<PathBuf, ModelError> {
    let already_there = std::fs::metadata(cache.model_path_for(expected.file_name))
        .is_ok_and(|metadata| metadata.len() == expected.num_bytes);
    if already_there {
        registry.update(size, Phase::Ready, expected.num_bytes);
        on_progress(&registry.status(size));
        return Ok(cache.model_path_for(expected.file_name));
    }

    registry.reset_cancellation();
    let cancel = registry.cancel_flag();

    std::fs::create_dir_all(cache.dir())?;
    let final_path = cache.model_path_for(expected.file_name);
    let part = cache.part_path_for(expected.file_name);

    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    // A part longer than the whole file means the download was corrupt.
    let have = if have > expected.num_bytes {
        tokio::fs::remove_file(&part).await?;
        0
    } else {
        have
    };

    registry.update(size, Phase::Connecting, have);
    on_progress(&registry.status(size));

    let client = reqwest::Client::builder()
        .build()
        .map_err(|error| ModelError::Download(error.to_string()))?;

    let url = format!("{base}/{}", expected.file_name);
    let mut request = client.get(&url);
    if have > 0 {
        request = request.header("Range", format!("bytes={have}-"));
    }

    let response = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            settle(registry, size, have, false);
            return Err(ModelError::Download(error.to_string()));
        }
    };

    let status = response.status();
    if !status.is_success() {
        settle(registry, size, have, false);
        return Err(ModelError::Download(format!("{status} from {url}")));
    }

    // A server that ignores the range request sends 200 and the whole body, so
    // the prefix already on disk must be discarded rather than doubled up.
    let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT;
    let mut have = if have > 0 && !resumed {
        tokio::fs::remove_file(&part).await?;
        0
    } else {
        have
    };

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(have > 0)
        .truncate(have == 0)
        .open(&part)
        .await?;

    registry.update(size, Phase::Downloading, have);
    on_progress(&registry.status(size));

    let mut stream = response.bytes_stream();
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::SeqCst) {
            file.flush().await?;
            settle(registry, size, have, false);
            return Err(ModelError::Cancelled);
        }
        let chunk = chunk.map_err(|error| ModelError::Download(error.to_string()))?;
        file.write_all(&chunk).await?;
        have += chunk.len() as u64;

        registry.update(size, Phase::Downloading, have);
        on_progress(&registry.status(size));
    }

    file.flush().await?;
    drop(file);

    if have != expected.num_bytes {
        settle(registry, size, have, false);
        return Err(ModelError::SizeMismatch {
            name: expected.file_name.to_owned(),
            actual: have,
            expected: expected.num_bytes,
        });
    }

    registry.update(size, Phase::Verifying, have);
    on_progress(&registry.status(size));

    if let Err(error) = verify_against(&part, expected).await {
        settle(registry, size, have, false);
        return Err(error);
    }

    tokio::fs::rename(&part, &final_path).await?;
    settle(registry, size, have, true);
    on_progress(&registry.status(size));

    Ok(final_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_size_has_a_url_pinned_to_one_revision() {
        for size in ModelSize::ALL {
            let url = resolve_url(entry(size));
            assert!(url.starts_with("https://huggingface.co/"));
            assert!(url.contains(MODEL_REVISION), "{size}: not pinned");
            assert!(url.ends_with(entry(size).file_name));
        }
    }

    #[test]
    fn digests_are_lowercase_hex_of_the_right_length() {
        for size in ModelSize::ALL {
            let digest = entry(size).sha256;
            assert_eq!(digest.len(), 64, "{size}: not a sha256");
            assert!(
                digest
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase())
            );
        }
    }

    #[test]
    fn sizes_increase_with_the_model() {
        let small = entry(ModelSize::Small).num_bytes;
        let medium = entry(ModelSize::Medium).num_bytes;
        let large = entry(ModelSize::Large).num_bytes;
        assert!(small < medium && medium < large);
        assert_eq!(
            entry(ModelSize::Medium).file_name,
            "muscriptor-medium-f16.gguf"
        );
    }

    #[test]
    fn a_registry_reports_a_status_for_every_size() {
        let registry = Registry::new();
        for size in ModelSize::ALL {
            let status = registry.status(size);
            assert_eq!(status.size, size);
            assert_eq!(status.phase, Phase::Idle);
            assert_eq!(status.total_bytes, entry(size).num_bytes);
            assert!(!status.is_busy());
            assert!(!status.is_ready());
            assert_eq!(status.fraction(), 0.0);
        }
        assert_eq!(registry.statuses().len(), ModelSize::ALL.len());
    }

    #[test]
    fn progress_reports_a_clamped_fraction() {
        let mut status = Status::new(ModelSize::Small);
        status.total_bytes = 200;
        status.downloaded_bytes = 50;
        assert!((status.fraction() - 0.25).abs() < 1e-9);
        status.downloaded_bytes = 300;
        assert_eq!(status.fraction(), 1.0, "clamped");
    }

    #[test]
    fn the_default_cache_is_under_a_neunote_models_directory() {
        let dir = Cache::platform_default().dir().to_path_buf();
        assert!(
            dir.ends_with("neunote/models") || dir.ends_with("neunote\\models"),
            "unexpected cache directory: {dir:?}"
        );
    }

    #[test]
    fn two_caches_do_not_share_directories() {
        let a = Cache::new("/tmp/neunote-a");
        let b = Cache::new("/tmp/neunote-b");
        assert_ne!(a.dir(), b.dir());
        assert_eq!(
            a.model_path(ModelSize::Small),
            PathBuf::from("/tmp/neunote-a/muscriptor-small-f16.gguf")
        );
        assert_eq!(
            a.part_path_for("muscriptor-small-f16.gguf"),
            PathBuf::from("/tmp/neunote-a/muscriptor-small-f16.gguf.part")
        );
    }

    #[tokio::test]
    async fn a_missing_model_reports_not_installed() {
        let cache = Cache::new("/tmp/neunote-absent-for-test");
        match cache.verify(ModelSize::Large).await.unwrap_err() {
            ModelError::Io(error) => assert_eq!(error.kind(), std::io::ErrorKind::NotFound),
            other => panic!("expected a not-found io error, got {other:?}"),
        }
    }
}
