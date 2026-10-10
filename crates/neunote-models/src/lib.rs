#![forbid(unsafe_code)]

//! Model weights: where they live, how they are fetched, how they are checked.
//!
//! The weights are CC BY-NC 4.0 and are never committed. They are downloaded at
//! runtime into a cache directory and verified against digests compiled into
//! this crate -- never against a checksum fetched from the same server, which
//! would defeat the point of checking.
//!
//! The pins, the sizes and the digest check compile everywhere, including
//! wasm32. Fetching does not: it needs a filesystem to keep a resumable `.part`
//! file in and sockets reqwest can use, neither of which a browser has. A wasm
//! host obtains the weights itself and hands the bytes over, so it calls
//! [`verify_bytes`] rather than [`fetch`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use neunote_types::{CHECKPOINT_FORMAT_VERSION, ModelSize};
use sha2::{Digest, Sha256};
use thiserror::Error;

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

    // Only the downloader reports progress or owns a cancel flag; on wasm the
    // registry is read-only, because the host is what moves the weights.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
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

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn cancel_flag(&self) -> Arc<AtomicBool> {
        let mut flags = self.cancel.lock().expect("cancel lock");
        flags.clear();
        let flag = Arc::new(AtomicBool::new(false));
        flags.push(flag.clone());
        flag
    }
}

/// Check a buffer of weights against an expected entry.
///
/// The portable half of verification, for a host that already holds the bytes.
/// Compares the length first, then the digest, and reports both as the same
/// errors the file path reports so a caller can treat the two identically.
pub fn verify_bytes(bytes: &[u8], expected: &ModelEntry) -> Result<(), ModelError> {
    if bytes.len() as u64 != expected.num_bytes {
        return Err(ModelError::SizeMismatch {
            name: expected.file_name.to_owned(),
            actual: bytes.len() as u64,
            expected: expected.num_bytes,
        });
    }

    let actual = hex::encode(Sha256::digest(bytes));
    if actual != expected.sha256 {
        return Err(ModelError::Checksum {
            name: expected.file_name.to_owned(),
            actual,
            expected: expected.sha256.to_owned(),
        });
    }

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
mod download;
#[cfg(not(target_arch = "wasm32"))]
pub use download::{Cache, base_url, cache, fetch, fetch_entry, verify_against};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_bytes_rejects_a_buffer_of_the_wrong_length() {
        let error = verify_bytes(&[0u8; 16], entry(ModelSize::Small)).unwrap_err();
        assert!(
            matches!(
                error,
                ModelError::SizeMismatch {
                    actual: 16,
                    expected: 209_425_152,
                    ..
                }
            ),
            "got {error:?}"
        );
    }

    #[test]
    fn verify_bytes_rejects_the_right_length_with_the_wrong_bytes() {
        // A buffer as long as the small model but not its contents. Allocating
        // the real 200 MiB is not worth a unit test: the digest comparison runs
        // on whatever length arrives, and the length check is already covered.
        let expected = ModelEntry {
            size: ModelSize::Small,
            file_name: "synthetic.bin",
            num_bytes: 4,
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        };
        let error = verify_bytes(&[1u8, 2, 3, 4], &expected).unwrap_err();
        assert!(
            matches!(error, ModelError::Checksum { .. }),
            "got {error:?}"
        );
    }

    #[test]
    fn verify_bytes_accepts_bytes_matching_the_compiled_digest() {
        let digest = hex::encode(Sha256::digest(b"muscriptor"));
        let expected = ModelEntry {
            size: ModelSize::Small,
            file_name: "synthetic.bin",
            num_bytes: b"muscriptor".len() as u64,
            sha256: Box::leak(digest.into_boxed_str()),
        };
        assert!(verify_bytes(b"muscriptor", &expected).is_ok());
    }

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
}
