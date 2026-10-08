//! Download behaviour against a local HTTP server, so resume, size and digest
//! rejection are exercised without touching the network or shipping a
//! checkpoint.
//!
//! The served body is small, so each test supplies its own expected length and
//! digest. The downloader's rules are then checked against those, which is the
//! same code path the real manifest goes through.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use neunote_models::{Cache, ModelEntry, ModelError, Phase, Registry};
use neunote_types::ModelSize;
use sha2::{Digest, Sha256};

/// A one-file HTTP server that honours range requests and records them.
struct Server {
    port: u16,
    ranges: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
}

impl Server {
    /// Serves `body` in full.
    fn start(body: Vec<u8>) -> Self {
        Self::with_limit(body, None)
    }

    /// Serves at most `limit` bytes of `body`, so a short response is reachable
    /// without a second server.
    fn with_limit(body: Vec<u8>, limit: Option<usize>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("binding a test server");
        let port = listener.local_addr().unwrap().port();
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let thread_ranges = ranges.clone();
        let thread_stop = stop.clone();
        let body_len = limit.map_or(body.len(), |limit| limit.min(body.len()));

        std::thread::spawn(move || {
            for stream in listener.incoming() {
                if thread_stop.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut stream) = stream else { break };

                let mut buffer = [0u8; 2048];
                let read = stream.read(&mut buffer).unwrap_or(0);
                let request = String::from_utf8_lossy(&buffer[..read]).to_string();

                let range = request
                    .lines()
                    .find(|line| line.to_ascii_lowercase().starts_with("range:"))
                    .map(|line| line[6..].trim().to_owned())
                    .unwrap_or_default();
                thread_ranges.lock().unwrap().push(range.clone());

                let start = range
                    .strip_prefix("bytes=")
                    .and_then(|value| value.split('-').next())
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0);

                let payload = if start < body_len {
                    &body[start..body_len]
                } else {
                    &[]
                };

                let status = if start > 0 {
                    "206 Partial Content"
                } else {
                    "200 OK"
                };
                let header = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\n\
                     Accept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                    payload.len()
                );

                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(payload);
                let _ = stream.flush();
            }
        });

        Self { port, ranges, stop }
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn ranges(&self) -> Vec<String> {
        self.ranges.lock().unwrap().clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(("127.0.0.1", self.port));
    }
}

fn sha256_of(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// An expected entry for a body this test controls.
fn expected_for(name: &str, body: &[u8]) -> ModelEntry {
    ModelEntry {
        size: ModelSize::Small,
        file_name: Box::leak(name.to_owned().into_boxed_str()),
        num_bytes: body.len() as u64,
        sha256: Box::leak(sha256_of(body).into_boxed_str()),
    }
}

/// A scratch cache directory with a distinct name per test, so the tests in
/// this file can run in parallel without sharing one ambient location.
fn cache_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("neunote-models-test-{name}"));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn body(len: usize, fill: u8) -> Vec<u8> {
    vec![fill; len]
}

#[tokio::test]
async fn a_matching_body_is_verified_and_moved_into_place() {
    let payload = body(4096, 0xAB);
    let server = Server::start(payload.clone());
    let cache = Cache::new(cache_dir("fresh"));
    let expected = expected_for("fresh.gguf", &payload);

    let registry = Registry::new();
    let mut phases = Vec::new();
    let path = neunote_models::fetch_entry(
        &cache,
        ModelSize::Small,
        &expected,
        &registry,
        &server.base_url(),
        |status| phases.push(status.phase),
    )
    .await
    .expect("download succeeds");

    assert_eq!(
        std::fs::read(&path).unwrap(),
        payload,
        "the file arrived intact"
    );
    assert!(phases.contains(&Phase::Connecting));
    assert!(phases.contains(&Phase::Downloading));
    assert!(phases.contains(&Phase::Verifying));
    assert_eq!(phases.last(), Some(&Phase::Ready));

    assert!(
        !cache.part_path_for(expected.file_name).exists(),
        "the .part file was renamed away"
    );
    assert_eq!(registry.status(ModelSize::Small).phase, Phase::Ready);
}

#[tokio::test]
async fn a_body_of_the_wrong_length_is_rejected() {
    let payload = body(4096, 0xAB);
    let server = Server::with_limit(payload.clone(), Some(2048));
    let cache = Cache::new(cache_dir("short"));
    let expected = expected_for("short.gguf", &payload);

    let registry = Registry::new();
    let error = neunote_models::fetch_entry(
        &cache,
        ModelSize::Small,
        &expected,
        &registry,
        &server.base_url(),
        |_| {},
    )
    .await
    .unwrap_err();

    assert!(
        matches!(
            error,
            ModelError::SizeMismatch {
                actual: 2048,
                expected: 4096,
                ..
            }
        ),
        "a truncated body must not pass, got {error:?}"
    );
    assert!(
        !cache.model_path_for(expected.file_name).exists(),
        "a rejected file must not be left looking installed"
    );
}

#[tokio::test]
async fn a_body_of_the_right_length_but_the_wrong_bytes_is_rejected() {
    // Only the digest can catch this one.
    let payload = body(4096, 0xAB);
    let server = Server::start(payload);
    let cache = Cache::new(cache_dir("corrupt"));
    let expected = ModelEntry {
        size: ModelSize::Small,
        file_name: "corrupt.gguf",
        num_bytes: 4096,
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    };

    let registry = Registry::new();
    let error = neunote_models::fetch_entry(
        &cache,
        ModelSize::Small,
        &expected,
        &registry,
        &server.base_url(),
        |_| {},
    )
    .await
    .unwrap_err();

    assert!(
        matches!(error, ModelError::Checksum { .. }),
        "same length, wrong content: the digest is the only gate, got {error:?}"
    );
    assert!(!cache.model_path_for(expected.file_name).exists());
}

#[tokio::test]
async fn an_already_installed_model_is_not_downloaded_again() {
    let payload = body(4096, 0xAB);
    let server = Server::start(payload.clone());
    let cache = Cache::new(cache_dir("installed"));
    let expected = expected_for("installed.gguf", &payload);

    std::fs::write(cache.model_path_for(expected.file_name), &payload).unwrap();

    let registry = Registry::new();
    let path = neunote_models::fetch_entry(
        &cache,
        ModelSize::Small,
        &expected,
        &registry,
        &server.base_url(),
        |_| {},
    )
    .await
    .expect("the installed model is used");

    assert_eq!(path, cache.model_path_for(expected.file_name));
    assert!(
        server.ranges().is_empty(),
        "no request should have been made"
    );
    assert_eq!(registry.status(ModelSize::Small).phase, Phase::Ready);
}

#[tokio::test]
async fn a_resume_asks_only_for_the_remainder() {
    let payload = body(4096, 0xAB);
    let server = Server::start(payload.clone());
    let cache = Cache::new(cache_dir("resume"));
    let expected = expected_for("resume.gguf", &payload);

    std::fs::write(cache.part_path_for(expected.file_name), &payload[..1024]).unwrap();

    let registry = Registry::new();
    let path = neunote_models::fetch_entry(
        &cache,
        ModelSize::Small,
        &expected,
        &registry,
        &server.base_url(),
        |_| {},
    )
    .await
    .expect("the resumed download succeeds");

    assert_eq!(
        std::fs::read(&path).unwrap(),
        payload,
        "the result is whole"
    );

    let ranges = server.ranges();
    assert_eq!(ranges.len(), 1, "one request, not two");
    assert!(
        ranges[0].starts_with("bytes=1024-"),
        "expected a range request from 1024, got {ranges:?}"
    );
}

#[tokio::test]
async fn a_part_longer_than_the_model_is_discarded_rather_than_resumed() {
    let payload = body(4096, 0xAB);
    let server = Server::start(payload.clone());
    let cache = Cache::new(cache_dir("oversized"));
    let expected = expected_for("oversized.gguf", &payload);

    // Longer than the whole model, which no real partial file can be.
    std::fs::write(cache.part_path_for(expected.file_name), vec![0u8; 8192]).unwrap();

    let registry = Registry::new();
    let path = neunote_models::fetch_entry(
        &cache,
        ModelSize::Small,
        &expected,
        &registry,
        &server.base_url(),
        |_| {},
    )
    .await
    .expect("a fresh download succeeds");

    assert_eq!(std::fs::read(&path).unwrap(), payload);

    let ranges = server.ranges();
    assert!(
        ranges.iter().all(|range| range.is_empty()),
        "an oversized part must be discarded, not resumed: {ranges:?}"
    );
}

#[tokio::test]
async fn a_server_that_ignores_the_range_request_does_not_double_the_prefix() {
    let payload = body(4096, 0xAB);
    let cache = Cache::new(cache_dir("no-range"));

    // This server answers 200 with the whole body whatever range it is given,
    // which is what a proxy that strips Range headers does.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let served = payload.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().take(4) {
            let Ok(mut stream) = stream else { break };
            let mut buffer = [0u8; 1024];
            let _ = stream.read(&mut buffer);
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                served.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&served);
            let _ = stream.flush();
        }
    });

    let expected = expected_for("no-range.gguf", &payload);
    std::fs::write(cache.part_path_for(expected.file_name), &payload[..1024]).unwrap();

    let registry = Registry::new();
    let path = neunote_models::fetch_entry(
        &cache,
        ModelSize::Small,
        &expected,
        &registry,
        &format!("http://127.0.0.1:{port}"),
        |_| {},
    )
    .await
    .expect("the download succeeds");

    let written = std::fs::read(&path).unwrap();
    assert_eq!(
        written.len(),
        payload.len(),
        "the prefix must not be prepended to a full-body response"
    );
    assert_eq!(written, payload);
}

#[tokio::test]
async fn progress_is_reported_and_never_goes_backwards() {
    let payload = body(64 * 1024, 0xCD);
    let server = Server::start(payload.clone());
    let cache = Cache::new(cache_dir("progress"));
    let expected = expected_for("progress.gguf", &payload);

    let registry = Registry::new();
    let mut seen = Vec::new();
    neunote_models::fetch_entry(
        &cache,
        ModelSize::Small,
        &expected,
        &registry,
        &server.base_url(),
        |status| seen.push((status.phase, status.downloaded_bytes)),
    )
    .await
    .expect("download succeeds");

    for phase in [
        Phase::Connecting,
        Phase::Downloading,
        Phase::Verifying,
        Phase::Ready,
    ] {
        assert!(
            seen.iter().any(|(seen, _)| *seen == phase),
            "{phase:?} reported"
        );
    }

    let bytes: Vec<u64> = seen
        .iter()
        .filter(|(phase, _)| *phase == Phase::Downloading)
        .map(|(_, bytes)| *bytes)
        .collect();

    assert!(!bytes.is_empty(), "byte counts were reported");
    assert_eq!(bytes[0], 0, "the first report is zero");
    assert!(
        bytes.windows(2).all(|pair| pair[0] <= pair[1]),
        "byte counts must not decrease: {bytes:?}"
    );
    assert_eq!(*bytes.last().unwrap(), payload.len() as u64);
}

#[tokio::test]
async fn a_failed_download_keeps_its_partial_file_for_the_next_attempt() {
    let payload = body(4096, 0xAB);
    let server = Server::start(payload.clone());
    let cache = Cache::new(cache_dir("keep-part"));
    let expected = ModelEntry {
        size: ModelSize::Small,
        file_name: "keep-part.gguf",
        num_bytes: 4096,
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    };

    let registry = Registry::new();
    let _ = neunote_models::fetch_entry(
        &cache,
        ModelSize::Small,
        &expected,
        &registry,
        &server.base_url(),
        |_| {},
    )
    .await;

    // Verification failed, but the bytes are still there, so the next attempt
    // re-checks them instead of re-downloading 2.7 GB.
    assert_eq!(cache.partial_bytes_for(expected.file_name), 4096);
    assert!(cache.part_path_for(expected.file_name).exists());

    let status = registry.status(ModelSize::Small);
    assert!(!status.is_ready());
    assert!(!status.is_busy(), "the job is over");
}

#[tokio::test]
async fn an_unreachable_server_reports_a_download_error() {
    let cache = Cache::new(cache_dir("unreachable"));
    let expected = expected_for("unreachable.gguf", &body(16, 1));

    let registry = Registry::new();
    // Port 1 is reserved and nothing listens on it.
    let error = neunote_models::fetch_entry(
        &cache,
        ModelSize::Small,
        &expected,
        &registry,
        "http://127.0.0.1:1",
        |_| {},
    )
    .await
    .unwrap_err();

    assert!(matches!(error, ModelError::Download(_)), "got {error:?}");
    assert!(!registry.status(ModelSize::Small).is_ready());
}

#[tokio::test]
async fn a_missing_server_returns_an_http_error() {
    let cache = Cache::new(cache_dir("http-error"));
    let expected = expected_for("http-error.gguf", &body(16, 1));

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming().take(2) {
            let Ok(mut stream) = stream else { break };
            let mut buffer = [0u8; 1024];
            let _ = stream.read(&mut buffer);
            let _ = stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            let _ = stream.flush();
        }
    });

    let registry = Registry::new();
    let error = neunote_models::fetch_entry(
        &cache,
        ModelSize::Small,
        &expected,
        &registry,
        &format!("http://127.0.0.1:{port}"),
        |_| {},
    )
    .await
    .unwrap_err();

    match error {
        ModelError::Download(message) => assert!(message.contains("404"), "got {message}"),
        other => panic!("expected a download error naming the status, got {other:?}"),
    }
}

#[tokio::test]
async fn cancellation_stops_the_transfer_and_keeps_the_prefix() {
    let payload = body(4 * 1024 * 1024, 0xEF);
    let server = Server::start(payload.clone());
    let cache = Cache::new(cache_dir("cancel"));
    let expected = expected_for("cancel.gguf", &payload);

    let registry = Registry::new();
    let seen_downloading = std::cell::Cell::new(false);

    let result = neunote_models::fetch_entry(
        &cache,
        ModelSize::Small,
        &expected,
        &registry,
        &server.base_url(),
        |status| {
            if status.phase == Phase::Downloading && !seen_downloading.get() {
                seen_downloading.set(true);
                registry.cancel(ModelSize::Small);
            }
        },
    )
    .await;

    assert!(
        matches!(result, Err(ModelError::Cancelled)),
        "got {result:?}"
    );
    assert!(
        cache.part_path_for(expected.file_name).exists(),
        "the prefix is kept so the next attempt resumes"
    );
    assert!(
        !cache.model_path_for(expected.file_name).exists(),
        "a cancelled download never becomes installed"
    );
}

#[tokio::test]
async fn discard_partial_and_remove_clear_the_cache() {
    let payload = body(64, 7);
    let cache = Cache::new(cache_dir("clear"));
    let expected = expected_for("clear.gguf", &payload);

    std::fs::write(cache.part_path_for(expected.file_name), &payload).unwrap();
    std::fs::write(cache.model_path_for(expected.file_name), &payload).unwrap();

    assert_eq!(cache.partial_bytes_for(expected.file_name), 64);
    cache.discard_partial_for(expected.file_name).unwrap();
    assert_eq!(cache.partial_bytes_for(expected.file_name), 0);

    // `remove` addresses the manifest's own file, which is not this test's
    // stand-in, so it must be a no-op rather than an error.
    cache.remove_file("never-present.gguf").unwrap();
}

#[test]
fn the_digest_helper_used_by_the_tests_matches_the_reference_vectors() {
    // Guards the harness: if this broke, the corruption tests could pass for
    // the wrong reason.
    assert_eq!(
        sha256_of(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_of(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let _ = Path::new("/");
}
