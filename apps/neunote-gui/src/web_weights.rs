//! The browser's checkpoint store: OPFS, with the same digest pin the CLI has.
//!
//! Nothing is fetched until the user accepts the weights' licence, the bytes
//! land in OPFS under the manifest's name, and the file counts as usable only
//! once its SHA-256 matches the compiled-in pin. A browser tab can hold `small`;
//! `medium` and `large` are refused rather than downloaded into a heap that
//! would die.

use std::rc::Rc;

use neunote_models::manifest::{self, ModelEntry};
use neunote_types::ModelSize;
use sha2::{Digest, Sha256};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    FileSystemDirectoryHandle, FileSystemFileHandle, Request, RequestInit, RequestMode, Response,
};

/// Above this a browser tab is the wrong place for a checkpoint: the bytes have
/// to fit in wasm memory alongside the activations.
const BROWSER_LIMIT: u64 = 400 * 1024 * 1024;

const ACCEPTED_KEY: &str = "neunote:weights-licence-accepted";

type Resolved = Rc<dyn Fn(Result<Rc<Vec<u8>>, String>)>;
type Progress = Rc<dyn Fn(u64, u64)>;

pub(crate) fn licence_accepted() -> bool {
    window_storage()
        .and_then(|storage| storage.get_item(ACCEPTED_KEY).ok().flatten())
        .is_some()
}

pub(crate) fn record_acceptance() {
    if let Some(storage) = window_storage() {
        let _ = storage.set_item(ACCEPTED_KEY, neunote_models::WEIGHTS_LICENSE);
    }
}

/// Hand back a checkpoint this browser already holds.
pub(crate) fn resolve(size: ModelSize, done: Resolved) {
    let entry = manifest::entry(size);
    let name = entry.file_name.to_owned();

    with_models_dir(move |dir| {
        then(
            dir.get_file_handle(&name),
            move |value| match value.dyn_into::<FileSystemFileHandle>() {
                Ok(handle) => read_bytes(handle, done),
                Err(_) => done(Err(format!(
                    "{} is not in this browser yet -- download it once and it stays.",
                    entry.file_name
                ))),
            },
        );
    });
}

/// Fetch the checkpoint, check it against the pin, and keep it in OPFS.
pub(crate) fn fetch(size: ModelSize, progress: Progress, done: Resolved) {
    let entry = manifest::entry(size);

    if entry.num_bytes > BROWSER_LIMIT {
        done(Err(format!(
            "{} is {} MB: too big for a browser tab. Use the desktop `neunote`, or pick the file yourself.",
            entry.file_name,
            entry.num_bytes / (1024 * 1024)
        )));
        return;
    }

    if !licence_accepted() {
        done(Err(format!(
            "the weights are {} -- accept that before downloading.",
            neunote_models::WEIGHTS_LICENSE
        )));
        return;
    }

    let Some(window) = web_sys::window() else {
        done(Err(String::from("no window to fetch from")));
        return;
    };

    let init = RequestInit::new();
    init.set_mode(RequestMode::Cors);
    let request = match Request::new_with_str_and_init(&manifest::resolve_url(entry), &init) {
        Ok(request) => request,
        Err(error) => {
            done(Err(format!("cannot ask for the weights: {error:?}")));
            return;
        }
    };

    let total = entry.num_bytes;
    then(window.fetch_with_request(&request), move |value| {
        let Ok(response) = value.dyn_into::<Response>() else {
            done(Err(String::from("the fetch did not answer")));
            return;
        };

        if !response.ok() {
            done(Err(format!(
                "the server answered {} for {}",
                response.status(),
                entry.file_name
            )));
            return;
        }

        progress(0, total);

        then(
            response.array_buffer().expect("a Response can be buffered"),
            move |value| {
                let Ok(buffer) = value.dyn_into::<js_sys::ArrayBuffer>() else {
                    done(Err(String::from("the response carried no bytes")));
                    return;
                };

                let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
                progress(bytes.len() as u64, total);

                let digest = hex::encode(Sha256::digest(&bytes));
                if digest != entry.sha256 {
                    done(Err(format!(
                        "{} hashed to {digest}, not the pinned digest. Nothing was kept.",
                        entry.file_name
                    )));
                    return;
                }

                store(entry, bytes, done);
            },
        );
    });
}

/// Write the verified bytes into OPFS, then read them back the way a run will.
fn store(entry: &'static ModelEntry, bytes: Vec<u8>, done: Resolved) {
    let name = entry.file_name.to_owned();

    with_models_dir(move |dir| {
        let options = web_sys::FileSystemGetFileOptions::new();
        options.set_create(true);

        then(
            dir.get_file_handle_with_options(&name, &options),
            move |value| {
                let Ok(handle) = value.dyn_into::<FileSystemFileHandle>() else {
                    done(Err(String::from("the browser would not create the file")));
                    return;
                };

                then(handle.create_writable(), move |value| {
                    let Ok(stream) = value.dyn_into::<web_sys::FileSystemWritableFileStream>()
                    else {
                        done(Err(String::from("the file would not open for writing")));
                        return;
                    };

                    let array = js_sys::Uint8Array::from(bytes.as_slice());
                    then(
                        stream
                            .write_with_buffer_source(array.as_ref())
                            .expect("a writable stream accepts bytes"),
                        move |_| {
                            then(stream.close(), move |_| read_back(entry, done));
                        },
                    );
                });
            },
        );
    });
}

fn read_back(entry: &'static ModelEntry, done: Resolved) {
    let name = entry.file_name.to_owned();

    with_models_dir(move |dir| {
        then(
            dir.get_file_handle(&name),
            move |value| match value.dyn_into::<FileSystemFileHandle>() {
                Ok(handle) => read_bytes(handle, done),
                Err(_) => done(Err(String::from(
                    "the download vanished before it could be read",
                ))),
            },
        );
    });
}

fn read_bytes(handle: FileSystemFileHandle, done: Resolved) {
    then(handle.get_file(), move |value| {
        let file: web_sys::File = match value.dyn_into() {
            Ok(file) => file,
            Err(_) => {
                done(Err(String::from("the stored checkpoint is not a file")));
                return;
            }
        };

        then(
            file.array_buffer(),
            move |value| match value.dyn_into::<js_sys::ArrayBuffer>() {
                Ok(buffer) => done(Ok(Rc::new(js_sys::Uint8Array::new(&buffer).to_vec()))),
                Err(_) => done(Err(String::from("the stored checkpoint is unreadable"))),
            },
        );
    });
}

fn window_storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

fn with_models_dir(then_dir: impl FnOnce(FileSystemDirectoryHandle) + 'static) {
    let Some(storage) = web_sys::window().map(|window| window.navigator().storage()) else {
        return;
    };

    then(storage.get_directory(), move |root| {
        let Ok(root) = root.dyn_into::<FileSystemDirectoryHandle>() else {
            return;
        };

        let options = web_sys::FileSystemGetDirectoryOptions::new();
        options.set_create(true);
        then(
            root.get_directory_handle_with_options("models", &options),
            move |value| {
                if let Ok(dir) = value.dyn_into::<FileSystemDirectoryHandle>() {
                    then_dir(dir);
                }
            },
        );
    });
}

/// Resolve a promise, handing the callback `undefined` when it rejects -- a
/// missing file is a rejection, not an exception.
fn then(promise: js_sys::Promise, map: impl FnOnce(JsValue) + 'static) {
    let map = Rc::new(std::cell::RefCell::new(Some(map)));
    let on_reject = Rc::clone(&map);

    let fulfilled = Closure::once(move |value: JsValue| {
        if let Some(map) = map.borrow_mut().take() {
            map(value);
        }
    });
    let rejected = Closure::once(move |_error: JsValue| {
        if let Some(map) = on_reject.borrow_mut().take() {
            map(JsValue::UNDEFINED);
        }
    });

    let _ = promise.then(&fulfilled).catch(&rejected);
    fulfilled.forget();
    rejected.forget();
}
