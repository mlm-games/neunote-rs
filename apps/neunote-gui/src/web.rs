//! The web shell: a file input for audio and for the checkpoint, and a
//! download for the MIDI.

use std::rc::Rc;

use neunote_types::ModelSize;
use neunote_ui::{LoadedAudio, LoadedWeights, Shell, Transport};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{Blob, File, FileList, HtmlElement, HtmlInputElement, Url};

pub(crate) fn shell() -> Shell {
    // cpal's audioworklet host is the browser's output device, so playback is
    // the same cpal stream the desktop host opens.
    let transport = crate::audio::Device::open().map(|device| Rc::new(device) as Rc<dyn Transport>);

    Shell {
        transport,
        // A browser tab has no model cache: the checkpoint lives in OPFS, and
        // `resolve_weights` reads it back from there.
        cached_weights: Rc::new(|_size: ModelSize| None),

        resolve_weights: Rc::new(crate::web_weights::resolve),

        fetch_weights: Rc::new(|size, accepted, progress, done| {
            crate::web_weights::fetch(size, accepted, progress, done)
        }),

        copy_midi: Rc::new(|_name, _bytes| {
            Err(String::from(
                "a browser cannot hand a MIDI file to a DAW -- use Save MIDI",
            ))
        }),

        licence_accepted: Rc::new(crate::web_weights::licence_accepted),

        set_licence_accepted: Rc::new(crate::web_weights::record_acceptance),

        pick_audio: Rc::new(|done| {
            let picked = {
                let done = done.clone();
                move |file: Option<File>| {
                    let Some(file) = file else {
                        done(None);
                        return;
                    };
                    let done = done.clone();
                    let name = file.name();
                    read_file(file, move |bytes| done(Some(LoadedAudio { name, bytes })));
                }
            };
            input("audio/*,.wav,.flac,.mp3,.ogg,.oga,.opus,.aac,.m4a", picked);
        }),

        pick_weights: Rc::new(|done| {
            let picked = {
                let done = done.clone();
                move |file: Option<File>| {
                    let Some(file) = file else {
                        done(None);
                        return;
                    };
                    let done = done.clone();
                    let name = file.name();
                    read_file(file, move |bytes| done(Some(LoadedWeights { name, bytes })));
                }
            };
            input(".gguf", picked);
        }),

        save_midi: Rc::new(|name, bytes, done| {
            download(name, bytes);
            done(Ok(format!("downloaded {name}")));
        }),
    }
}

/// A hidden file input that hands the chosen file to `picked` and then takes
/// itself back out of the document.
fn input(accept: &str, picked: impl Fn(Option<File>) + 'static) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };

    let Ok(element) = document.create_element("input") else {
        return;
    };
    let Ok(element) = element.dyn_into::<HtmlInputElement>() else {
        return;
    };
    element.set_type("file");
    element.set_accept(accept);
    element.set_attribute("hidden", "").ok();

    let source = element.clone();
    let picked = Closure::once(move |_event: web_sys::Event| {
        let file: Option<File> = source.files().and_then(|files: FileList| files.get(0));

        if let Some(node) = source.dyn_ref::<web_sys::Node>()
            && let Some(parent) = node.parent_node()
        {
            let _ = parent.remove_child(node);
        }

        picked(file);
    });

    element.set_onchange(Some(picked.as_ref().unchecked_ref()));
    if let Some(body) = document.body() {
        let _ = body.append_child(&element);
    }
    element.click();

    // The browser owns the callback until it fires; nothing else can drop it.
    picked.forget();
}

fn read_file(file: File, done: impl FnOnce(Vec<u8>) + 'static) {
    let on_bytes = Closure::once(move |buffer: JsValue| {
        done(js_sys::Uint8Array::new(&buffer).to_vec());
    });
    let _ = file.array_buffer().then(&on_bytes);
    on_bytes.forget();
}

fn download(name: &str, bytes: &[u8]) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };

    let array = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::new();
    parts.push(&array.buffer());

    let Ok(blob) = Blob::new_with_u8_array_sequence(&parts) else {
        return;
    };
    let Ok(url) = Url::create_object_url_with_blob(&blob) else {
        return;
    };

    if let Ok(element) = document.create_element("a")
        && let Ok(anchor) = element.clone().dyn_into::<HtmlElement>()
    {
        let _ = anchor.set_attribute("href", &url);
        let _ = anchor.set_attribute("download", name);
        anchor.click();
    }

    // The download reads the blob asynchronously, so let the click land first.
    let release = Closure::wrap(Box::new(move || {
        Url::revoke_object_url(&url).ok();
    }) as Box<dyn FnMut()>);
    let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
        release.as_ref().unchecked_ref(),
        1_000,
    );
    release.forget();
}
