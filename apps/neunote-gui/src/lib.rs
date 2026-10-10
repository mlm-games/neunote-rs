#![deny(unsafe_code)]

//! The neunote shells: one view tree, three platforms.
//!
//! The desktop shell answers the view's file requests with a dialog and the
//! filesystem; the Android shell with the system picker and the app's own
//! storage; the web shell with a file input and a download. Everything between
//! them -- state, the job, the piano roll -- is `neunote-ui`.

#[cfg(target_os = "android")]
mod android;
#[cfg(not(target_arch = "wasm32"))]
mod audio;
#[cfg(all(not(target_os = "android"), not(target_arch = "wasm32")))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
mod web_weights;

use neunote_ui::root;

#[cfg(all(not(target_os = "android"), not(target_arch = "wasm32")))]
pub fn desktop_main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::builder()
        .parse_env("RUST_LOG")
        .filter_level(log::LevelFilter::Info)
        .init();

    Ok(repose_platform::run_desktop_app_with_config(
        root(native::shell()),
        repose_platform::AppConfig {
            window_title: String::from("neunote"),
            // The toolbar, the quantise panel and the footer all want width,
            // and 1280x800 px is 1024 dp at a 1.25 scale.
            window_size: (1600, 1000),
            ..Default::default()
        },
    )?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn wasm_start() -> Result<(), wasm_bindgen::JsValue> {
    let mut options = repose_platform::web::WebOptions::new(None);
    options.set_prevent_default(true);
    repose_platform::web::run_web_app(root(web::shell()), options)
}

/// The NDK looks this up by name when the activity starts, so the export needs
/// the unsafe attribute; the function itself only calls safe code.
#[cfg(target_os = "android")]
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn android_main(app: winit::platform::android::activity::AndroidApp) {
    rlobkit_app_events::android_log::init(
        env!("CARGO_PKG_NAME"),
        concat!(env!("CARGO_PKG_NAME"), "=info"),
    );

    if let Some(dir) = app.internal_data_path() {
        game_utils::set_android_data_dir(dir);
    }

    rlobkit_dialogs::init_shared_pending_state();
    rlobkit_dialogs::init_with_android_context(
        app.vm_as_ptr().cast(),
        app.activity_as_ptr().cast(),
    );

    rlobkit_app_events::insets::set_on_insets(|insets| {
        repose_core::locals::set_window_insets_default(repose_core::locals::WindowInsets {
            top: insets.top,
            bottom: insets.bottom,
            left: insets.left,
            right: insets.right,
            ime_bottom: insets.ime_bottom,
        });
    });

    rlobkit_app_events::theme::set_on_theme(|_| repose_platform::wake_event_loop());
    rlobkit_app_events::back::set_on_back(|| {
        rlobkit_app_events::back::BackOutcome::PropagateToSystem
    });

    let pending = android::Pending::new();
    let shell = android::shell(&pending);
    let pump = pending.clone();
    let mut ui = root(shell);

    if let Err(error) = repose_platform::android::run_android_app_with_options(
        app,
        move |scheduler, context| {
            pump.pump();
            ui(scheduler, context)
        },
        repose_platform::android::AndroidOptions::default(),
    ) {
        log::error!("neunote failed: {error:?}");
    }
}
