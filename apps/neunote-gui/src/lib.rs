#![forbid(unsafe_code)]

//! The neunote shells: one view tree, two platforms.
//!
//! The desktop shell answers the view's file requests with a dialog and the
//! filesystem; the web shell with a file input and a download. Everything
//! between them -- state, the job, the piano roll -- is `neunote-ui`.

#[cfg(all(not(target_os = "android"), not(target_arch = "wasm32")))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;

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
