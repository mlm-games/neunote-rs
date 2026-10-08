fn main() {
    #[cfg(all(not(target_os = "android"), not(target_arch = "wasm32")))]
    if let Err(error) = neunote_gui::desktop_main() {
        eprintln!("neunote: {error}");
        std::process::exit(1);
    }
}
