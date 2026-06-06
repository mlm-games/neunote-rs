fn main() {
    println!("NeuralNote Core v0.1.0");
    println!("Audio-to-MIDI transcription engine in Rust");
    println!();
    println!("Usage: Add this crate as a dependency and use BasicPitch API.");
    println!();
    println!("Example:");
    println!("  let mut bp = BasicPitch::new(..);");
    println!("  bp.transcribe(&audio);");
    println!("  for e in bp.note_events() {{");
    println!("      println!(\"{{:?}}\", e);");
    println!("  }}");
}
