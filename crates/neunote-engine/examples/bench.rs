//! Wall-clock timing for one chunk of inference, phase by phase.
//!
//! ```bash
//! cargo run --release -p neunote-engine --example bench [decode steps]
//! ```
//!
//! The audio is synthetic: the arithmetic is content-independent, so a fixed
//! number of decode steps says what a real chunk costs without needing the
//! fixture.

use std::path::PathBuf;
use std::time::Instant;

use neunote_engine::Model;

fn weights_path() -> PathBuf {
    let dir = std::env::var_os("NEUNOTE_WEIGHTS_DIR").map_or_else(
        || {
            let home = std::env::var_os("HOME").expect("HOME");
            PathBuf::from(home).join(".local/share/neunote/models")
        },
        PathBuf::from,
    );
    dir.join("muscriptor-small-f16.gguf")
}

fn chunk() -> Vec<f32> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    (0..80_000)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state as f32 / u64::MAX as f32 - 0.5) * 0.4;
            let time = index as f32 / 16_000.0;
            let tone = 0.3 * (2.0 * std::f32::consts::PI * 220.0 * time).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 523.25 * time).sin();
            noise + tone
        })
        .collect()
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    best as i32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let steps: usize = std::env::args()
        .nth(1)
        .map(|arg| arg.parse().expect("decode steps"))
        .unwrap_or(456);

    let path = weights_path();
    let mut model = Model::load(&path)?;
    let hp = *model.hparams();
    let samples = chunk();

    let started = Instant::now();
    let conditioning = model.encode_audio(&samples)?;
    let front = started.elapsed();

    model.reset();
    let started = Instant::now();
    let mut logits = model.prefill(&conditioning, &[hp.initial_token_id])?;
    let prefill = started.elapsed();

    let started = Instant::now();
    for _ in 1..steps {
        let next = argmax(&logits);
        logits = model.decode(next)?;
    }
    let decode = started.elapsed();

    let layers = hp.n_layer as f64;
    let dim = hp.dim as f64;
    let prefill_columns = (samples.len() / hp.hop_length + 1 + 3) as f64;
    let matmul_per_layer = 8.0 * dim * dim + 4.0 * dim * hp.ffn_dim as f64;
    let prefill_gflop = layers * matmul_per_layer * prefill_columns * 2.0 / 1e9;
    // The head is one column, the attention cost grows with the cache, and the
    // front-end is a rounding error; the four layer matmuls are what a decode
    // step spends its time on.
    let decode_gflop = layers * matmul_per_layer * 2.0 * steps as f64 / 1e9;

    println!("front-end   {:>8.1} ms", front.as_secs_f64() * 1e3);
    println!(
        "prefill     {:>8.1} ms   {prefill_gflop:.2} GFLOP -> {:.1} GFLOP/s",
        prefill.as_secs_f64() * 1e3,
        prefill_gflop / prefill.as_secs_f64()
    );
    println!(
        "decode      {:>8.1} ms   {decode_gflop:.2} GFLOP -> {:.1} GFLOP/s over {steps} steps ({:.2} ms/step)",
        decode.as_secs_f64() * 1e3,
        decode_gflop / decode.as_secs_f64(),
        decode.as_secs_f64() * 1e3 / steps as f64
    );
    println!(
        "chunk total {:>8.1} ms for {} s of audio",
        (front + prefill + decode).as_secs_f64() * 1e3,
        samples.len() as f64 / hp.sample_rate as f64
    );

    kernels(hp.dim);

    Ok(())
}

/// One matmul, on its own: how wide `columns` is decides which kernel runs,
/// and whether the weight is f16 decides whether conversion is in the loop.
fn kernels(dim: usize) {
    use neunote_engine::gguf::Weight;

    let rows = 3 * dim;
    let narrow = vec![0.0f32; rows];
    let wide = vec![0.0f32; rows * 505];
    let one = vec![0.0f32; dim];
    let many: Vec<f32> = (0..dim * 505).map(|i| (i % 101) as f32 / 101.0).collect();

    println!(
        "\n{} threads, {rows} x {dim} matmul, 100 rounds",
        rayon::current_num_threads()
    );

    for kind in ["f16", "f32"] {
        let weight = if kind == "f16" {
            Weight::F16(
                (0..rows * dim)
                    .map(|i| half::f16::from_f32((i % 97) as f32 / 97.0))
                    .collect(),
            )
        } else {
            Weight::F32((0..rows * dim).map(|i| (i % 97) as f32 / 97.0).collect())
        };

        let started = Instant::now();
        for _ in 0..100 {
            neunote_engine::ops::matmul(&mut narrow.clone(), &weight, &one, 1, None);
        }
        let step = started.elapsed() / 100;

        let started = Instant::now();
        for _ in 0..10 {
            neunote_engine::ops::matmul(&mut wide.clone(), &weight, &many, 505, None);
        }
        let prefill = started.elapsed() / 10;

        let gflop = rows as f64 * dim as f64 * 2.0 / 1e9;
        println!(
            "  {kind}: columns=1 {:>9.3} ms  ({:>6.2} GFLOP/s)   columns=505 {:>8.2} ms  ({:>6.1} GFLOP/s)",
            step.as_secs_f64() * 1e3,
            gflop / step.as_secs_f64(),
            prefill.as_secs_f64() * 1e3,
            gflop * 505.0 / prefill.as_secs_f64()
        );
    }
}
