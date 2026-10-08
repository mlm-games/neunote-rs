//! The ladder, steps 1 to 3: every stage of inference against tensors dumped
//! from the reference.
//!
//! `testdata/refs/small.bin` is produced by running `muscriptor.cpp` over
//! `testdata/audio/fixture_3chunks_16k.wav` with the `small` checkpoint. No
//! tolerance here was chosen to make the port pass: the bounds sit where the
//! reference's own numbers do, and a mismatch names the stage rather than the
//! harness.
//!
//! Steps 1 and 2 are where the architecture details in AGENTS.md get proven.
//! Step 3 is the one that matters: an exact token match means the whole forward
//! pass agrees, not merely that it is close.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use neunote_engine::{Model, Stop};

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the crate lives two levels below the workspace root")
        .to_path_buf()
}

fn refs_path() -> PathBuf {
    std::env::var_os("NEUNOTE_REF_DIR")
        .map_or_else(|| workspace().join("testdata/refs/small.bin"), |dir| {
            PathBuf::from(dir).join("small.bin")
        })
}

fn weights_path() -> PathBuf {
    std::env::var_os("NEUNOTE_WEIGHTS_DIR").map_or_else(
        || {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join(".local/share/neunote/models/muscriptor-small-f16.gguf")
        },
        |dir| PathBuf::from(dir).join("muscriptor-small-f16.gguf"),
    )
}

/// One named tensor in the dump, in ggml's shape order: `shape[0]` contiguous.
struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

struct Refs {
    tensors: HashMap<String, Tensor>,
}

impl Refs {
    fn load(path: &Path) -> Self {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));

        let mut at = 0usize;
        let mut take = |count: usize| -> &[u8] {
            let slice = &bytes[at..at + count];
            at += count;
            slice
        };

        assert_eq!(
            take(8),
            b"NNEEDMP\0",
            "not a neunote reference dump"
        );
        let count = u32::from_le_bytes(take(4).try_into().unwrap()) as usize;

        let mut tensors = HashMap::with_capacity(count);
        for _ in 0..count {
            let name_len = u32::from_le_bytes(take(4).try_into().unwrap()) as usize;
            let name = std::str::from_utf8(take(name_len)).unwrap().to_owned();

            let n_dims = u32::from_le_bytes(take(4).try_into().unwrap()) as usize;
            let shape: Vec<usize> = (0..n_dims)
                .map(|_| u64::from_le_bytes(take(8).try_into().unwrap()) as usize)
                .collect();

            let elements = u64::from_le_bytes(take(8).try_into().unwrap()) as usize;
            let data: Vec<f32> = take(elements * 4)
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect();

            assert_eq!(
                data.len(),
                shape.iter().product::<usize>(),
                "tensor '{name}' disagrees with its own shape"
            );
            tensors.insert(name, Tensor { shape, data });
        }

        assert_eq!(at, bytes.len(), "trailing bytes in the dump");
        Self { tensors }
    }

    fn get(&self, name: &str) -> &Tensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("the dump has no tensor named '{name}'"))
    }

    fn shape(&self, name: &str) -> &[usize] {
        &self.get(name).shape
    }

    fn data(&self, name: &str) -> &[f32] {
        &self.get(name).data
    }

    fn scalar(&self, name: &str) -> f32 {
        let tensor = self.get(name);
        assert_eq!(tensor.data.len(), 1, "'{name}' is not a scalar");
        tensor.data[0]
    }
}

struct Gap {
    max_abs: f32,
    rms: f32,
    cosine: f32,
    worst: usize,
    finite_mismatch: bool,
}

fn compare(got: &[f32], want: &[f32]) -> Gap {
    assert_eq!(got.len(), want.len(), "length mismatch");

    let mut max_abs = 0.0f32;
    let mut squares = 0.0f64;
    let mut dot = 0.0f64;
    let mut norm_got = 0.0f64;
    let mut norm_want = 0.0f64;
    let mut worst = 0usize;
    let mut finite_mismatch = false;

    for (index, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
        if a.is_finite() != b.is_finite() {
            finite_mismatch = true;
        }

        let abs = (a - b).abs();
        if abs > max_abs {
            max_abs = abs;
            worst = index;
        }

        squares += f64::from(a - b) * f64::from(a - b);
        dot += f64::from(a) * f64::from(b);
        norm_got += f64::from(a) * f64::from(a);
        norm_want += f64::from(b) * f64::from(b);
    }

    let denominator = norm_got.sqrt() * norm_want.sqrt();
    Gap {
        max_abs,
        rms: (squares / got.len() as f64).sqrt() as f32,
        cosine: if denominator > 0.0 {
            (dot / denominator) as f32
        } else {
            1.0
        },
        worst,
        finite_mismatch,
    }
}

/// Assert a tensor is close, naming where it is not so a failure points at a
/// stage rather than at the harness.
#[track_caller]
fn expect(name: &str, got: &[f32], refs: &Refs, abs: f32, cosine_min: f32) {
    let gap = compare(got, refs.data(name));

    assert!(
        !gap.finite_mismatch,
        "{name}: one side has a non-finite value where the other does not"
    );
    assert!(
        gap.max_abs <= abs,
        "{name}: max abs {:.4e} > {abs:.4e} \
         (worst index {}, got {:e}, want {:e}; rms {:.4e})",
        gap.max_abs,
        gap.worst,
        got[gap.worst],
        refs.data(name)[gap.worst],
        gap.rms
    );
    assert!(
        gap.cosine >= cosine_min,
        "{name}: cosine {:.9} < {cosine_min:.9} (rms {:.4e})",
        gap.cosine,
        gap.rms
    );
}

/// Transpose `[rows, columns]` into ggml's order, where `ne[0]` is contiguous.
///
/// The dump holds what the reference's tensors hold, and ggml keeps its first
/// axis contiguous. The port's activations are the other way round, so every
/// comparison transposes first. Comparing the two as they lie looks like a large
/// numeric disagreement and is not one.
fn to_ggml_order(activation: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    assert_eq!(activation.len(), rows * columns);
    let mut out = vec![0.0f32; activation.len()];
    for column in 0..columns {
        for row in 0..rows {
            out[row + rows * column] = activation[row * columns + column];
        }
    }
    out
}

/// The fixture, as 16 kHz mono f32.
///
/// The reference reads the same WAV with the same normalisation, so a difference
/// here is in the reader rather than in the model.
fn fixture() -> Vec<f32> {
    let path = workspace().join("testdata/audio/fixture_3chunks_16k.wav");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));

    let mut at = 12usize;
    let (mut format, mut channels, mut rate) = (0u16, 0u16, 0u32);
    let mut samples = Vec::new();

    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;

        if id == b"fmt " {
            format = u16::from_le_bytes(bytes[at + 8..at + 10].try_into().unwrap());
            channels = u16::from_le_bytes(bytes[at + 10..at + 12].try_into().unwrap());
            rate = u32::from_le_bytes(bytes[at + 12..at + 16].try_into().unwrap());
        } else if id == b"data" {
            let raw = &bytes[at + 8..at + 8 + size];
            samples = match format {
                1 => raw
                    .chunks_exact(2)
                    .map(|word| i16::from_le_bytes(word.try_into().unwrap()) as f32 / 32768.0)
                    .collect(),
                3 => raw
                    .chunks_exact(4)
                    .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                    .collect(),
                other => panic!("unsupported WAV format {other}"),
            };
        }

        at += 8 + size + (size & 1);
    }

    assert_eq!((channels, rate), (1, 16_000), "the fixture is mono 16 kHz");
    assert_eq!(samples.len(), 240_000, "15 s at 16 kHz");
    samples
}

fn first_chunk() -> Vec<f32> {
    fixture()[..80_000].to_vec()
}

fn model() -> Model {
    let path = weights_path();
    assert!(
        path.exists(),
        "the small checkpoint is not at {}.\n\
         Fetch it with `neunote models fetch --size small`, or point NEUNOTE_WEIGHTS_DIR at it.",
        path.display()
    );
    Model::load(&path).unwrap_or_else(|error| panic!("cannot load {}: {error}", path.display()))
}

fn refs() -> Refs {
    let path = refs_path();
    assert!(path.exists(), "the reference dump is not at {}", path.display());
    Refs::load(&path)
}

/// The dump was made from the reference's own `Hparams`, so reading them back is
/// part of the check: a port that guesses a dimension cannot pass this.
#[test]
fn the_metadata_agrees_with_the_reference() {
    let refs = refs();
    let hp = *model().hparams();

    assert_eq!(hp.dim as f32, refs.scalar("hparams.dim"));
    assert_eq!(hp.n_layer as f32, refs.scalar("hparams.n_layer"));
    assert_eq!(hp.n_head as f32, refs.scalar("hparams.n_head"));
    assert_eq!(hp.head_dim as f32, refs.scalar("hparams.head_dim"));
    assert_eq!(hp.ffn_dim as f32, refs.scalar("hparams.ffn_dim"));
    assert_eq!(hp.vocab_size as f32, refs.scalar("hparams.vocab_size"));
    assert_eq!(
        hp.initial_token_id as f32,
        refs.scalar("hparams.initial_token_id")
    );
    assert_eq!(
        hp.logit_mask_start as f32,
        refs.scalar("hparams.logit_mask_start")
    );
    assert_eq!(hp.layer_norm_eps, refs.scalar("hparams.layer_norm_epsilon"));
    assert_eq!(hp.max_period, refs.scalar("hparams.max_period"));
    assert_eq!(hp.n_fft as f32, refs.scalar("hparams.n_fft"));
    assert_eq!(hp.hop_length as f32, refs.scalar("hparams.hop_length"));
    assert_eq!(hp.n_mels as f32, refs.scalar("hparams.n_mels"));
    assert_eq!(hp.log_eps, refs.scalar("hparams.log_eps"));
    assert_eq!(hp.sample_rate as f32, refs.scalar("hparams.sample_rate"));
    assert_eq!(hp.frame_rate as f32, refs.scalar("hparams.frame_rate"));
}

/// Ladder step 1, part one: the STFT magnitudes.
///
/// 501 frames of 1025 bins. pffft and this port's radix-2 FFT sum in different
/// orders, so the bound is on the transform's own noise, not on bit-equality.
#[test]
fn step_1_the_stft_magnitudes_match() {
    let refs = refs();
    let model = model();
    let hp = model.hparams();

    let spectrum = neunote_engine::stft::magnitudes(
        &first_chunk(),
        hp.n_fft,
        hp.hop_length,
        model.stft_window(),
    )
    .expect("the fixture is long enough to reflect-pad");

    assert_eq!(spectrum.len(), 501 * 1025, "501 frames of 1025 bins");
    expect("spectrum", &spectrum, &refs, 1e-3, 0.999_999);
}

/// Ladder step 1, part two: every stage of the conditioning front-end.
///
/// This is where the details in AGENTS.md get proven: the checkpoint's
/// filterbank rather than a regenerated one, magnitudes rather than powers, the
/// log epsilon, and a mask whose length comes from the waveform, so the last of
/// the 501 frames comes out exactly zero rather than merely small.
///
/// Each stage is checked on its own, so a failure names the operation that
/// diverged rather than the front-end as a whole.
#[test]
fn step_1_the_conditioning_front_end_matches_stage_by_stage() {
    let refs = refs();
    let model = model();
    let chunk = first_chunk();
    let hp = *model.hparams();

    assert_eq!(hp.mel_frames(chunk.len()), 501, "80000 / 160 + 1");

    let spectrum =
        neunote_engine::stft::magnitudes(&chunk, hp.n_fft, hp.hop_length, model.stft_window())
            .expect("the fixture is long enough to reflect-pad");
    let stages = model
        .encode_conditioning_stages(&spectrum, chunk.len())
        .expect("encoding the fixture");

    let frames = 501;
    assert_eq!(stages.mel.len(), hp.n_mels * frames);
    assert_eq!(stages.embedding.len(), hp.dim * frames);

    // The filterbank comes from the checkpoint: each mel bin is a weighted sum
    // over one frame's spectrum bins.
    expect(
        "cond.mel",
        &to_ggml_order(&stages.mel, hp.n_mels, frames),
        &refs,
        1e-3,
        0.999_999,
    );

    // The log is natural, of `mel + eps`. Magnitudes rather than powers is what
    // makes this defined: a power would have taken some bins negative, and this
    // tensor has positive entries, so a power would have put NaNs where the
    // reference has real numbers.
    assert!(
        stages.logmel.iter().all(|value| value.is_finite()),
        "the log-mel is finite everywhere"
    );
    expect(
        "cond.logmel",
        &to_ggml_order(&stages.logmel, hp.n_mels, frames),
        &refs,
        1e-3,
        0.999_999,
    );

    expect(
        "cond.proj",
        &to_ggml_order(&stages.projection, hp.dim, frames),
        &refs,
        5e-3,
        0.999_999,
    );

    // The last frame is masked, and masked means exactly zero, not small. A frame
    // is a column of the port's `[dim, frames]` layout, so it is strided.
    let last = (0..hp.dim).map(|row| stages.embedding[row * frames + 500]);
    assert!(
        last.clone().all(|value| value == 0.0),
        "the frame past the audio must be exactly zero"
    );
    let mut inside = (0..hp.dim).map(|row| stages.embedding[row * frames + 499]);
    assert!(
        inside.any(|value| value != 0.0),
        "the last frame inside the audio must not be masked"
    );
    expect(
        "cond.embed",
        &to_ggml_order(&stages.embedding, hp.dim, frames),
        &refs,
        5e-3,
        0.999_999,
    );
}

/// The prefix's last four columns: the final mel frame, the dataset row, the
/// instrument row, the token.
///
/// The order is `[mel, dataset, instrument, tokens]`, the reverse of
/// `ConditioningProvider`'s iteration. Reading it the other way round still
/// produces plausible numbers, so it is checked rather than derived.
#[test]
fn step_2_the_prefix_order_is_mel_then_dataset_then_instrument_then_token() {
    let refs = refs();
    let mut model = model();
    let chunk = first_chunk();
    let initial = model.hparams().initial_token_id;

    let conditioning = model.encode_audio(&chunk).expect("encoding the fixture");
    model.prefill(&conditioning, &[initial]).expect("prefill");
    let hp = *model.hparams();

    let tail = refs.data("pre.prefix.tail");
    assert_eq!(refs.shape("pre.prefix.tail"), &[4, hp.dim]);

    // Four columns packed as `[4, dim]`: element (which, index) at
    // `which * dim + index`.
    let column = |which: usize, index: usize| tail[which * hp.dim + index];

    // The final mel frame is masked, and masked means exactly zero.
    assert!(
        (0..hp.dim).all(|index| column(0, index) == 0.0),
        "the last mel frame is masked in the prefix too"
    );

    // The next two columns are the two class conditioners, both on row 1.
    let dataset = refs.data("cond.dataset_name");
    let instrument = refs.data("cond.instrument_group");
    for (index, (row, group)) in dataset.iter().zip(instrument).enumerate() {
        assert!(
            (column(1, index) - row).abs() < 1e-6,
            "column 501 is the dataset row, not something else, at {index}"
        );
        assert!(
            (column(2, index) - group).abs() < 1e-6,
            "column 502 is the instrument row, not something else, at {index}"
        );
    }

    // And the last is the initial token's embedding row.
    let token = refs.data("pre.tok_embed");
    for (index, value) in token.iter().enumerate() {
        assert!(
            (column(3, index) - value).abs() < 1e-6,
            "column 503 is the token embedding, at {index}"
        );
    }
}

/// Ladder step 2: the prefill logits.
///
/// Cosine similarity and the top-1 id are the pair that matters. A port that is
/// close but not right can agree on one argmax and diverge on the next, so both
/// are asserted.
#[test]
fn step_2_the_prefill_logits_match() {
    let refs = refs();
    let mut model = model();
    let initial = model.hparams().initial_token_id;

    let conditioning = model.encode_audio(&first_chunk()).expect("encoding the fixture");
    let logits = model.prefill(&conditioning, &[initial]).expect("prefill");

    let hp = *model.hparams();
    let want = refs.data("prefill.logits_masked");
    assert_eq!(logits.len(), hp.vocab_size);
    assert_eq!(want.len(), hp.vocab_size);

    assert_eq!(
        argmax(&logits),
        argmax(want),
        "the argmax moved: the reference's top-1 was {}",
        argmax(want)
    );

    expect("prefill.logits_masked", &logits, &refs, 0.25, 0.999_9);
}

/// A conditioning row changes the prefix and therefore the decoding. It is not a
/// filter on a fixed prompt, so the two prefill results must differ.
#[test]
fn step_2_a_selected_instrument_changes_the_prefix_and_the_logits() {
    let refs = refs();
    let mut model = model();
    let conditioning = model.encode_audio(&first_chunk()).expect("encoding the fixture");
    let tokens = [model.hparams().initial_token_id];

    let unconditional = model.prefill(&conditioning, &tokens).expect("prefill");

    // Group 0 is row 2. The cache is cleared first: a second prefill without a
    // reset continues the sequence rather than starting a new one, and would
    // compare against the reference's fresh chunk.
    model.reset();
    model.set_instrument_rows(&[2]);
    assert_eq!(model.instrument_rows(), [2]);
    let piano = model.prefill(&conditioning, &tokens).expect("prefill");

    expect("prefill.piano.logits_masked", &piano, &refs, 0.25, 0.999_9);

    // The two must not be the same numbers. Whether the argmax moves is not
    // something to assert: on this fixture it does not, and that is a fact about
    // the fixture rather than about the conditioning.
    let shifted = piano
        .iter()
        .zip(&unconditional)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        shifted > 1.0,
        "one conditioning row changed the logits by at most {shifted}"
    );

    model.reset();
    model.set_instrument_rows(&[]);
    assert_eq!(
        model.instrument_rows(),
        [neunote_engine::model::NULL_CONDITIONING_ROW],
        "an empty selection restores the unconditional path"
    );
}

/// Ladder step 3: greedy tokens for one chunk.
///
/// The token stream is the whole forward pass turned into a decision. Exact
/// equality is the claim worth making, and it is why steps 1 and 2 exist.
#[test]
fn step_3_the_greedy_tokens_match_exactly() {
    let refs = refs();
    let mut model = model();

    let conditioning = model
        .encode_audio(&first_chunk())
        .expect("encoding the fixture");
    let result = model
        .generate(&conditioning, &[], 2_000, 1)
        .expect("generating a chunk");

    let want: Vec<i32> = refs
        .data("generate.tokens")
        .iter()
        .map(|value| *value as i32)
        .collect();

    if result.tokens != want {
        let at = result
            .tokens
            .iter()
            .zip(&want)
            .position(|(a, b)| a != b)
            .unwrap_or(result.tokens.len().min(want.len()));
        panic!(
            "the token streams differ at {at}: {} tokens against the reference's {}, \
             first divergence {:?} vs {:?}, and the lengths are {} and {}",
            result.tokens.len(),
            want.len(),
            result.tokens.get(at),
            want.get(at),
            result.tokens.len(),
            want.len()
        );
    }

    assert_eq!(result.stop, Stop::Eos, "the reference stopped on EOS");
}

/// Ladder step 2, stage by stage: layer 0's intermediates.
///
/// The prefill logits say "the transformer is wrong". These say *where*. The
/// divergence grows down the layer -- a few times 1e-6 at the input, a few times
/// 1e-3 at the output -- which is fourteen layers' worth of f32 accumulation
/// order, and nothing structural.
#[test]
fn step_2_layer_zero_matches_stage_by_stage() {
    let refs = refs();
    let mut model = model();
    let hp = *model.hparams();
    let n_new = 501 + 1 + 1 + 1;

    let conditioning = model.encode_audio(&first_chunk()).expect("encoding the fixture");
    let mut trace = neunote_engine::model::Trace::default();
    let logits = model
        .prefill_traced(&conditioning, &[hp.initial_token_id], &mut trace)
        .expect("prefill");

    let stages = [
        ("pre.layer_in", hp.dim),
        ("blk.0.norm1", hp.dim),
        ("blk.0.attn_ctx", hp.dim),
        ("blk.0.attn_out", hp.dim),
        ("blk.0.res1", hp.dim),
        ("blk.0.norm2", hp.dim),
        ("blk.0.ffn_pre_gelu", hp.ffn_dim),
        ("blk.0.ffn_gelu", hp.ffn_dim),
        ("blk.0.ffn_out", hp.dim),
        ("blk.0.out", hp.dim),
    ];

    // The dump keeps eight columns of each, packed as `[8, rows]`.
    let kept = 8;
    for (name, rows) in stages {
        assert_eq!(refs.shape(name), &[kept, rows], "{name} shape");

        let got = trace
            .read(name)
            .unwrap_or_else(|| panic!("{name} was not captured"));
        assert_eq!(got.len(), rows * n_new, "{name} length");

        let mut port = vec![0.0f32; rows * kept];
        for column in 0..kept {
            for row in 0..rows {
                port[column * rows + row] = got[row * n_new + column];
            }
        }

        expect(name, &port, &refs, 5e-3, 0.999_999);
    }

    expect("post.logits_raw", &logits, &refs, 0.05, 0.999_999);
}

/// The sinusoidal position table, checked against the reference's own rows.
///
/// Cosine before sine, and the exponent denominator is `half - 1`: either detail
/// wrong still produces a table that looks like positions.
#[test]
fn the_position_table_matches() {
    let refs = refs();
    let model = model();
    let hp = model.hparams();

    let want = refs.data("position_table");
// The table is `[position, dim]`: positions are rows, and the dump stores them
    // the same way round.
    assert_eq!(refs.shape("position_table"), &[16, hp.dim]);

    let got = neunote_engine::model::position_table(16, hp.dim, hp.max_period);
    assert!(
        got.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()),
        "the position table is built differently"
    );
}

/// The first maximum, matching the reference's `max_element`: ties go to the
/// lowest id.
fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    best as i32
}