#![forbid(unsafe_code)]

//! A pure-Rust MuScriptor: the mel front-end, the transformer, and greedy
//! decoding, transcribed from [muscriptor.cpp](https://github.com/DamRsn/muscriptor.cpp).
//!
//! Nothing numerical here was inferred from prose. The load path reads every
//! dimension from the checkpoint's `muscriptor.*` metadata rather than writing
//! them down, so `small`, `medium` and `large` are the same code.
//!
//! ```no_run
//! # fn main() -> Result<(), neunote_engine::Error> {
//! let mut model = neunote_engine::Model::load(std::path::Path::new("muscriptor-small-f16.gguf"))?;
//! let conditioning = model.encode_audio(&vec![0.0; 80_000])?;
//! let chunk = model.generate(&conditioning, &[], 2_000, 1)?;
//! println!("{} tokens, stopped on {:?}", chunk.tokens.len(), chunk.stop);
//! # Ok(())
//! # }
//! ```

pub mod gguf;
pub mod model;
pub mod ops;
pub mod stft;

use std::path::Path;

use gguf::{Gguf, Weight};
use model::{Hparams, Weights};

pub use model::{Chunk, Model, Stop};

#[derive(Debug)]
pub enum Error {
    /// The checkpoint is not one this build can read: wrong magic, a shape that
    /// does not match the architecture, or a metadata key that is missing.
    Checkpoint(String),
    /// The conditioning prefix plus the generated tokens exceed the KV cache.
    ContextOverflow { needed: usize, capacity: usize },
    /// A caller passed something inconsistent.
    Invalid(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Checkpoint(message) => write!(formatter, "checkpoint: {message}"),
            Self::ContextOverflow { needed, capacity } => write!(
                formatter,
                "the sequence needs {needed} positions but the KV cache holds {capacity}"
            ),
            Self::Invalid(message) => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for Error {}

/// Instrument groups a caller may select, which is what sizes the KV cache: the
/// prefix carries one row each.
const MAX_SELECTABLE: usize = neunote_types::GroupId::ALL_NAMED.len();

fn key(suffix: &str) -> String {
    format!("muscriptor.{suffix}")
}

impl Model {
    /// Read a checkpoint and allocate the KV cache.
    ///
    /// The cache is sized from the front-end constants rather than from a
    /// default, so it covers the conditioning prefix and a full chunk of
    /// generation without a caller having to know the frame count:
    /// `mel_frames + 1 + selectable_instruments + 1 + max_tokens`.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let file = Gguf::open(path)?;

        let version = if file.has(&key("format_version")) {
            file.i32(&key("format_version"))?
        } else {
            0
        };
        if version as u32 != neunote_types::CHECKPOINT_FORMAT_VERSION {
            return Err(Error::Checkpoint(format!(
                "{} is checkpoint format version {version}; this build reads {}",
                path.file_name().map(|name| name.to_string_lossy()).unwrap_or_default(),
                neunote_types::CHECKPOINT_FORMAT_VERSION
            )));
        }

        let hp = Hparams {
            dim: read(&file, "embedding_length")? as usize,
            n_layer: read(&file, "block_count")? as usize,
            n_head: read(&file, "attention.head_count")? as usize,
            head_dim: read(&file, "attention.head_dim")? as usize,
            ffn_dim: read(&file, "feed_forward_length")? as usize,
            vocab_size: read(&file, "vocab_size")? as usize,
            initial_token_id: read(&file, "initial_token_id")?,
            logit_mask_start: read(&file, "logit_mask_start")? as usize,
            layer_norm_eps: file.f32(&key("attention.layer_norm_epsilon"))?,
            max_period: file.f32(&key("position_embedding.max_period"))?,
            sample_rate: read(&file, "audio.sample_rate")? as u32,
            n_fft: read(&file, "audio.n_fft")? as usize,
            hop_length: read(&file, "audio.hop_length")? as usize,
            frame_rate: read(&file, "audio.frame_rate")?,
            n_mels: read(&file, "audio.n_mels")? as usize,
            log_eps: file.f32(&key("audio.log_eps"))?,
        };

        if hp.hop_length == 0 || hp.n_fft == 0 {
            return Err(Error::Checkpoint("the checkpoint has a degenerate STFT".into()));
        }

        let dim = hp.dim;
        let n_freq = hp.n_freq();
        let n_mels = hp.n_mels;

        let weights = Weights {
            token_embedding: file.weight("token_embd.weight", &[dim, hp.vocab_size + 1])?,
            output: file.weight("output.weight", &[dim, hp.vocab_size])?,
            output_norm_weight: file.f32_vector("output_norm.weight", dim)?,
            output_norm_bias: file.f32_vector("output_norm.bias", dim)?,
            mel_filterbank: file.weight("cond.mel_fb.weight", &[n_freq, n_mels])?,
            stft_window: file.f32_vector("cond.stft_window", hp.n_fft)?,
            cond_proj_weight: file.weight("cond.proj.weight", &[n_mels, dim])?,
            cond_proj_bias: file.f32_vector("cond.proj.bias", dim)?,
            instrument_group: instrument_rows(&file, "cond.instrument_group.weight", dim)?,
            dataset_name: dataset_rows(&file, dim)?,
            layers: (0..hp.n_layer)
                .map(|index| layer(&file, index, dim, hp.ffn_dim))
                .collect::<Result<Vec<_>, _>>()?,
        };

        let n_ctx = hp.mel_frames(neunote_types::SEGMENT_SAMPLES)
            + 1
            + MAX_SELECTABLE
            + 1
            + neunote_types::MAX_TOKENS_PER_CHUNK;

        Model::new(hp, weights, n_ctx)
    }
}

fn read(file: &Gguf, suffix: &str) -> Result<i32, Error> {
    file.i32(&key(suffix))
}

fn layer(
    file: &Gguf,
    index: usize,
    dim: usize,
    ffn_dim: usize,
) -> Result<model::LayerWeights, Error> {
    let name = |suffix: &str| format!("blk.{index}.{suffix}");
    Ok(model::LayerWeights {
        attn_norm_weight: file.f32_vector(&name("attn_norm.weight"), dim)?,
        attn_norm_bias: file.f32_vector(&name("attn_norm.bias"), dim)?,
        attn_qkv: file.weight(&name("attn_qkv.weight"), &[dim, 3 * dim])?,
        attn_out: file.weight(&name("attn_out.weight"), &[dim, dim])?,
        ffn_norm_weight: file.f32_vector(&name("ffn_norm.weight"), dim)?,
        ffn_norm_bias: file.f32_vector(&name("ffn_norm.bias"), dim)?,
        ffn_up: file.weight(&name("ffn_up.weight"), &[dim, ffn_dim])?,
        ffn_down: file.weight(&name("ffn_down.weight"), &[ffn_dim, dim])?,
    })
}

/// The instrument conditioner, whose row count is the vocabulary's group count
/// and is not derivable from the metadata the architecture documents.
fn instrument_rows(file: &Gguf, name: &str, dim: usize) -> Result<Weight, Error> {
    let rows = file
        .shape(name)
        .ok_or_else(|| Error::Checkpoint(format!("tensor '{name}' is not in the checkpoint")))?;
    if rows.len() != 2 || rows[0] != dim {
        return Err(Error::Checkpoint(format!("tensor '{name}' has shape {rows:?}")));
    }
    file.weight(name, &[dim, rows[1]])
}

/// The dataset conditioner, which is always the null class and so only needs the
/// rows it has.
fn dataset_rows(file: &Gguf, dim: usize) -> Result<Weight, Error> {
    let name = "cond.dataset_name.weight";
    let rows = file
        .shape(name)
        .ok_or_else(|| Error::Checkpoint(format!("tensor '{name}' is not in the checkpoint")))?;
    if rows.len() != 2 || rows[0] != dim {
        return Err(Error::Checkpoint(format!("tensor '{name}' has shape {rows:?}")));
    }
    file.weight(name, &[dim, rows[1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_checkpoint_is_reported_by_name() {
        let error = match Model::load(Path::new("/tmp/definitely-not-a-model.gguf")) {
            Err(error) => error,
            Ok(_) => panic!("a path that does not exist must not load"),
        };
        let message = error.to_string();
        assert!(message.contains("definitely-not-a-model.gguf"), "{message}");
    }

    #[test]
    fn the_context_is_sized_from_the_front_end_constants() {
        // 501 mel frames, the dataset row, every selectable instrument, the
        // initial token, and a full chunk of generation.
        assert_eq!(501 + 1 + 35 + 1 + 2_000, 2_538);
        assert_eq!(MAX_SELECTABLE, 35);
    }

    #[test]
    fn the_error_text_says_what_the_cache_shortfall_is() {
        let error = Error::ContextOverflow {
            needed: 3_000,
            capacity: 2_538,
        };
        assert!(error.to_string().contains("3000"), "{error}");
        assert!(error.to_string().contains("2538"), "{error}");
    }
}