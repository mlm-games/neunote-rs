#![forbid(unsafe_code)]

//! The MuScriptor engine behind [`crate::Engine`].
//!
//! Everything around the model -- chunking, prelude forcing, note assembly,
//! MIDI export -- is engine-independent and already tested. This is the one
//! implementation of the trait that runs real inference.

use std::path::Path;

use neunote_engine::{Chunk as EngineChunk, Model, Stop as EngineStop};
use neunote_types::{MAX_TOKENS_PER_CHUNK, SEGMENT_SAMPLES};

use crate::{Chunk, ChunkRequest, Engine};

/// One model instance serves one job at a time.
///
/// Not `Sync`, and never shared: the KV cache is the whole state of a chunk's
/// decode, so a concurrent call would interleave two sequences through one
/// cache.
pub struct Muscriptor {
    model: Model,
}

impl Muscriptor {
    pub fn load(path: &Path) -> Result<Self, String> {
        Model::load(path)
            .map(|model| Self { model })
            .map_err(|error| format!("cannot load {}: {error}", path.display()))
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Wrap a model a host loaded itself -- from bytes it fetched or picked
    /// rather than from a path.
    pub fn from_model(model: Model) -> Self {
        Self { model }
    }
}

impl Engine for Muscriptor {
    fn segment_samples(&self) -> usize {
        SEGMENT_SAMPLES
    }

    fn generate(&mut self, request: ChunkRequest<'_>) -> Result<Chunk, String> {
        self.model.set_instrument_rows(request.instrument_rows);
        self.model.set_forbidden_tokens(request.forbidden);

        let conditioning = self
            .model
            .encode_audio(request.samples)
            .map_err(|error| error.to_string())?;

        let chunk: EngineChunk = self
            .model
            .generate(&conditioning, request.prompt, MAX_TOKENS_PER_CHUNK, eos_id())
            .map_err(|error| error.to_string())?;

        Ok(Chunk {
            tokens: chunk.tokens,
            stop: match chunk.stop {
                EngineStop::Eos => crate::Stop::Eos,
                EngineStop::Budget => crate::Stop::Budget,
            },
        })
    }
}

fn eos_id() -> i32 {
    neunote_tokenizer::EOS_ID
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_segment_is_five_seconds() {
        let path = std::env::var_os("NEUNOTE_WEIGHTS_DIR").map_or_else(
            || {
                std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                    .join(".local/share/neunote/models/muscriptor-small-f16.gguf")
            },
            |dir| std::path::PathBuf::from(dir).join("muscriptor-small-f16.gguf"),
        );
        if !path.exists() {
            return;
        }

        let engine = Muscriptor::load(&path).expect("loading the small checkpoint");
        assert_eq!(Engine::segment_samples(&engine), SEGMENT_SAMPLES);
        assert_eq!(SEGMENT_SAMPLES, 80_000);
    }
}
