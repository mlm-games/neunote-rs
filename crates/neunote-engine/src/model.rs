#![forbid(unsafe_code)]

//! The transformer: prefix assembly, KV cache, greedy decoding.
//!
//! Stateful like the reference. `reset` clears the cache and rewinds to
//! position 0, `prefill` consumes a chunk's conditioning prefix plus its prompt,
//! and each `decode` advances one position. Activations are `[reduction, tokens]`
//! throughout, which is ggml's layout, so a GGUF's weights are used exactly as
//! they lie in the file.

use rayon::prelude::*;

use crate::Error;
use crate::gguf::Weight;
use crate::ops;
use crate::stft;

/// Architecture and front-end constants, all read from the checkpoint's
/// `muscriptor.*` metadata rather than written down here.
#[derive(Debug, Clone, Copy)]
pub struct Hparams {
    pub dim: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    /// `card`: the number of real logits.
    pub vocab_size: usize,
    pub initial_token_id: i32,
    /// `_compute_logits` forces `logits[logit_mask_start..]` to negative
    /// infinity whatever the card is, so `medium` and `large` carry ids that
    /// their larger embedding table can otherwise produce but never do.
    pub logit_mask_start: usize,
    pub layer_norm_eps: f32,
    pub max_period: f32,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub frame_rate: i32,
    pub n_mels: usize,
    pub log_eps: f32,
}

impl Hparams {
    pub fn n_freq(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Frames a full segment produces, including the one past the audio.
    pub fn mel_frames(&self, segment_samples: usize) -> usize {
        stft::frame_count(segment_samples, self.hop_length)
    }
}

pub struct LayerWeights {
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,
    pub attn_qkv: Weight,
    pub attn_out: Weight,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn_norm_bias: Vec<f32>,
    pub ffn_up: Weight,
    pub ffn_down: Weight,
}

pub struct Weights {
    pub token_embedding: Weight,
    pub output: Weight,
    pub output_norm_weight: Vec<f32>,
    pub output_norm_bias: Vec<f32>,
    pub mel_filterbank: Weight,
    pub stft_window: Vec<f32>,
    pub cond_proj_weight: Weight,
    pub cond_proj_bias: Vec<f32>,
    pub instrument_group: Weight,
    pub dataset_name: Weight,
    pub layers: Vec<LayerWeights>,
}

/// Named intermediates from a forward pass.
///
/// The reference exposes the same thing, and it is what makes a mismatch
/// locatable: without it a wrong logit says only that the transformer is wrong,
/// and with it the first stage that diverges names itself. Capturing copies each
/// named stage, so a production run passes no trace at all.
///
/// Every tensor is `[reduction, columns]` -- the port's own activation order.
/// The reference's are the transpose, since ggml keeps `ne[0]` contiguous, so a
/// comparison transposes.
#[derive(Default)]
pub struct Trace {
    stages: Vec<(String, Vec<f32>)>,
}

impl Trace {
    pub fn read(&self, name: &str) -> Option<&[f32]> {
        self.stages
            .iter()
            .find(|(stage, _)| stage == name)
            .map(|(_, values)| values.as_slice())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.read(name).is_some()
    }
}

fn capture(trace: &mut Option<&mut Trace>, name: &str, tensor: &[f32]) {
    if let Some(trace) = trace {
        trace.stages.push((name.to_owned(), tensor.to_vec()));
    }
}

/// What a chunk's generation stopped on.
///
/// The reference treats running out of budget as a warning rather than a
/// failure, so a stream that ends without EOS is a real outcome and must not be
/// reported as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    Eos,
    Budget,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub tokens: Vec<i32>,
    pub stop: Stop,
}

/// The conditioning front-end's stages, in the order it computes them.
///
/// Every tensor is `[rows, frames]`: the reduction axis is the outer one, which
/// is what every matmul here wants. The reference's own buffers are the
/// transpose of this, since ggml keeps `ne[0]` contiguous.
#[derive(Debug, Clone)]
pub struct Conditioning {
    pub mel: Vec<f32>,
    pub logmel: Vec<f32>,
    /// The projection before the frame mask, so a caller can see the projection
    /// and the mask separately.
    pub projection: Vec<f32>,
    pub embedding: Vec<f32>,
}

pub struct Model {
    hp: Hparams,
    weights: Weights,
    /// Per layer, `[n_head][n_ctx][head_dim]`: one head's keys or values for a
    /// whole position are contiguous, which is what the attention loop reads.
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    n_ctx: usize,
    positions: Vec<f32>,
    n_past: usize,
    instrument_rows: Vec<i32>,
    forbidden: Vec<bool>,
}

impl Model {
    pub(crate) fn new(hp: Hparams, weights: Weights, n_ctx: usize) -> Result<Self, Error> {
        if !hp.dim.is_multiple_of(2) {
            return Err(Error::Checkpoint(
                "the model dimension must be even for sinusoidal positions".into(),
            ));
        }
        if hp.head_dim * hp.n_head != hp.dim {
            return Err(Error::Checkpoint(format!(
                "inconsistent head geometry: {} heads x {} != {}",
                hp.n_head, hp.head_dim, hp.dim
            )));
        }

        let cache_len = hp.n_head * n_ctx * hp.head_dim;
        Ok(Self {
            positions: position_table(n_ctx, hp.dim, hp.max_period),
            k_cache: (0..hp.n_layer).map(|_| vec![0.0; cache_len]).collect(),
            v_cache: (0..hp.n_layer).map(|_| vec![0.0; cache_len]).collect(),
            hp,
            weights,
            n_ctx,
            n_past: 0,
            instrument_rows: vec![NULL_CONDITIONING_ROW],
            forbidden: Vec::new(),
        })
    }

    pub fn hparams(&self) -> &Hparams {
        &self.hp
    }

    pub fn n_past(&self) -> usize {
        self.n_past
    }

    pub fn context_size(&self) -> usize {
        self.n_ctx
    }

    /// One prefix position per selected instrument. Empty restores the
    /// unconditional path: a single null-class row.
    ///
    /// This changes the prefix length rather than just its contents, which is
    /// why a selection is not a filter on a fixed prompt.
    pub fn set_instrument_rows(&mut self, rows: &[i32]) {
        self.instrument_rows = if rows.is_empty() {
            vec![NULL_CONDITIONING_ROW]
        } else {
            rows.to_vec()
        };
    }

    pub fn instrument_rows(&self) -> &[i32] {
        &self.instrument_rows
    }

    /// Token ids forced to negative infinity on top of the reserved-id mask,
    /// on every forward pass, as `_compute_logits` does.
    pub fn set_forbidden_tokens(&mut self, ids: &[i32]) {
        self.forbidden.clear();

        if ids.is_empty() {
            return;
        }

        self.forbidden = vec![false; self.hp.vocab_size];
        for id in ids {
            if let Ok(index) = usize::try_from(*id)
                && index < self.hp.vocab_size
            {
                self.forbidden[index] = true;
            }
        }
    }

    /// Clear the KV cache and rewind to position 0. Call between chunks.
    pub fn reset(&mut self) {
        self.n_past = 0;
        for layer in 0..self.hp.n_layer {
            self.k_cache[layer].fill(0.0);
            self.v_cache[layer].fill(0.0);
        }
    }

    /// The checkpoint's STFT window, rather than a regenerated Hann window. The
    /// reference's is periodic -- it divides by `n_fft`, not `n_fft - 1` -- and
    /// the difference is enough to move greedy tokens.
    pub fn stft_window(&self) -> &[f32] {
        &self.weights.stft_window
    }

    /// STFT magnitudes for one chunk, then the front-end over them, as
    /// `encodeConditioning(stft().magnitudes(samples), stft().nFrames(len), len)`.
    pub fn encode_audio(&self, samples: &[f32]) -> Result<Vec<f32>, Error> {
        let spectrum = stft::magnitudes(
            samples,
            self.hp.n_fft,
            self.hp.hop_length,
            &self.weights.stft_window,
        )?;
        self.encode_conditioning(&spectrum, samples.len())
    }

    /// The conditioning front-end, over STFT magnitudes laid out
    /// `[n_frames][n_freq]` as `magnitudes` returns them.
    ///
    /// `log(filterbank . spectrum + eps)`, projected, then masked. The mask is
    /// derived from the waveform length and not from the frame count: a centred
    /// STFT always emits one frame past `length / hop`, and that frame has to
    /// come out exactly zero rather than merely small.
    pub fn encode_conditioning(
        &self,
        spectrum: &[f32],
        n_samples: usize,
    ) -> Result<Vec<f32>, Error> {
        Ok(self
            .encode_conditioning_stages(spectrum, n_samples)?
            .embedding)
    }

    /// The same front-end, keeping every stage.
    ///
    /// `encode_conditioning` returns only the embedding, which is all a caller
    /// wants. This is what the reference comparison uses: a divergence then
    /// points at one operation instead of at the front-end as a whole.
    pub fn encode_conditioning_stages(
        &self,
        spectrum: &[f32],
        n_samples: usize,
    ) -> Result<Conditioning, Error> {
        let hp = &self.hp;
        let frames = stft::frame_count(n_samples, hp.hop_length);
        let expected = frames * hp.n_freq();

        if spectrum.len() != expected {
            return Err(Error::Checkpoint(format!(
                "spectrum has {} values, expected {expected} ({frames} frames x {})",
                spectrum.len(),
                hp.n_freq()
            )));
        }

        let valid = n_samples / hp.hop_length;

        // `magnitudes` hands back [n_frames][n_freq], the layout the reference's
        // own vector uses. ggml declares that same buffer as ne = [n_freq,
        // n_frames] -- which is the identical memory layout, so element (k, t)
        // of the ggml tensor is frame t's bin k. A matmul here reads the
        // reduction axis contiguous, so the buffer is transposed rather than
        // reinterpreted: skipping this silently multiplies each frame's spectrum
        // by a different filterbank row.
        let mut by_bin = vec![0.0f32; expected];
        for frame in 0..frames {
            for bin in 0..hp.n_freq() {
                by_bin[bin * frames + frame] = spectrum[frame * hp.n_freq() + bin];
            }
        }

        let mut mel = vec![0.0f32; hp.n_mels * frames];
        ops::matmul(
            &mut mel,
            &self.weights.mel_filterbank,
            &by_bin,
            frames,
            None,
        );

        let mut logmel = mel.clone();
        for value in logmel.iter_mut() {
            *value = (*value + hp.log_eps).ln();
        }

        let mut projected = vec![0.0f32; hp.dim * frames];
        ops::matmul(
            &mut projected,
            &self.weights.cond_proj_weight,
            &logmel,
            frames,
            Some(&self.weights.cond_proj_bias),
        );

        // The mask length comes from the waveform, not from the frame count, so the
        // frame past the audio is masked away. A frame is a column of the
        // `[dim, frames]` layout, so it is strided rather than contiguous.
        let mut embedding = projected.clone();
        for row in 0..hp.dim {
            for frame in valid..frames {
                embedding[row * frames + frame] = 0.0;
            }
        }

        Ok(Conditioning {
            mel,
            logmel,
            projection: projected,
            embedding,
        })
    }

    /// The first forward pass of a chunk: the conditioning embedding, then the
    /// class embeddings, then the tokens, with the positions added once after
    /// the whole prefix is assembled.
    pub fn prefill(&mut self, conditioning: &[f32], tokens: &[i32]) -> Result<Vec<f32>, Error> {
        self.forward(Some(conditioning), tokens, None)
    }

    /// A prefill that also returns the stages named in `trace`.
    pub fn prefill_traced(
        &mut self,
        conditioning: &[f32],
        tokens: &[i32],
        trace: &mut Trace,
    ) -> Result<Vec<f32>, Error> {
        self.forward(Some(conditioning), tokens, Some(trace))
    }

    /// One autoregressive step.
    pub fn decode(&mut self, token: i32) -> Result<Vec<f32>, Error> {
        self.forward(None, &[token], None)
    }

    /// Greedy-decode a chunk, teacher-forcing `prompt`.
    ///
    /// `[initial_token, prompt...]` goes through one square-causal prefill,
    /// which is what the reference does: it writes the prompt into the
    /// generation sequence and starts decoding from its end rather than
    /// stepping through it.
    ///
    /// The returned stream is `prompt` followed by what was generated, and the
    /// decode state machine needs the prompt to see it -- so it is in the answer,
    /// not beside it.
    ///
    /// `max_tokens` bounds the prompt and the generated tokens together, as
    /// `max_gen_len` does upstream.
    pub fn generate(
        &mut self,
        conditioning: &[f32],
        prompt: &[i32],
        max_tokens: usize,
        eos_id: i32,
    ) -> Result<Chunk, Error> {
        self.reset();

        let mut prefill_tokens = Vec::with_capacity(prompt.len() + 1);
        prefill_tokens.push(self.hp.initial_token_id);
        prefill_tokens.extend_from_slice(prompt);

        let mut logits = self.prefill(conditioning, &prefill_tokens)?;

        let mut tokens = prompt.to_vec();
        let mut stop = Stop::Budget;

        for step in prompt.len()..max_tokens {
            let next = argmax(&logits);
            tokens.push(next);

            if next == eos_id {
                stop = Stop::Eos;
                break;
            }

            // Skip the last forward pass once the budget is spent: nothing reads
            // its logits and it would take one more KV position.
            if step + 1 < max_tokens {
                logits = self.decode(next)?;
            }
        }

        Ok(Chunk { tokens, stop })
    }

    /// One forward pass. `conditioning` is `Some` for a chunk's prefill, which
    /// is what prepends the prefix; `None` marks a decode step.
    fn forward(
        &mut self,
        conditioning: Option<&[f32]>,
        tokens: &[i32],
        mut trace: Option<&mut Trace>,
    ) -> Result<Vec<f32>, Error> {
        if tokens.is_empty() {
            return Err(Error::Invalid(
                "a forward pass needs at least one token".into(),
            ));
        }

        let Model {
            hp,
            weights,
            k_cache,
            v_cache,
            n_ctx,
            positions,
            n_past,
            instrument_rows,
            forbidden,
            ..
        } = self;

        let dim = hp.dim;

        // The class embeddings ride along with the mel frames on the first pass:
        // one dataset row, and one row per selected instrument.
        let prepended = conditioning.is_some();
        let n_frames = conditioning.map_or(0, |embedding| embedding.len() / dim);
        let n_instrument = if prepended { instrument_rows.len() } else { 0 };
        let n_new = if prepended {
            n_frames + 1 + n_instrument + tokens.len()
        } else {
            tokens.len()
        };
        let n_kv = *n_past + n_new;

        if n_kv > *n_ctx {
            return Err(Error::ContextOverflow {
                needed: n_kv,
                capacity: *n_ctx,
            });
        }

        if prepended && conditioning.map(<[f32]>::len) != Some(n_frames * dim) {
            return Err(Error::Invalid(format!(
                "conditioning has {} values, expected {}",
                conditioning.map_or(0, <[f32]>::len),
                n_frames * dim
            )));
        }

        let mut x = vec![0.0f32; dim * n_new];

        if prepended {
            // Strided, not a flat copy. `conditioning` is `[dim, frames]` and `x`
            // is `[dim, n_new]`, and the two have different row pitches, so a
            // flat copy lands every row at the wrong column -- off by
            // `n_new - frames` elements per row. It still looks like a mel
            // embedding.
            let embedding = conditioning.expect("checked above");
            for row in 0..dim {
                for column in 0..n_frames {
                    x[row * n_new + column] = embedding[row * n_frames + column];
                }
            }

            let mut column = n_frames;

            let dataset = weights
                .dataset_name
                .row_as_f32(NULL_CONDITIONING_ROW as usize, dim);
            for index in 0..dim {
                x[index * n_new + column] = dataset[index];
            }
            column += 1;

            for row in instrument_rows.iter() {
                let embedding = weights.instrument_group.row_as_f32(*row as usize, dim);
                for index in 0..dim {
                    x[index * n_new + column] = embedding[index];
                }
                column += 1;
            }
        }

        let first_token = if prepended {
            n_frames + 1 + n_instrument
        } else {
            0
        };
        for (offset, token) in tokens.iter().enumerate() {
            let embedding = weights.token_embedding.row_as_f32(*token as usize, dim);
            for index in 0..dim {
                x[index * n_new + first_token + offset] = embedding[index];
            }
        }

        // Positions are added once, after the prefix is assembled: a mel frame
        // and a token at the same offset share one position.
        //
        // The table is `[position][dim]` and `x` is `[dim][column]`, so this is
        // a strided read rather than a straight zip. Zipping them would add
        // position `c`'s value to activation row `c` instead of row 0, which
        // still produces plausible numbers downstream.
        for column in 0..n_new {
            let row = &positions[(*n_past + column) * dim..(*n_past + column + 1) * dim];
            for (index, position) in row.iter().enumerate() {
                x[index * n_new + column] += *position;
            }
        }
        capture(&mut trace, "pre.layer_in", &x);

        let mask = ops::bottom_right_causal_mask(*n_past, n_new, n_kv);

        for layer in 0..hp.n_layer {
            // Only layer 0 is traced, and asking per layer rather than branching
            // per stage keeps the trace out of the layer's own code.
            let layer_trace = if layer == 0 {
                trace.as_deref_mut()
            } else {
                None
            };
            block(
                &mut x,
                &weights.layers[layer],
                &mut k_cache[layer],
                &mut v_cache[layer],
                &mask,
                &Geometry {
                    dim,
                    head: hp.n_head,
                    head_dim: hp.head_dim,
                    ffn_dim: hp.ffn_dim,
                    norm_eps: hp.layer_norm_eps,
                    past: *n_past,
                    new: n_new,
                    kv: n_kv,
                    ctx: *n_ctx,
                },
                layer_trace,
            );
        }

        let normed = ops::layer_norm(
            &x,
            &weights.output_norm_weight,
            &weights.output_norm_bias,
            hp.layer_norm_eps,
            n_new,
        );

        // Only the last position is ever sampled, so the head runs on one row.
        let last = n_new - 1;
        let column: Vec<f32> = (0..dim).map(|index| normed[index * n_new + last]).collect();

        capture(&mut trace, "post.out_norm", &normed);

        let mut logits = vec![0.0f32; hp.vocab_size];
        ops::matmul(&mut logits, &weights.output, &column, 1, None);
        capture(&mut trace, "post.logits_raw", &logits);

        for slot in logits.iter_mut().skip(hp.logit_mask_start) {
            *slot = f32::NEG_INFINITY;
        }
        for (index, slot) in logits.iter_mut().enumerate() {
            if forbidden.get(index).copied().unwrap_or(false) {
                *slot = f32::NEG_INFINITY;
            }
        }

        *n_past = n_kv;
        Ok(logits)
    }
}

/// The row every class conditioner lands on when nothing is selected.
///
/// `tokenize(None)` gives `1 + (-1)`, and the conditioner embeds `inputs + 1`.
/// Row 0 is a real embedding the model never uses, and reading it instead still
/// produces plausible numbers.
pub const NULL_CONDITIONING_ROW: i32 = 1;

/// `create_sin_embedding`, evaluated on the host in fp32.
///
/// Two details here change every number downstream: the exponent denominator
/// is `half - 1` rather than `half`, and the halves are ordered cosine-then-sine
/// rather than the more common sine-then-cosine. The table is fp32 even when the
/// weights are not, because f16 cannot represent odd integers above 2048 and
/// adjacent positions would collapse onto the same embedding.
pub fn position_table(n_ctx: usize, dim: usize, max_period: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut table = vec![0.0f32; n_ctx * dim];

    for (position, row) in table.chunks_mut(dim).enumerate() {
        for index in 0..half {
            let exponent = index as f32 / (half - 1) as f32;
            let phase = position as f32 / max_period.powf(exponent);
            row[index] = phase.cos();
            row[half + index] = phase.sin();
        }
    }

    table
}

/// The shape every layer shares. A block's arguments are all either weights or
/// geometry, and eight scalars read as noise at the call site.
#[derive(Clone, Copy)]
struct Geometry {
    dim: usize,
    head: usize,
    head_dim: usize,
    ffn_dim: usize,
    /// From the checkpoint, never written down here.
    norm_eps: f32,
    past: usize,
    new: usize,
    kv: usize,
    ctx: usize,
}

/// One transformer layer, in place: reads `x`, writes `x + attention + feed-forward`.
///
/// `trace` is `Some` only for layer 0, and only for the ladder.
fn block(
    x: &mut [f32],
    weights: &LayerWeights,
    keys: &mut [f32],
    values: &mut [f32],
    mask: &[f32],
    shape: &Geometry,
    mut trace: Option<&mut Trace>,
) {
    let Geometry {
        dim,
        head,
        head_dim,
        ffn_dim,
        norm_eps,
        past,
        new,
        kv,
        ctx,
    } = *shape;

    let normed = ops::layer_norm(
        x,
        &weights.attn_norm_weight,
        &weights.attn_norm_bias,
        norm_eps,
        new,
    );
    capture(&mut trace, "blk.0.norm1", &normed);

    // in_proj packs (q, k, v) with q outermost and head_dim innermost, so a
    // head's vector is `head_dim` contiguous values inside its row.
    let mut qkv = vec![0.0f32; 3 * dim * new];
    ops::matmul(&mut qkv, &weights.attn_qkv, &normed, new, None);

    let mut queries = vec![0.0f32; dim * new];
    for head_index in 0..head {
        let base = head_index * head_dim;
        for token in 0..new {
            for offset in 0..head_dim {
                queries[(head_index * new + token) * head_dim + offset] =
                    qkv[(base + offset) * new + token];

                let slot = (head_index * ctx + past + token) * head_dim + offset;
                keys[slot] = qkv[(dim + base + offset) * new + token];
                values[slot] = qkv[(2 * dim + base + offset) * new + token];
            }
        }
    }

    let mut context = vec![0.0f32; dim * new];
    attend(
        &queries,
        keys,
        values,
        mask,
        1.0 / (head_dim as f32).sqrt(),
        new,
        kv,
        ctx,
        head_dim,
        &mut context,
    );
    capture(&mut trace, "blk.0.attn_ctx", &context);

    let mut projected = vec![0.0f32; dim * new];
    ops::matmul(&mut projected, &weights.attn_out, &context, new, None);
    capture(&mut trace, "blk.0.attn_out", &projected);
    for (slot, value) in x.iter_mut().zip(&projected) {
        *slot += *value;
    }
    capture(&mut trace, "blk.0.res1", x);

    let normed = ops::layer_norm(
        x,
        &weights.ffn_norm_weight,
        &weights.ffn_norm_bias,
        norm_eps,
        new,
    );
    capture(&mut trace, "blk.0.norm2", &normed);

    let mut hidden = vec![0.0f32; ffn_dim * new];
    ops::matmul(&mut hidden, &weights.ffn_up, &normed, new, None);
    capture(&mut trace, "blk.0.ffn_pre_gelu", &hidden);

    ops::gelu_erf_in_place(&mut hidden);
    capture(&mut trace, "blk.0.ffn_gelu", &hidden);

    ops::matmul(&mut projected, &weights.ffn_down, &hidden, new, None);
    capture(&mut trace, "blk.0.ffn_out", &projected);
    for (slot, value) in x.iter_mut().zip(&projected) {
        *slot += *value;
    }
    capture(&mut trace, "blk.0.out", x);
}

/// Scaled dot-product attention over the cache.
///
/// One head's keys for a whole position are contiguous, and the cache is laid
/// out `[n_head][n_ctx][head_dim]`, so a head's key matrix is exactly the
/// `[n_kv, head_dim]` this wants. `context` comes back in `[dim][n_new]`,
/// merging the heads side by side within a column.
#[allow(clippy::too_many_arguments)]
fn attend(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    mask: &[f32],
    scale: f32,
    n_new: usize,
    n_kv: usize,
    n_ctx: usize,
    head_dim: usize,
    context: &mut [f32],
) {
    let key_stride = n_ctx * head_dim;

    // One head per task. `par_chunks_mut` hands each closure its own head's
    // output columns, and `par_chunks` its own slice of the cache.
    context
        .par_chunks_mut(n_new * head_dim)
        .zip(keys.par_chunks(key_stride))
        .zip(values.par_chunks(key_stride))
        .enumerate()
        .for_each(|(head, ((out, keys), values))| {
            let query_base = head * n_new * head_dim;
            let mut scores = vec![0.0f32; n_kv];
            let mut merged = vec![0.0f32; head_dim];

            for token in 0..n_new {
                let query =
                    &queries[query_base + token * head_dim..query_base + (token + 1) * head_dim];

                for (key, slot) in scores.iter_mut().enumerate() {
                    let vector = &keys[key * head_dim..(key + 1) * head_dim];
                    *slot = crate::simd::dot_f32(query, vector) * scale;
                }

                let masked = &mask[token * n_kv..(token + 1) * n_kv];
                ops::masked_softmax(&mut scores, masked, n_kv);

                merged.fill(0.0);
                for (key, probability) in scores.iter().enumerate() {
                    if *probability == 0.0 {
                        continue;
                    }
                    let vector = &values[key * head_dim..(key + 1) * head_dim];
                    crate::simd::axpy_scale(&mut merged, *probability, vector);
                }

                for offset in 0..head_dim {
                    out[offset * n_new + token] = merged[offset];
                }
            }
        });
}

/// The first maximum, so ties resolve to the lowest id as the reference's
/// `max_element` does.
fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    best as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_position_table_puts_cosine_before_sine() {
        let table = position_table(4, 8, 10_000.0);
        let half = 4;

        // Row 0 is all ones and all zeros, whichever half each half occupies.
        assert_eq!(&table[0..half], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&table[half..8], &[0.0, 0.0, 0.0, 0.0]);

        // The exponent runs to 1 at the last index, so the highest frequency
        // there has period `max_period`.
        for position in 0..4 {
            let row = &table[position * 8..(position + 1) * 8];
            for index in 0..half {
                let exponent = index as f32 / (half - 1) as f32;
                let phase = position as f32 / 10_000.0f32.powf(exponent);
                assert!(
                    (row[index] - phase.cos()).abs() < 1e-5,
                    "cos at {position},{index}"
                );
                assert!(
                    (row[half + index] - phase.sin()).abs() < 1e-5,
                    "sin at {position},{index}"
                );
            }
        }
    }

    #[test]
    fn the_exponent_denominator_is_half_minus_one() {
        let table = position_table(2, 8, 10_000.0);
        let half = 4;

        // Index `half - 1` divides by `half - 1`, giving exponent 1 and period
        // exactly `max_period`. Using `half` instead would leave it at
        // (half - 1)/half.
        let last = table[8 + half - 1];
        let expected = (1.0f32 / 10_000.0).cos();
        assert!((last - expected).abs() < 1e-6, "{last} vs {expected}");
    }

    #[test]
    fn adjacent_positions_stay_distinct() {
        // The reason the table is fp32: in f16, position 2050 and 2051 collapse.
        let table = position_table(2_060, 8, 10_000.0);
        let (a, b) = (2_049 * 8, 2_050 * 8);
        assert!(table[a..a + 8] != table[b..b + 8]);
    }

    #[test]
    fn argmax_breaks_ties_towards_the_lowest_id() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
        // Masked-off logits must not win.
        assert_eq!(argmax(&[f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY]), 1);
    }
}
