# AGENTS.md

## What this repo is becoming

`neunote-rs` is being rewritten from a Basic Pitch port to a pure-Rust
[MuScriptor](https://github.com/muscriptor/muscriptor) host, matching
[NeuralNote](https://github.com/DamRsn/NeuralNote) v2's behaviour.

No C, no C++, no FFI. The inference engine is a pure-Rust reimplementation of
the model; every other layer was already Rust.

## The reference

[muscriptor.cpp](https://github.com/DamRsn/muscriptor.cpp) (MIT) is the source
of truth for everything numerical. It is not vendored and not linked. Its
`docs/MODEL.md`, `docs/TOKENIZER.md` and `cpp/src/*.cpp` are what a port reads,
and its `testdata/vectors/` is what a port tests against.

Everything here is transcribed from that reference or its docs, never inferred
from prose. Where a constant is quoted, it was read off the reference source.

## Crates

| Crate | Contents | Needs weights |
|---|---|---|
| `neunote-types` | Notes, instrument groups, constants | no |
| `neunote-tokenizer` | MT3 vocabulary, decode state machine, note assembly | no |
| `neunote-audio` | Decode to mono f32, resample to 16 kHz, chunking, peaks | no |
| `neunote-midi` | One track per instrument, drums on channel 10 | no |
| `neunote-models` | Manifest pins, resumable download, digest verification | no |
| `neunote-cli` | The pipeline around the engine, and the `neunote` binary | no |
| `neunote-engine` | GGUF reader, mel front-end, transformer, greedy decode | yes |

`neunote-cli` owns the seam the engine plugs into -- `pipeline::Engine` -- and
drives everything around it; `muscriptor::Muscriptor` is the one implementation
of that trait, and the only place the two crates meet. `neunote transcribe`
decodes, resamples, runs the model, and writes MIDI.

## Hard constraints

These are transcribed, not chosen. Do not "clean them up".

- **16 kHz mono f32 input**, in whole chunks of 80,000 samples (5.0 s).
- **`shift` is a step count, not milliseconds.** Value `v` means
  `start_tick + v`, at 100 steps per second. `shift 0` is a no-op, not a rewind.
  Reading it as a delta produces plausible, progressively wrong timing.
- **Prefix order** is `[mel, dataset_name, instrument_group…, tokens]`, the
  reverse of `ConditioningProvider`'s iteration order.
- **Class embeddings:** null class is row 1, group `g` is row `g + 2`.
- **Sinusoidal positions:** cosine half first, exponent denominator `half_dim - 1`.
- **GELU is the exact erf form**, never the tanh approximation.
- **Causal mask is bottom-right aligned**, so it covers a prefilled prompt.
- **The last mel frame is always masked.** The mask length comes from the
  waveform (80000/160 = 500) while the centred STFT produces 501.
- **Mel filterbank and STFT window come from the checkpoint.** Do not
  recompute the filterbank: it differs from `melscale_fbanks` by ~2e-4 relative,
  which is enough to flip greedy tokens.
- **Reflect padding omits the edge sample at both ends:** left pad is
  `x[1024] … x[1]`.
- **The STFT yields magnitudes** (`power = 1.0`), not powers.
- **`logits[1393:]` is always −inf.** `medium` and `large` have a card of 1395
  but ids 1393 and 1394 can never be produced.
- **The open-note map keeps insertion order.** `finish` emits ends in that
  order; a hash map reorders the output events.
- **Open notes carry across chunks; the registers do not.**
- **A `pitch` token past `next_seek_time` is dropped**, or chunk boundaries
  duplicate notes.
- **`is_drum` is a real field on `NoteEvent`, not `program == DRUM_PROGRAM`.**
  Programs 128 and 129 belong to no instrument group, so a *melodic* note can
  carry program 128, and conflating the two puts it in the wrong trimming
  channel.
- **Velocity is 100** everywhere. The model predicts no dynamics.
- **Never run inference on the UI or audio thread.** One model instance serves
  one job at a time.

These three are layout invariants rather than model facts, and each of them was
wrong at least once:

- **Activations are `[dim, tokens]`, the transpose of ggml's.** ggml keeps
  `ne[0]` contiguous, so a GGUF's weights and this port's activations disagree
  about which axis is contiguous. Every transfer between them -- the mel
  spectrum, the conditioning, the positions -- is a strided copy. A flat
  `copy_from_slice` between two such layouts reads the wrong cells and still
  produces numbers of the right size.
- **LayerNorm reduces over `dim`,** giving one mean and variance per token, with
  the affine pair indexed by `dim`. Reducing over the tokens instead also
  produces plausible numbers.
- **A slice along `ne[1]` is strided, not packed.** Slicing `n` columns out of
  a `[rows, cols]` buffer gives `n` runs of `rows`, so it is stored `[n, rows]`.
  Packing it as `[rows, n]` keeps the first `rows` elements of each column and
  records a shape that looks right.

## Model weights

CC BY-NC 4.0, non-commercial only. Downloaded at runtime, never committed.

| Size | File | Bytes | SHA-256 |
|---|---|---|---|
| small | `muscriptor-small-f16.gguf` | 209,425,152 | `925f55af65a20ebc4f8b45ceaf095a12b72493d436cb112623cd0041a1af23d4` |
| medium | `muscriptor-medium-f16.gguf` | 618,442,496 | `3850cc9e5b436b17a09bd25b8f2615cb3366ab96a71e7b50f73a793a917fdf03` |
| large | `muscriptor-large-f16.gguf` | 2,739,142,176 | `35a750fb1ab1e77195cdc2c0b9b4aeea2f4d59f11f729f02af9920c4854ef72e` |

Repo `DamRsn/muscriptor-gguf`, revision
`d7045f94e8b19427f4ff9542975035e66596e51c`, directory `v1`.

The container *is* a standard GGUF v3 file; what is custom is the
architecture, `muscriptor`, and the `muscriptor.*` metadata keys. `small` has 122
tensors and 47 metadata keys. They are a conversion for `muscriptor.cpp` and are
explicitly not loadable by llama.cpp or whisper.cpp, which expect a vocabulary
and an architecture they recognise -- a general GGUF *reader* opens them fine,
which is why `neunote-engine` has its own rather than depending on one. Verify
against the compiled-in digest, never a checksum fetched from the repo.

## Testing

```bash
cargo test --release --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Use `--release`. The engine tests run real inference, and a debug build of them
takes minutes per step rather than seconds.

Every crate has tests. The legacy Basic Pitch pipeline that used to live here
was deleted rather than kept as a fallback, so there is no path that produces
notes without MuScriptor.

`crates/neunote-tokenizer/tests/reference_vectors.rs` replays all 17 vectors in
`testdata/vectors/note_vectors.json`, vendored from the reference. Passing means
the decode state machine and note assembly agree with `OpenNoteTracker` and
`NoteAssembler` exactly. That needs no weights and no ML runtime, so it is the
cheapest possible check that a change is safe.

`testdata/vectors/tables.json` is the generated instrument-group table the
reference asserts its own `instrument_groups.inc` against.

`testdata/audio/fixture_3chunks_16k.wav` is the reference's own 15 s fixture.
`neunote-audio` decodes it to check the sample rate, length and that each chunk
carries signal; `neunote-cli` runs the binary against it for the end-to-end
tests.

### The ladder

`crates/neunote-engine/tests/ladder.rs` runs steps 1 to 3;
`crates/neunote-cli/tests/reference_notes.rs` runs step 4. Both compare against
`testdata/refs/small.bin`, which is dumped from `muscriptor.cpp` itself rather
than from anything here -- `tools/dump_refs.cpp` builds the reference, drives
its public API, and writes the tensors. `tools/README.md` has the commands. That
file is the only reason a claim about parity means anything: it is the
reference's own arithmetic, and it was produced before this port existed.

The engine tests need the `small` checkpoint at
`~/.local/share/neunote/models/muscriptor-small-f16.gguf`, or
`NEUNOTE_WEIGHTS_DIR` pointing at it. They are slow -- a minute for the ladder
and two or three for the notes, which transcribes the whole fixture twice over
-- so run them in release.

1. **The STFT magnitudes**, then every stage of the conditioning front-end in
   turn: the filterbank, the log, the projection, the masked embedding. Each is
   checked on its own so a failure names the operation.
2. **The prefill logits**, and layer 0's intermediates between them, so a
   divergence is located rather than merely detected. Cosine similarity and the
   top-1 id are both asserted, with a conditioning row selected as well as not.
3. **Greedy tokens for one chunk: an exact match.** All 457 of them. The stream
   is the whole forward pass turned into a decision, which is why steps 1 and 2
   exist.
4. **A whole file's notes.** 451 notes over three chunks, with two carried
   across a boundary and a tail that `finish()` closes. Onset and offset within
   10 ms, pitch, program and the drum flag exact.

All four pass. Tolerances are where the reference's own f32 accumulation order
puts the numbers, not where this port's happen to land: the divergence grows
from 3e-6 at the layer input to 1.8e-3 at its output, which is fourteen layers
of summing in a different order.

Step 5, CPU f32 against GPU f16, does not apply: this is a CPU-only engine.

### The engine contract

`pipeline::Engine::generate` takes a `ChunkRequest` -- one zero-padded chunk, the
forced tie prologue, the conditioning rows, the forbidden ids -- and returns a
`Chunk` whose `Stop` says whether EOS ended the stream or the budget did. The
pipeline replays the returned token stream through the tracker exactly once,
breaking at EOS. The forced prompt is *inside* that stream, because the
reference's own `generate` puts it there: feeding it separately as well would
double-feed every chunk boundary.

A run that ends without EOS is a real outcome, not an error. The reference treats
running out of budget as a warning, so `Stop::Budget` carries it and nothing
appends an EOS the model did not emit.

## Licence

Code GPL-3.0. Weights CC BY-NC 4.0. Soundfonts carry their own licences. Keep
the three separate in `NOTICE`; the code licence does not extend to the weights.