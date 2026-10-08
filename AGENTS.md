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
| `neunote-engine` | **not yet written.** Mel front-end, transformer, greedy decode. | yes |

Everything except `neunote-engine` is done. `neunote-cli` defines the seam the
engine plugs into -- `pipeline::Engine` -- and drives everything around it, so
adding the engine is an implementation of one trait rather than a rewrite.
`neunote transcribe` therefore reads and resamples the audio, then fails with
an explanation instead of writing an empty MIDI file.

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

## Model weights

CC BY-NC 4.0, non-commercial only. Downloaded at runtime, never committed.

| Size | File | Bytes | SHA-256 |
|---|---|---|---|
| small | `muscriptor-small-f16.gguf` | 209,425,152 | `925f55af65a20ebc4f8b45ceaf095a12b72493d436cb112623cd0041a1af23d4` |
| medium | `muscriptor-medium-f16.gguf` | 618,442,496 | `3850cc9e5b436b17a09bd25b8f2615cb3366ab96a71e7b50f73a793a917fdf03` |
| large | `muscriptor-large-f16.gguf` | 2,739,142,176 | `35a750fb1ab1e77195cdc2c0b9b4aeea2f4d59f11f729f02af9920c4854ef72e` |

Repo `DamRsn/muscriptor-gguf`, revision
`d7045f94e8b19427f4ff9542975035e66596e51c`, directory `v1`.

These are **not** standard GGUF. They are a conversion for `muscriptor.cpp` and
are explicitly not loadable by llama.cpp or whisper.cpp, so a generic GGUF
reader will not open them. Verify against the compiled-in digest, never a
checksum fetched from the repo.

## Testing

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

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

The engine has no tests yet, and cannot until there is something to compare it
to. The ladder, in order:

1. Mel of one chunk vs a dump from the reference. Bound the error.
2. Prefill logits vs a dump. Cosine similarity and top-1 token.
3. Greedy tokens for one chunk vs a dump. Exact match.
4. A whole file's notes vs a dump. Onset within 10 ms, pitch and program exact.
5. CPU f32 vs GPU f16: event-identical, or a documented tolerance.

Steps 1 and 2 are where the architecture details above get proven. Do not claim
parity before step 3 passes.

The engine implements `neunote_cli::pipeline::Engine`. Its `generate` receives
one zero-padded chunk, the forced tie prologue (empty on the first chunk, and
empty when forcing is off), and the forbidden-token ids. It must return the
generated tokens *including* EOS and including the forced prompt, because the
pipeline replays the prompt through the tracker separately.

## Licence

Code GPL-3.0. Weights CC BY-NC 4.0. Soundfonts carry their own licences. Keep
the three separate in `NOTICE`; the code licence does not extend to the weights.