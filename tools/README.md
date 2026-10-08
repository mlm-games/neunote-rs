# tools

`dump_refs.cpp` regenerates `testdata/refs/*.bin`, the golden tensors the test
ladder in `crates/neunote-engine/tests/ladder.rs` and
`crates/neunote-cli/tests/reference_notes.rs` compare against.

It is the only C++ in this repository and it is **not part of the build**. It
drives `muscriptor.cpp`'s public API from outside, so the numbers it produces are
the reference's own rather than this port's. Nothing in `cargo build` or
`cargo test` compiles it, and the workspace has no C++ toolchain dependency.

## Regenerating a dump

```sh
git clone https://github.com/DamRsn/muscriptor.cpp /tmp/mcpp
cd /tmp/mcpp
cmake -S cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

# built against the reference's static library and ggml's
g++ -std=c++23 -O2 -o build/dump_refs "$OLDPWD/tools/dump_refs.cpp" \
    -Icpp/include -Icpp/src -Ibuild/_deps/ggml-src/include -Icpp/third_party/pffft \
    build/libmuscriptor_ggml.a build/libpffft.a \
    $(find build/_deps/ggml-build -name '*.a') -lpthread -lm -ldl

./build/dump_refs \
    ~/.local/share/neunote/models/muscriptor-small-f16.gguf \
    "$OLDPWD/testdata/audio/fixture_3chunks_16k.wav" \
    /tmp/refs-small.bin

cp /tmp/refs-small.bin "$OLDPWD/testdata/refs/small.bin"
```

Each tensor is cached under `build/refcache`, so re-running to change one entry
does not pay for the whole fixture again. The transcription alone is several
minutes of CPU.

## Why the file is 7 MB

The front-end tensors are kept whole -- the spectrum, all four conditioning
stages, the logits -- because each isolates one operation and a bisection needs
to know which one moved. The transformer's per-layer intermediates are kept as a
few columns each: they are megabytes apiece, and a layout mistake shows up in
column zero.

## Slices are gathered, never packed

ggml keeps `ne[0]` contiguous, so a slice along `ne[1]` has its kept elements
`ne[0]` floats apart in the source. The dump stores such a slice as
`[kept, ne[0]]` and says so in the recorded shape. A slice packed as
`[ne[0], kept]` keeps only the first `ne[0]` elements of each column, which
records a plausible shape and completely wrong contents -- and reads as a large
numeric disagreement rather than as a broken tool.