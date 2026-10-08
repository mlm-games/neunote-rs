# Reference vectors

`note_vectors.json` and `tables.json` are vendored verbatim from
[muscriptor.cpp](https://github.com/DamRsn/muscriptor.cpp) (MIT),
`testdata/vectors/`, at revision `d7045f94e8b19427f4ff9542975035e66596e51c`.

They are the reference's own hand-written decode cases, so `neunote-tokenizer`
can be tested for exact parity with `OpenNoteTracker` and `NoteAssembler`
without any model weights.
