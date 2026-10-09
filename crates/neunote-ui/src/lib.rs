#![forbid(unsafe_code)]

//! The neunote UI: one view tree over Repose, shared by the desktop shell and
//! the web shell.
//!
//! Everything platform-specific arrives through [`Shell`] -- opening a file,
//! writing one. What lives here is the state a transcription needs (source
//! audio, weights, options), the job that runs it, and the views that show it:
//! a piano roll, a track list, a progress line.

mod edit;
mod instruments;
mod job;
mod piano_roll;
mod quantize;
mod roll;
mod tracks;
mod waveform;
mod view;

pub use view::{
    LoadedAudio, LoadedWeights, Mix, Mode, Shell, TrackMix, Transport, root,
};
