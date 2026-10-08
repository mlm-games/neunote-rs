#![forbid(unsafe_code)]

//! The command line tool's library half.
//!
//! Everything except argument parsing lives here so it can be tested against
//! the reference's audio fixture without spawning a process.

pub mod pipeline;
