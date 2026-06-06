// Basic Pitch model constants
pub const NUM_HARMONICS: usize = 8;
pub const NUM_FREQ_IN: usize = 264;
pub const NUM_FREQ_OUT: usize = 88;
pub const BASIC_PITCH_SAMPLE_RATE: f64 = 22050.0;
pub const MIDI_OFFSET: usize = 21;
pub const FFT_HOP: usize = 256;
pub const AUDIO_SAMPLE_RATE: u32 = 22050;
pub const MAX_NOTE_IDX: usize = 87;
pub const AUDIO_WINDOW_LENGTH: f64 = 2.0;
pub const ANNOTATIONS_BASE_FREQUENCY: f32 = 27.5;
pub const CONTOURS_BINS_PER_SEMITONE: usize = 3;
pub const MIN_MIDI_NOTE: u8 = 21;
pub const MAX_MIDI_NOTE: u8 = 108;

// CNN architecture constants
pub const N_FREQ_BINS_CONTOURS: usize = NUM_FREQ_OUT * CONTOURS_BINS_PER_SEMITONE; // 264

/// Derived model frame rate
pub const ANNOTATIONS_FPS: usize = AUDIO_SAMPLE_RATE as usize / FFT_HOP;
/// Number of frames in the time-frequency representations
pub const ANNOT_N_FRAMES: usize = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH as usize;
/// Number of samples in the (clipped) audio input
pub const AUDIO_N_SAMPLES: usize = AUDIO_SAMPLE_RATE as usize * AUDIO_WINDOW_LENGTH as usize - FFT_HOP;
