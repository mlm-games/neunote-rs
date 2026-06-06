use crate::midi::events::NoteEvent;
use crate::ml::cnn::BasicPitchCNN;
use crate::ml::constants::*;
use crate::ml::notes::{self, ConvertParams};

/// Full Basic Pitch transcription pipeline
pub struct BasicPitch {
    cnn: BasicPitchCNN,
    params: ConvertParams,
    contours_pg: Vec<f32>,
    notes_pg: Vec<f32>,
    onsets_pg: Vec<f32>,
    note_events: Vec<NoteEvent>,
    num_frames: usize,
}

impl BasicPitch {
    /// Create a new pipeline with loaded CNN weights.
    /// Each weight slice is flat in [t][kf][fi][fo] order (RTNeural JSON format).
    pub fn new(
        contour_w1: &[f32], contour_b1: &[f32],
        contour_w2: &[f32], contour_b2: &[f32],
        note_w1: &[f32], note_b1: &[f32],
        note_w2: &[f32], note_b2: &[f32],
        onset1_w: &[f32], onset1_b: &[f32],
        onset2_w: &[f32], onset2_b: &[f32],
    ) -> Self {
        let cnn = BasicPitchCNN::new(
            contour_w1, contour_b1,
            contour_w2, contour_b2,
            note_w1, note_b1,
            note_w2, note_b2,
            onset1_w, onset1_b,
            onset2_w, onset2_b,
        );
        Self {
            cnn,
            params: ConvertParams::default(),
            contours_pg: vec![],
            notes_pg: vec![],
            onsets_pg: vec![],
            note_events: vec![],
            num_frames: 0,
        }
    }

    /// Set transcription parameters
    pub fn set_parameters(
        &mut self,
        note_sensitivity: f32,
        split_sensitivity: f32,
        min_note_duration_ms: f32,
    ) {
        self.params.frame_threshold = 1.0 - note_sensitivity;
        self.params.onset_threshold = 1.0 - split_sensitivity;
        let hop_sec = FFT_HOP as f64 / BASIC_PITCH_SAMPLE_RATE;
        self.params.min_note_length =
            (min_note_duration_ms as f64 / 1000.0 / hop_sec).round() as usize;
        self.params.pitch_bend = notes::PitchBendMode::Multi;
        self.params.melodia_trick = true;
        self.params.infer_onsets = true;
    }

    /// Transcribe audio to MIDI notes.
    ///
    /// `audio_22050`: audio samples at 22050 Hz sample rate
    pub fn transcribe(&mut self, audio_22050: &[f32]) {
        // Run features (CQT + Harmonic stacking) — in practice this would use
        // the ONNX model. For now, we need to compute features.
        // TODO: Implement CQT + harmonic stacking
            
        // For a working version, we still need the ONNX CQT model.
        // This will be implemented when the ONNX runtime is integrated.
        
        // Placeholder: compute features using a stub
        let num_samples = audio_22050.len();
        let max_frames = num_samples / FFT_HOP;
        // Each frame has NUM_HARMONICS * NUM_FREQ_IN features
        let features_size = max_frames * NUM_HARMONICS * NUM_FREQ_IN;
        let features = vec![0.0f32; features_size];

        // Run CNN
        self.num_frames = max_frames;
        self.contours_pg = vec![0.0; self.num_frames * NUM_FREQ_IN];
        self.notes_pg = vec![0.0; self.num_frames * NUM_FREQ_OUT];
        self.onsets_pg = vec![0.0; self.num_frames * NUM_FREQ_OUT];

        // Check if zero frames
        if self.num_frames == 0 {
            return;
        }

        // Run CNN inference
        self.cnn.process_all_frames(
            &features,
            self.num_frames,
            &mut self.contours_pg,
            &mut self.notes_pg,
            &mut self.onsets_pg,
        );

        // Convert posteriorgrams to note events
        self.note_events = notes::posteriorgrams_to_notes(
            &self.notes_pg,
            &self.onsets_pg,
            &self.contours_pg,
            self.num_frames,
            &self.params,
        );
    }

    /// Re-run note extraction with updated parameters (no need to re-run CNN)
    pub fn update_midi(&mut self) {
        if self.num_frames == 0 {
            return;
        }
        self.note_events = notes::posteriorgrams_to_notes(
            &self.notes_pg,
            &self.onsets_pg,
            &self.contours_pg,
            self.num_frames,
            &self.params,
        );
    }

    pub fn note_events(&self) -> &[NoteEvent] {
        &self.note_events
    }

    pub fn num_frames(&self) -> usize {
        self.num_frames
    }
}
