use std::path::Path;

use crate::midi::events::NoteEvent;
use crate::ml::cnn::BasicPitchCNN;
use crate::ml::constants::*;
use crate::ml::features::FeatureExtractor;
use crate::ml::notes::{self, ConvertParams};
use crate::ml::weights::CnnWeights;

/// Full Basic Pitch transcription pipeline
pub struct BasicPitch {
    cnn: BasicPitchCNN,
    features: FeatureExtractor,
    params: ConvertParams,
    contours_pg: Vec<f32>,
    notes_pg: Vec<f32>,
    onsets_pg: Vec<f32>,
    note_events: Vec<NoteEvent>,
    num_frames: usize,
}

impl BasicPitch {
    pub fn new(w: &CnnWeights, model_dir: &Path) -> Self {
        let cnn = BasicPitchCNN::new(w);
        Self {
            cnn,
            features: FeatureExtractor::new(model_dir),
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

    /// Transcribe audio at 22050 Hz to MIDI notes
    pub fn transcribe(&mut self, audio_22050: &[f32]) {
        let (features, model_frames) = self.features.compute_features(audio_22050);
        self.num_frames = model_frames;
        if self.num_frames == 0 {
            self.note_events = vec![];
            return;
        }

        self.contours_pg = vec![0.0; self.num_frames * NUM_FREQ_IN];
        self.notes_pg = vec![0.0; self.num_frames * NUM_FREQ_OUT];
        self.onsets_pg = vec![0.0; self.num_frames * NUM_FREQ_OUT];

        self.cnn.process_all_frames(
            &features,
            self.num_frames,
            &mut self.contours_pg,
            &mut self.notes_pg,
            &mut self.onsets_pg,
        );

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
