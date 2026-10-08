use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::InferenceFact;

/// Compute CQT + Harmonic Stacking features using the pre-trained ONNX model.
/// Output per frame: [channel][feature] = [harmonic * NUM_FREQ + freq_bin] layout
pub struct FeatureExtractor {
    model_bytes: Vec<u8>,
    plans: HashMap<usize, Arc<InferenceSimplePlan>>,
}

impl FeatureExtractor {
    pub fn new(model_dir: &Path) -> Self {
        let model_path = model_dir.join("features_model.onnx");
        let model_bytes = std::fs::read(&model_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", model_path.display(), e));
        Self {
            model_bytes,
            plans: HashMap::new(),
        }
    }

    /// Returns (features, num_frames) where features is a flat Vec<f32> in
    /// [frame][channel][feature] layout.
    pub fn compute_features(&mut self, audio_22050: &[f32]) -> (Vec<f32>, usize) {
        let num_samples = audio_22050.len();
        if num_samples < 2048 {
            return (vec![], 0);
        }

        let model_bytes = &self.model_bytes;
        let runnable = self.plans.entry(num_samples).or_insert_with(|| {
            let infered = onnx()
                .model_for_read(&mut &model_bytes[..])
                .unwrap()
                .with_input_fact(
                    0,
                    InferenceFact::dt_shape(f32::datum_type(), [1, num_samples, 1]),
                )
                .unwrap();
            infered.into_runnable().unwrap()
        });

        let input_array =
            tract_ndarray::Array3::from_shape_fn((1, num_samples, 1), |(_, s, _)| audio_22050[s]);
        let input_tensor = Tensor::from(input_array);

        let result = runnable.run(tvec!(input_tensor.into())).unwrap();
        let output = result[0].to_plain_array_view::<f32>().unwrap();

        let num_frames = output.shape()[1];
        let num_freq = output.shape()[2] as usize;
        let num_harm = output.shape()[3] as usize;

        let mut features = vec![0.0f32; num_frames * num_harm * num_freq];

        for t in 0..num_frames {
            let frame_out = &mut features[t * num_harm * num_freq..];
            for f in 0..num_freq {
                for h in 0..num_harm {
                    frame_out[h * num_freq + f] = output[[0, t, f, h]];
                }
            }
        }

        (features, num_frames)
    }
}
