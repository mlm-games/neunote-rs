use crate::ml::constants::*;

fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.max(0.0);
    }
}

fn sigmoid_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// A single Conv1D layer operating along the feature (frequency) axis.
struct Conv1DLayer {
    weights: Vec<Vec<Vec<f32>>>, // [out_ch][in_ch][kernel_pos]
    bias: Vec<f32>,
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    in_features: usize,
    out_features: usize,
}

impl Conv1DLayer {
    fn new(
        weights: Vec<Vec<Vec<f32>>>,
        bias: Vec<f32>,
        in_features: usize,
        padding_same: bool,
    ) -> Self {
        let out_ch = weights.len();
        let in_ch = weights[0].len();
        let kernel_size = weights[0][0].len();
        let out_features = if padding_same {
            in_features
        } else {
            in_features + 1 - kernel_size
        };
        Self {
            weights,
            bias,
            in_ch,
            out_ch,
            kernel_size,
            in_features,
            out_features,
        }
    }

    /// Same-padding 1D convolution along the feature axis
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let kw = self.kernel_size;
        let kw_half = kw / 2;
        let mut output = vec![0.0; self.out_ch * self.out_features];

        for oc in 0..self.out_ch {
            for pos in 0..self.out_features {
                let mut sum = 0.0;
                for ic in 0..self.in_ch {
                    for k in 0..kw {
                        let src = (pos as i32 + k as i32 - kw_half as i32)
                            .clamp(0, self.in_features as i32 - 1) as usize;
                        sum +=
                            self.weights[oc][ic][k] * input[ic * self.in_features + src];
                    }
                }
                output[oc * self.out_features + pos] = sum + self.bias[oc];
            }
        }
        output
    }
}

/// Conv2D layer — kernel_size_time Conv1D layers, one per time step,
/// applied to adjacent frames and summed.
pub struct Conv2D {
    layers: Vec<Conv1DLayer>,
    pub num_filters_in: usize,
    pub num_filters_out: usize,
    pub num_features_in: usize,
    pub num_features_out: usize,
    pub kernel_size_time: usize,
    pub in_size: usize,
    pub out_size: usize,
}

impl Conv2D {
    pub fn new(
        weights_flat: &[f32],
        bias: &[f32],
        num_filters_in: usize,
        num_filters_out: usize,
        num_features_in: usize,
        kernel_size_time: usize,
        kernel_size_feature: usize,
        valid_pad: bool,
    ) -> Self {
        let padding_same = !valid_pad;
        let num_features_out = if padding_same {
            num_features_in
        } else {
            num_features_in + 1 - kernel_size_feature
        };

        let t_stride = kernel_size_feature * num_filters_in * num_filters_out;
        let mut layers = Vec::with_capacity(kernel_size_time);

        for t in 0..kernel_size_time {
            let mut w: Vec<Vec<Vec<f32>>> =
                vec![vec![vec![0.0; kernel_size_feature]; num_filters_in]; num_filters_out];
            for kf in 0..kernel_size_feature {
                for fi in 0..num_filters_in {
                    for fo in 0..num_filters_out {
                        let src = t * t_stride
                            + kf * num_filters_in * num_filters_out
                            + fi * num_filters_out
                            + fo;
                        w[fo][fi][kf] = weights_flat[src];
                    }
                }
            }
            layers.push(Conv1DLayer::new(
                w,
                bias.to_vec(),
                num_features_in,
                padding_same,
            ));
        }

        Self {
            layers,
            num_filters_in,
            num_filters_out,
            num_features_in,
            num_features_out,
            kernel_size_time,
            in_size: num_filters_in * num_features_in,
            out_size: num_filters_out * num_features_out,
        }
    }

    /// Batch forward: input [num_frames * in_size], output [num_frames * out_size]
    pub fn forward_batch(&self, input: &[f32], output: &mut [f32], num_frames: usize) {
        let half_t = self.kernel_size_time / 2;
        for f in 0..num_frames {
            let out_start = f * self.out_size;
            let mut accum = vec![0.0; self.out_size];

            for t in 0..self.kernel_size_time {
                let src_f = (f as i32 + t as i32 - half_t as i32)
                    .clamp(0, num_frames as i32 - 1) as usize;
                let frame_in = &input[src_f * self.in_size..][..self.in_size];
                let conv_out = self.layers[t].forward(frame_in);
                for i in 0..self.out_size {
                    accum[i] += conv_out[i];
                }
            }

            output[out_start..out_start + self.out_size].copy_from_slice(&accum);
        }
    }
}

// ============================================================================
// Complete Basic Pitch CNN: 4 sub-models with their wiring
// ============================================================================

pub struct BasicPitchCNN {
    contour_conv1: Conv2D,   // 8→8, 264feat, kernel_t=3, kernel_f=39
    contour_conv2: Conv2D,   // 8→1, 264feat, kernel_t=5, kernel_f=5
    note_conv1: Conv2D,      // 1→32, 264feat, kernel_t=7, kernel_f=7
    note_conv2: Conv2D,      // 32→1, 88feat, kernel_t=7, kernel_f=3
    onset_input_conv: Conv2D, // 8→32, 264feat, kernel_t=5, kernel_f=5
    onset_output_conv: Conv2D,// 33→1, 88feat, kernel_t=3, kernel_f=3
}

impl BasicPitchCNN {
    /// Weights and biases for all 6 conv layers.
    /// Each weight array is flat in [t][kf][fi][fo] order (RTNeural JSON format).
    pub fn new(
        contour_w1: &[f32], contour_b1: &[f32],
        contour_w2: &[f32], contour_b2: &[f32],
        note_w1: &[f32], note_b1: &[f32],
        note_w2: &[f32], note_b2: &[f32],
        onset1_w: &[f32], onset1_b: &[f32],
        onset2_w: &[f32], onset2_b: &[f32],
    ) -> Self {
        Self {
            contour_conv1: Conv2D::new(contour_w1, contour_b1, 8, 8, 264, 3, 39, false),
            contour_conv2: Conv2D::new(contour_w2, contour_b2, 8, 1, 264, 5, 5, false),
            note_conv1: Conv2D::new(note_w1, note_b1, 1, 32, 264, 7, 7, false),
            note_conv2: Conv2D::new(note_w2, note_b2, 32, 1, 88, 7, 3, false),
            onset_input_conv: Conv2D::new(onset1_w, onset1_b, 8, 32, 264, 5, 5, false),
            onset_output_conv: Conv2D::new(onset2_w, onset2_b, 33, 1, 88, 3, 3, false),
        }
    }

    /// Process all frames at once.
    ///
    /// - `features`: [num_frames * NUM_HARMONICS * NUM_FREQ_IN] stacked CQT features
    /// - `out_contours`: [num_frames * NUM_FREQ_IN], filled with contour posteriorgram values
    /// - `out_notes`: [num_frames * NUM_FREQ_OUT], filled with note posteriorgram values
    /// - `out_onsets`: [num_frames * NUM_FREQ_OUT], filled with onset posteriorgram values
    pub fn process_all_frames(
        &self,
        features: &[f32],
        num_frames: usize,
        out_contours: &mut [f32],
        out_notes: &mut [f32],
        out_onsets: &mut [f32],
    ) {
        // === Contour model: 2112 → 264 ===
        // conv1(8→8, 264feat, 3×39) → ReLU → conv2(8→1, 264feat, 5×5) → Sigmoid → 264
        let mut c1 = vec![0.0; num_frames * 8 * NUM_FREQ_IN];
        self.contour_conv1.forward_batch(features, &mut c1, num_frames);
        relu_inplace(&mut c1);

        let mut c2 = vec![0.0; num_frames * 1 * NUM_FREQ_IN];
        self.contour_conv2.forward_batch(&c1, &mut c2, num_frames);
        sigmoid_inplace(&mut c2);

        // Copy contour output: 1 × 264 per frame
        for f in 0..num_frames {
            let src = f * 1 * NUM_FREQ_IN;
            let dst = f * NUM_FREQ_IN;
            out_contours[dst..dst + NUM_FREQ_IN].copy_from_slice(&c2[src..src + NUM_FREQ_IN]);
        }

        // === Note model: 264 → 88 ===
        // Input is the sigmoid-activated contour output (264 per frame)
        let note_in = out_contours; // 264 per frame
        let mut n1 = vec![0.0; num_frames * 32 * NUM_FREQ_IN];
        self.note_conv1.forward_batch(note_in, &mut n1, num_frames);
        relu_inplace(&mut n1);

        let mut n2 = vec![0.0; num_frames * 1 * 88];
        self.note_conv2.forward_batch(&n1, &mut n2, num_frames);
        sigmoid_inplace(&mut n2);

        // Copy note output: 88 per frame
        for f in 0..num_frames {
            let src = f * 88;
            let dst = f * NUM_FREQ_OUT;
            out_notes[dst..dst + NUM_FREQ_OUT].copy_from_slice(&n2[src..src + 88]);
        }

        // === Onset model ===
        // Onset input conv: 2112 → 32*264 = 8448 per frame (ReLU)
        let mut o1 = vec![0.0; num_frames * 32 * NUM_FREQ_IN];
        self.onset_input_conv.forward_batch(features, &mut o1, num_frames);
        relu_inplace(&mut o1);

        // Concat operation: per frame, combine note output (88) with onset_input
        // to produce 33 * 88 = 2904 values per frame for onset output conv
        let concat_input_size = 33 * NUM_FREQ_OUT;
        let mut concat = vec![0.0; num_frames * concat_input_size];
        for f in 0..num_frames {
            let note_frame = &n2[f * 88..(f + 1) * 88];
            let onset_frame = &o1[f * 32 * NUM_FREQ_IN..(f + 1) * 32 * NUM_FREQ_IN];
            let out_frame = &mut concat[f * concat_input_size..(f + 1) * concat_input_size];

            for i in 0..NUM_FREQ_OUT {
                // First element: note output for this frequency bin
                out_frame[i * 33] = note_frame[i];
                // Next 32 elements: onset_input downsampled from 264→88 (stride 3)
                for j in 0..32 {
                    out_frame[i * 33 + 1 + j] = onset_frame[j * NUM_FREQ_IN + i * 3];
                }
            }
        }

        // Onset output conv: 33*88 → 1*88 per frame (Sigmoid)
        self.onset_output_conv
            .forward_batch(&concat, out_onsets, num_frames);
        sigmoid_inplace(out_onsets);
    }
}
