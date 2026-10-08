use crate::ml::constants::*;
use crate::ml::weights::CnnWeights;

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
/// Input/output layout: [num_channels * in_features + position]  (RTNeural convention)
struct Conv1DLayer {
    weights: Vec<Vec<Vec<f32>>>, // [out_ch][in_ch][kernel_pos]
    bias: Vec<f32>,
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    stride: usize,
    in_features: usize,
    out_features: usize,
}

impl Conv1DLayer {
    fn new(
        weights: Vec<Vec<Vec<f32>>>,
        bias: Vec<f32>,
        in_features: usize,
        stride: usize,
        padding_same: bool,
    ) -> Self {
        let out_ch = weights.len();
        let in_ch = weights[0].len();
        let kernel_size = weights[0][0].len();
        let out_features = if padding_same {
            in_features.div_ceil(stride)
        } else {
            (in_features - kernel_size) / stride + 1
        };
        Self {
            weights,
            bias,
            in_ch,
            out_ch,
            kernel_size,
            stride,
            in_features,
            out_features,
        }
    }

    /// 1D convolution along the feature axis with stride
    /// Input: [in_ch * in_features], Output: [out_ch * out_features]
    /// Layout: [channel * in_features + position] (RTNeural convention)
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        let kw = self.kernel_size;
        let kw_half = kw / 2;
        let s = self.stride;

        for oc in 0..self.out_ch {
            for pos in 0..self.out_features {
                let mut sum = 0.0;
                for k in 0..kw {
                    let src = (pos * s) as i32 + k as i32 - kw_half as i32;
                    if src >= 0 && src < self.in_features as i32 {
                        for ic in 0..self.in_ch {
                            sum += self.weights[oc][ic][k]
                                * input[ic * self.in_features + src as usize];
                        }
                    }
                }
                output[oc * self.out_features + pos] = sum + self.bias[oc];
            }
        }
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weights_flat: &[f32],
        bias: &[f32],
        num_filters_in: usize,
        num_filters_out: usize,
        num_features_in: usize,
        kernel_size_time: usize,
        kernel_size_feature: usize,
        stride: usize,
        valid_pad: bool,
    ) -> Self {
        let padding_same = !valid_pad;
        let num_features_out = if padding_same {
            num_features_in.div_ceil(stride)
        } else {
            (num_features_in - kernel_size_feature) / stride + 1
        };

        let t_stride = kernel_size_feature * num_filters_in * num_filters_out;
        let mut layers = Vec::with_capacity(kernel_size_time);

        for t in 0..kernel_size_time {
            let mut w = Vec::with_capacity(num_filters_out);
            for fo in 0..num_filters_out {
                let mut w_fo = Vec::with_capacity(num_filters_in);
                for fi in 0..num_filters_in {
                    let slice: Vec<f32> = (0..kernel_size_feature)
                        .map(|kf| {
                            weights_flat[t * t_stride
                                + kf * num_filters_in * num_filters_out
                                + fi * num_filters_out
                                + fo]
                        })
                        .collect();
                    w_fo.push(slice);
                }
                w.push(w_fo);
            }
            let layer_bias = if t == 0 {
                bias.to_vec()
            } else {
                vec![0.0; bias.len()]
            };
            layers.push(Conv1DLayer::new(
                w,
                layer_bias,
                num_features_in,
                stride,
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
        let mut conv_out = vec![0.0; self.out_size];
        for f in 0..num_frames {
            let out_start = f * self.out_size;
            let out_slice = &mut output[out_start..out_start + self.out_size];
            out_slice.fill(0.0);

            for t in 0..self.kernel_size_time {
                let src_f =
                    (f as i32 + t as i32 - half_t as i32).clamp(0, num_frames as i32 - 1) as usize;
                let frame_in = &input[src_f * self.in_size..][..self.in_size];
                self.layers[t].forward(frame_in, &mut conv_out);
                for i in 0..self.out_size {
                    out_slice[i] += conv_out[i];
                }
            }
        }
    }
}

// ============================================================================
// Complete Basic Pitch CNN: 4 sub-models with their wiring
// ============================================================================

pub struct BasicPitchCNN {
    contour_conv1: Conv2D,     // 8→8, 264feat, kernel_t=3, kernel_f=39, stride=1
    contour_conv2: Conv2D,     // 8→1, 264feat, kernel_t=5, kernel_f=5, stride=1
    note_conv1: Conv2D,        // 1→32, 264feat, kernel_t=7, kernel_f=7, stride=3
    note_conv2: Conv2D,        // 32→1, 88feat, kernel_t=7, kernel_f=3, stride=1
    onset_input_conv: Conv2D,  // 8→32, 264feat, kernel_t=5, kernel_f=5, stride=3
    onset_output_conv: Conv2D, // 33→1, 88feat, kernel_t=3, kernel_f=3, stride=1
}

impl BasicPitchCNN {
    pub fn new(w: &CnnWeights) -> Self {
        Self {
            contour_conv1: Conv2D::new(&w.contour_w1, &w.contour_b1, 8, 8, 264, 3, 39, 1, false),
            contour_conv2: Conv2D::new(&w.contour_w2, &w.contour_b2, 8, 1, 264, 5, 5, 1, false),
            note_conv1: Conv2D::new(&w.note_w1, &w.note_b1, 1, 32, 264, 7, 7, 3, false),
            note_conv2: Conv2D::new(&w.note_w2, &w.note_b2, 32, 1, 88, 7, 3, 1, false),
            onset_input_conv: Conv2D::new(&w.onset1_w, &w.onset1_b, 8, 32, 264, 5, 5, 3, false),
            onset_output_conv: Conv2D::new(&w.onset2_w, &w.onset2_b, 33, 1, 88, 3, 3, 1, false),
        }
    }

    /// Process all frames at once.
    ///
    /// All buffers use [channel * num_features + position] (RTNeural) layout.
    pub fn process_all_frames(
        &self,
        features: &[f32],
        num_frames: usize,
        out_contours: &mut [f32],
        out_notes: &mut [f32],
        out_onsets: &mut [f32],
    ) {
        const CONV1D_CH: usize = 8;
        const NOTE_CH: usize = 32;

        // === Contour model: 2112 → 264 ===
        let mut c1 = vec![0.0; num_frames * CONV1D_CH * NUM_FREQ_IN];
        self.contour_conv1
            .forward_batch(features, &mut c1, num_frames);
        relu_inplace(&mut c1);

        let mut c2 = vec![0.0; num_frames * NUM_FREQ_IN];
        self.contour_conv2.forward_batch(&c1, &mut c2, num_frames);
        sigmoid_inplace(&mut c2);

        out_contours[..num_frames * NUM_FREQ_IN].copy_from_slice(&c2);

        // === Note model: 264 → 88 ===
        let note_in = out_contours;
        let mut n1 = vec![0.0; num_frames * NOTE_CH * NUM_FREQ_OUT];
        self.note_conv1.forward_batch(note_in, &mut n1, num_frames);
        relu_inplace(&mut n1);

        let mut n2 = vec![0.0; num_frames * NUM_FREQ_OUT];
        self.note_conv2.forward_batch(&n1, &mut n2, num_frames);
        sigmoid_inplace(&mut n2);

        out_notes[..num_frames * NUM_FREQ_OUT].copy_from_slice(&n2);

        // === Onset model ===
        let mut o1 = vec![0.0; num_frames * NOTE_CH * NUM_FREQ_OUT];
        self.onset_input_conv
            .forward_batch(features, &mut o1, num_frames);
        relu_inplace(&mut o1);

        // Concat note + onset_input per position: 33 channels × 88 features
        let concat_input_size = 33 * NUM_FREQ_OUT;
        let mut concat = vec![0.0; num_frames * concat_input_size];
        for f in 0..num_frames {
            let note_frame = &n2[f * NUM_FREQ_OUT..(f + 1) * NUM_FREQ_OUT];
            let onset_frame = &o1[f * NOTE_CH * NUM_FREQ_OUT..(f + 1) * NOTE_CH * NUM_FREQ_OUT];
            let out_frame = &mut concat[f * concat_input_size..(f + 1) * concat_input_size];

            // [channel][feature] layout: ch=0 is note, ch=1..32 are onset_input
            out_frame[..NUM_FREQ_OUT].copy_from_slice(note_frame);
            for j in 0..NOTE_CH {
                for i in 0..NUM_FREQ_OUT {
                    out_frame[(1 + j) * NUM_FREQ_OUT + i] = onset_frame[j * NUM_FREQ_OUT + i];
                }
            }
        }

        self.onset_output_conv
            .forward_batch(&concat, out_onsets, num_frames);
        sigmoid_inplace(out_onsets);
    }
}
