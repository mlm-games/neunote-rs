/// Simple audio resampler using linear interpolation
/// Downsamples from arbitrary sample rate to BASIC_PITCH_SAMPLE_RATE (22050 Hz)
pub struct Resampler {
    /// Low-pass filter state (first-order IIR)
    lp_state: f32,
    lp_coeff: f32, // coefficient for lowpass at nyquist of target
}

impl Default for Resampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Resampler {
    pub fn new() -> Self {
        Self {
            lp_state: 0.0,
            lp_coeff: 0.0,
        }
    }

    /// Prepare resampler for given input/output rates
    pub fn prepare(&mut self, input_rate: f64, output_rate: f64) {
        // Simple Butterworth-style lowpass at output nyquist
        let cutoff = (output_rate / 2.0) / input_rate;
        self.lp_coeff = cutoff.min(0.45) as f32;
    }

    /// Resample audio from input_rate to 22050 Hz
    pub fn process(&mut self, input: &[f32], input_rate: f64, output_rate: f64) -> Vec<f32> {
        if input_rate == output_rate {
            return input.to_vec();
        }

        let ratio = input_rate / output_rate;
        let output_len = (input.len() as f64 / ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        let mut frac: f64 = 0.0;
        let mut i: usize = 0;

        while i < input.len() - 1 {
            // Linear interpolation
            let sample = input[i] as f64 * (1.0 - frac) + input[i + 1] as f64 * frac;

            // Lowpass filter
            let filtered = self.lp_state + self.lp_coeff * (sample as f32 - self.lp_state);
            self.lp_state = filtered;

            output.push(filtered);

            frac += ratio;
            let advance = frac as usize;
            i += advance;
            frac -= advance as f64;
        }

        output
    }

    pub fn reset(&mut self) {
        self.lp_state = 0.0;
    }
}
