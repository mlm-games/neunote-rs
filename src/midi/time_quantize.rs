use serde::{Deserialize, Serialize};

use crate::midi::events::NoteEvent;

/// Time division for quantization grid
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TimeDivision {
    Whole = 0,
    Half,
    Third,
    Quarter,
    Sixth,
    Eighth,
    Twelfth,
    Sixteenth,
    TwentyFourth,
    ThirtySecond,
    FortyEighth,
    SixtyFourth,
}

impl TimeDivision {
    pub fn as_f64(self) -> f64 {
        match self {
            TimeDivision::Whole => 1.0,
            TimeDivision::Half => 1.0 / 2.0,
            TimeDivision::Third => 1.0 / 3.0,
            TimeDivision::Quarter => 1.0 / 4.0,
            TimeDivision::Sixth => 1.0 / 6.0,
            TimeDivision::Eighth => 1.0 / 8.0,
            TimeDivision::Twelfth => 1.0 / 12.0,
            TimeDivision::Sixteenth => 1.0 / 16.0,
            TimeDivision::TwentyFourth => 1.0 / 24.0,
            TimeDivision::ThirtySecond => 1.0 / 32.0,
            TimeDivision::FortyEighth => 1.0 / 48.0,
            TimeDivision::SixtyFourth => 1.0 / 64.0,
        }
    }
}

/// Information about DAW transport state for time quantization reference
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimeQuantizeInfo {
    pub bpm: f64,
    pub time_signature_num: u32,
    pub time_signature_denom: u32,
    pub ref_last_bar_qn: f64,
    pub ref_position_qn: f64,
    pub ref_position_seconds: f64,
}

impl Default for TimeQuantizeInfo {
    fn default() -> Self {
        Self {
            bpm: 120.0,
            time_signature_num: 4,
            time_signature_denom: 4,
            ref_last_bar_qn: 0.0,
            ref_position_qn: 0.0,
            ref_position_seconds: 0.0,
        }
    }
}

impl TimeQuantizeInfo {
    pub fn qn_to_sec(duration_qn: f64, bpm: f64) -> f64 {
        duration_qn * 60.0 / bpm
    }

    pub fn sec_to_qn(duration_sec: f64, bpm: f64) -> f64 {
        duration_sec * bpm / 60.0
    }

    pub fn start_qn(&self) -> f64 {
        self.ref_position_qn - Self::sec_to_qn(self.ref_position_seconds, self.bpm)
    }

    pub fn start_last_bar_qn(&self) -> f64 {
        let bar_dur = self.time_signature_num as f64 * 4.0 / self.time_signature_denom as f64;
        let n_bars = ((self.ref_last_bar_qn - self.start_qn()) / bar_dur).ceil();
        self.ref_last_bar_qn - n_bars * bar_dur
    }
}

/// Configuration for time quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeQuantizeOptions {
    pub enabled: bool,
    pub division: TimeDivision,
    pub quantize_force: f32, // 0.0 to 1.0
}

impl Default for TimeQuantizeOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            division: TimeDivision::Quarter,
            quantize_force: 0.0,
        }
    }
}

impl TimeQuantizeOptions {
    /// Quantize note events to a rhythmic grid
    pub fn quantize(&self, events: &[NoteEvent], info: &TimeQuantizeInfo) -> Vec<NoteEvent> {
        if !self.enabled {
            return events.to_vec();
        }

        let division_duration = self.division.as_f64();
        let start_pos_qn = info.start_qn() - info.start_last_bar_qn();

        events
            .iter()
            .map(|e| {
                let duration = e.duration();
                let new_start = quantize_time(
                    e.start_time,
                    info.bpm,
                    division_duration,
                    start_pos_qn,
                    self.quantize_force,
                );
                NoteEvent {
                    start_time: new_start,
                    end_time: new_start + duration,
                    ..e.clone()
                }
            })
            .collect()
    }
}

/// Quantize a single time value to the nearest grid division
fn quantize_time(
    event_time: f64,
    bpm: f64,
    division: f64,
    start_qn_offset: f64,
    force: f32,
) -> f64 {
    let sec_per_qn = 60.0 / bpm;
    let div_dur = division * 4.0 * sec_per_qn;
    let origin_shift = start_qn_offset * sec_per_qn;
    let shifted = event_time + origin_shift;

    let since_prev = shifted % div_dur;
    let prev_div = shifted - since_prev;
    let target = if since_prev < div_dur / 2.0 {
        prev_div
    } else {
        prev_div + div_dur
    };

    let quantized = shift_toward(shifted, target, force as f64);
    quantized - origin_shift
}

fn shift_toward(value: f64, target: f64, amount: f64) -> f64 {
    value + (target - value) * amount
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qn_conversions() {
        let bpm = 120.0;
        assert!((TimeQuantizeInfo::qn_to_sec(1.0, bpm) - 0.5).abs() < 1e-9);
        assert!((TimeQuantizeInfo::sec_to_qn(0.5, bpm) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_quantize_identity_when_disabled() {
        let opts = TimeQuantizeOptions {
            enabled: false,
            ..Default::default()
        };
        let info = TimeQuantizeInfo::default();
        let events = vec![
            NoteEvent {
                start_time: 0.1,
                end_time: 1.0,
                ..Default::default()
            },
        ];
        let out = opts.quantize(&events, &info);
        assert_eq!(out[0].start_time, 0.1);
    }

    #[test]
    fn test_quantize_with_force() {
        let opts = TimeQuantizeOptions {
            enabled: true,
            division: TimeDivision::Quarter,
            quantize_force: 1.0,
        };
        let info = TimeQuantizeInfo::default();
        let events = vec![
            NoteEvent {
                start_time: 0.51,
                end_time: 1.0,
                ..Default::default()
            },
        ];
        // At 120 BPM, quarter note = 0.5s. 0.51s should snap to 0.5s
        let out = opts.quantize(&events, &info);
        assert!((out[0].start_time - 0.5).abs() < 1e-6);
    }
}
