use crate::midi::events::NoteEvent;
use crate::ml::constants::*;

/// Parameters for note event extraction
#[derive(Debug, Clone)]
pub struct ConvertParams {
    pub onset_threshold: f32,
    pub frame_threshold: f32,
    pub min_note_length: usize,
    pub infer_onsets: bool,
    pub max_frequency: f32,
    pub min_frequency: f32,
    pub melodia_trick: bool,
    pub pitch_bend: PitchBendMode,
    pub energy_threshold: usize,
}

impl Default for ConvertParams {
    fn default() -> Self {
        Self {
            onset_threshold: 0.3,
            frame_threshold: 0.5,
            min_note_length: 11,
            infer_onsets: true,
            max_frequency: -1.0,
            min_frequency: -1.0,
            melodia_trick: true,
            pitch_bend: PitchBendMode::Multi,
            energy_threshold: 11,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PitchBendMode {
    None,
    Single,
    Multi,
}

/// Convert posteriorgrams into NoteEvent vectors
pub fn posteriorgrams_to_notes(
    notes_pg: &[f32],
    onsets_pg: &[f32],
    contours_pg: &[f32],
    num_frames: usize,
    params: &ConvertParams,
) -> Vec<NoteEvent> {
    if num_frames == 0 {
        return vec![];
    }

    let n_notes = NUM_FREQ_OUT;
    let n_contour = NUM_FREQ_IN;
    let last_frame = num_frames - 1;

    // Inferred onsets
    let onsets = if params.infer_onsets {
        inferred_onsets(onsets_pg, notes_pg, num_frames, n_notes)
    } else {
        onsets_pg.to_vec()
    };

    // Remaining energy (copy of notes_pg)
    let mut remaining = notes_pg.to_vec();

    // Precompute index list for melodia trick
    let mut remaining_idx: Vec<(usize, usize)> = (0..num_frames)
        .flat_map(|f| (0..n_notes).map(move |n| (f, n)))
        .collect();

    // Frequency constraints
    let max_note_idx = if params.max_frequency < 0.0 {
        n_notes - 1
    } else {
        (hz_to_midi(params.max_frequency).saturating_sub(MIDI_OFFSET as u8)) as usize
    };
    let min_note_idx = if params.min_frequency < 0.0 {
        0
    } else {
        (hz_to_midi(params.min_frequency).saturating_sub(MIDI_OFFSET as u8)) as usize
    };

    let mut events: Vec<NoteEvent> = Vec::new();

    // Backward pass: find onsets via argrelmax
    for frame_idx in (0..last_frame).rev() {
        for note_idx in (min_note_idx..=max_note_idx.min(n_notes - 1)).rev() {
            let onset = onsets[frame_idx * n_notes + note_idx];
            let prev = if frame_idx > 0 {
                onsets[(frame_idx - 1) * n_notes + note_idx]
            } else {
                onset
            };
            let next = onsets[(frame_idx + 1) * n_notes + note_idx];

            if onset < params.onset_threshold || onset < prev || onset < next {
                continue;
            }

            let mut i = frame_idx + 1;
            let mut k = 0;
            while i < last_frame && k < params.energy_threshold {
                if remaining[i * n_notes + note_idx] < params.frame_threshold {
                    k += 1;
                } else {
                    k = 0;
                }
                i += 1;
            }
            i -= k;

            if i - frame_idx <= params.min_note_length {
                continue;
            }

            let mut amplitude = 0.0f64;
            for f in frame_idx..i {
                amplitude += remaining[f * n_notes + note_idx] as f64;
                remaining[f * n_notes + note_idx] = 0.0;
                if note_idx < MAX_NOTE_IDX {
                    remaining[f * n_notes + note_idx + 1] = 0.0;
                }
                if note_idx > 0 {
                    remaining[f * n_notes + note_idx - 1] = 0.0;
                }
            }
            amplitude /= (i - frame_idx) as f64;

            events.push(NoteEvent {
                start_time: model_frame_to_time(frame_idx),
                end_time: model_frame_to_time(i),
                start_frame: frame_idx,
                end_frame: i,
                pitch: (note_idx + MIDI_OFFSET) as u8,
                amplitude: amplitude as f32,
                bends: vec![],
            });
        }
    }

    // Melodia trick: pick remaining high-energy points
    if params.melodia_trick {
        remaining_idx.sort_by(|&(f1, n1), &(f2, n2)| {
            remaining[f2 * n_notes + n2]
                .partial_cmp(&remaining[f1 * n_notes + n1])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &(frame_idx, note_idx) in &remaining_idx {
            if remaining[frame_idx * n_notes + note_idx] == 0.0 {
                continue;
            }
            if remaining[frame_idx * n_notes + note_idx] <= params.frame_threshold {
                break;
            }
            remaining[frame_idx * n_notes + note_idx] = 0.0;

            // Forward pass
            let mut fi = frame_idx + 1;
            let mut fk = 0;
            while fi < last_frame && fk < params.energy_threshold {
                fk = inhibit(&mut remaining, n_notes, fi, note_idx, params.frame_threshold, fk);
                fi += 1;
            }
            let i_end = fi - 1 - fk;

            // Backward pass (using isize for signed arithmetic)
            let mut bi = frame_idx as isize - 1;
            let mut bk = 0;
            while bi > 0 && bk < params.energy_threshold as isize {
                bk = inhibit(&mut remaining, n_notes, bi as usize, note_idx, params.frame_threshold, bk as usize) as isize;
                bi -= 1;
            }
            let i_start = (bi + 1 + bk) as usize;

            if i_end - i_start <= params.min_note_length {
                continue;
            }

            let mut amplitude = 0.0f64;
            for f in i_start..i_end {
                amplitude += notes_pg[f * n_notes + note_idx] as f64;
            }
            amplitude /= (i_end - i_start) as f64;

            events.push(NoteEvent {
                start_time: model_frame_to_time(i_start),
                end_time: model_frame_to_time(i_end),
                start_frame: i_start,
                end_frame: i_end,
                pitch: (note_idx + MIDI_OFFSET) as u8,
                amplitude: amplitude as f32,
                bends: vec![],
            });
        }
    }

    // Sort events
    events.sort_by(|a, b| {
        a.start_frame
            .cmp(&b.start_frame)
            .then(a.end_frame.cmp(&b.end_frame))
    });

    // Pitch bends
    if params.pitch_bend != PitchBendMode::None {
        add_pitch_bends(&mut events, contours_pg, num_frames, n_contour);
        if params.pitch_bend == PitchBendMode::Single {
            drop_overlapping_pitch_bends(&mut events);
        }
    }

    events
}

fn inhibit(
    pg: &mut [f32],
    n_notes: usize,
    frame_i: usize,
    note_i: usize,
    threshold: f32,
    k: usize,
) -> usize {
    let new_k = if pg[frame_i * n_notes + note_i] < threshold {
        k + 1
    } else {
        0
    };
    pg[frame_i * n_notes + note_i] = 0.0;
    if note_i < MAX_NOTE_IDX {
        pg[frame_i * n_notes + note_i + 1] = 0.0;
    }
    if note_i > 0 {
        pg[frame_i * n_notes + note_i - 1] = 0.0;
    }
    new_k
}

/// Infer onsets by computing differences in note posteriorgrams
fn inferred_onsets(
    onsets_pg: &[f32],
    notes_pg: &[f32],
    num_frames: usize,
    n_notes: usize,
) -> Vec<f32> {
    let num_diffs = 2;
    let mut notes_diff = vec![1.0f32; num_frames * n_notes];

    let mut max_min_notes_diff = 0.0f32;
    let mut max_onset = 0.0f32;

    for n in 0..num_diffs {
        let offset = n + 1;
        for i in 0..num_frames {
            let i_behind = i as i32 - offset as i32;
            for j in 0..n_notes {
                let behind = if i_behind >= 0 {
                    notes_pg[i_behind as usize * n_notes + j]
                } else {
                    0.0
                };
                let diff = notes_pg[i * n_notes + j] - behind;

                let idx = i * n_notes + j;
                if diff < notes_diff[idx] {
                    let diff = if diff < 0.0 { 0.0 } else { diff };
                    notes_diff[idx] = if i >= num_diffs { diff } else { 0.0 };
                }

                if n == num_diffs - 1 {
                    let onset = onsets_pg[idx];
                    if onset > max_onset {
                        max_onset = onset;
                    }
                    if notes_diff[idx] > max_min_notes_diff {
                        max_min_notes_diff = notes_diff[idx];
                    }
                }
            }
        }
    }

    if max_min_notes_diff > 0.0 {
        for val in notes_diff.iter_mut() {
            *val = max_onset * *val / max_min_notes_diff;
        }
    }

    for i in 0..num_frames * n_notes {
        notes_diff[i] = notes_diff[i].max(onsets_pg[i]);
    }

    notes_diff
}

fn model_frame_to_time(frame: usize) -> f64 {
    (frame * FFT_HOP) as f64 / BASIC_PITCH_SAMPLE_RATE
}

/// Add pitch bend values from contour posteriorgrams
fn add_pitch_bends(
    events: &mut [NoteEvent],
    contours_pg: &[f32],
    num_frames: usize,
    n_contour: usize,
) {
    let tolerance = 25i32;
    let gauss_std = 5.0f32;

    for event in events.iter_mut() {
        let note_idx = (CONTOURS_BINS_PER_SEMITONE as i32)
            * (event.pitch as i32 - 69
                + 12 * (440.0f32 / ANNOTATIONS_BASE_FREQUENCY).log2().round() as i32);

        let note_start_idx = (note_idx - tolerance).max(0) as usize;
        let note_end_idx = (N_FREQ_BINS_CONTOURS as i32).min(note_idx + tolerance + 1) as usize;

        // Compute gauss_start and pb_shift using the actual note_idx for cont/clamping
        let note_start_unclamped = note_idx - tolerance;
        let gauss_start = (-note_start_unclamped).max(0);
        let pb_shift = tolerance - (-note_start_unclamped).max(0);

        for i in event.start_frame..event.end_frame.min(num_frames - 1) {
            let mut best_bend = 0i32;
            let mut max_val = 0.0f32;

            for (j, j_global) in (note_start_idx..note_end_idx).enumerate() {
                let x = (gauss_start + j as i32) as f32;
                let n = x - tolerance as f32;
                let w = (-(n * n) / (2.0 * gauss_std * gauss_std)).exp()
                    * contours_pg[i * n_contour + j_global];

                if w > max_val {
                    best_bend = j as i32;
                    max_val = w;
                }
            }
            event.bends.push(best_bend - pb_shift as i32);
        }
    }
}

fn drop_overlapping_pitch_bends(events: &mut [NoteEvent]) {
    for i in 0..events.len().saturating_sub(1) {
        for j in i + 1..events.len() {
            if events[j].start_frame >= events[i].end_frame {
                break;
            }
            events[i].bends.clear();
            events[j].bends.clear();
        }
    }
}

fn hz_to_midi(hz: f32) -> u8 {
    (12.0 * (hz / 440.0).log2() + 69.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_note_creation() {
        let num_frames = 100;
        let n_notes = NUM_FREQ_OUT;
        let mut notes_pg = vec![0.0f32; num_frames * n_notes];
        let mut onsets_pg = vec![0.0f32; num_frames * n_notes];
        let contours_pg = vec![0.0f32; num_frames * NUM_FREQ_IN];

        let note_idx = 39;
        for f in 10..50 {
            notes_pg[f * n_notes + note_idx] = 0.9;
        }
        onsets_pg[10 * n_notes + note_idx] = 0.9;
        onsets_pg[11 * n_notes + note_idx] = 0.3;
        onsets_pg[9 * n_notes + note_idx] = 0.3;

        for f in 10..50 {
            if note_idx > 0 {
                notes_pg[f * n_notes + note_idx - 1] = 0.1;
            }
            if note_idx < MAX_NOTE_IDX {
                notes_pg[f * n_notes + note_idx + 1] = 0.1;
            }
        }

        let params = ConvertParams {
            onset_threshold: 0.5,
            frame_threshold: 0.3,
            min_note_length: 3,
            infer_onsets: false,
            ..Default::default()
        };

        let events = posteriorgrams_to_notes(
            &notes_pg,
            &onsets_pg,
            &contours_pg,
            num_frames,
            &params,
        );

        assert!(!events.is_empty(), "Should have at least one event");
        let has_c4 = events.iter().any(|e| e.pitch == 60);
        assert!(has_c4, "Should contain MIDI note 60 (C4)");
    }
}
