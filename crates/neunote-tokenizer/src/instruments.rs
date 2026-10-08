//! Hard instrument filtering: the token mask that backs the soft conditioning.

use neunote_types::GroupId;

use crate::vocab::{DRUM_COUNT, DRUM_FIRST, EventType, PROGRAM_COUNT, PROGRAM_FIRST, token_for};

/// Token ids the model may not produce, given a selection.
///
/// Every `program` token that is not a selected group's representative, and
/// every `drum` token unless `drums` is selected. Timing, pitch, velocity, tie
/// and the special tokens are never masked.
///
/// An empty selection is an error: the reference forbids every program and
/// every drum for one, which is not a useful thing to hand a decoder.
pub fn forbidden_token_ids(groups: &[GroupId]) -> Result<Vec<i32>, String> {
    if groups.is_empty() {
        return Err(
            "forbidden_token_ids needs a non-empty selection: the reference forbids every \
             program and every drum for an empty one"
                .to_owned(),
        );
    }

    let allow_drums = groups.contains(&GroupId::DRUMS);

    // Drums contributes no allowed program: it is not a program group, and the
    // reference skips it here rather than allowing its representative, which is
    // 96 because group 36 is also program 96's singleton group.
    let mut allowed = Vec::with_capacity(groups.len());
    for group in groups {
        if *group == GroupId::DRUMS {
            continue;
        }
        if let Some(program) = group.representative_program() {
            allowed.push(program as i32);
        }
    }

    let mut forbidden = Vec::with_capacity(PROGRAM_COUNT as usize + DRUM_COUNT as usize);
    for program in 0..PROGRAM_COUNT {
        if !allowed.contains(&program) {
            forbidden.push(PROGRAM_FIRST + program);
        }
    }

    if !allow_drums {
        for drum in 0..DRUM_COUNT {
            forbidden.push(DRUM_FIRST + drum);
        }
    }

    forbidden.sort_unstable();
    Ok(forbidden)
}

/// Which conditioning embedding rows a selection writes, one per group. With no
/// selection the caller writes exactly one position instead: the null class.
pub fn conditioning_rows(groups: &[GroupId]) -> Vec<i32> {
    groups
        .iter()
        .map(|group| group.conditioning_row())
        .collect()
}

#[allow(dead_code)]
fn program_token(program: i32) -> i32 {
    token_for(EventType::Program, program).unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piano_only_allows_exactly_the_piano_program() {
        // Piano only: program 0 survives, every other program and every drum does not.
        let forbidden = forbidden_token_ids(&[GroupId(0)]).unwrap();
        assert!(!forbidden.contains(&PROGRAM_FIRST));
        assert!(forbidden.contains(&DRUM_FIRST));
        assert!(forbidden.contains(&(PROGRAM_FIRST + 33)));
        assert_eq!(
            forbidden.len(),
            PROGRAM_COUNT as usize - 1 + DRUM_COUNT as usize
        );
    }

    #[test]
    fn drums_selection_unmasks_drum_tokens_but_not_program_96() {
        let forbidden = forbidden_token_ids(&[GroupId::DRUMS]).unwrap();
        assert!(!forbidden.contains(&DRUM_FIRST));
        // Drums is not a program group, so program 96 stays forbidden.
        assert!(forbidden.contains(&(PROGRAM_FIRST + 96)));
        assert_eq!(forbidden.len(), PROGRAM_COUNT as usize);
    }

    #[test]
    fn timing_pitch_velocity_and_tie_are_never_masked() {
        let forbidden = forbidden_token_ids(&[GroupId(0)]).unwrap();
        for id in 0..crate::vocab::PITCH_FIRST {
            assert!(!forbidden.contains(&id), "timing token {id} masked");
        }
        for id in crate::vocab::VELOCITY_FIRST..crate::vocab::PROGRAM_FIRST {
            assert!(!forbidden.contains(&id), "event token {id} masked");
        }
    }

    #[test]
    fn an_empty_selection_is_rejected() {
        assert!(forbidden_token_ids(&[]).is_err());
    }
}
