use std::collections::HashMap;

use inference_types::ValidationAmendment;

/// Collapse per-confusion proposals into the smallest coherent set of edits.
///
/// Priorities:
/// 1. Keep at most one description rewrite per message (last wins, shortest preferred on ties).
/// 2. Prefer description edits over validation reassignment/removal for the same example.
/// 3. Drop no-op description changes.
pub fn collate_minimal_amendments(proposals: &[ValidationAmendment]) -> Vec<ValidationAmendment> {
    let mut description_by_message: HashMap<i32, ValidationAmendment> = HashMap::new();
    let mut validation_by_example: HashMap<i32, ValidationAmendment> = HashMap::new();

    for proposal in proposals {
        match proposal {
            ValidationAmendment::Description {
                message_id,
                old_description,
                new_description,
                ..
            } => {
                if new_description.trim() == old_description.trim()
                    || new_description.trim().is_empty()
                {
                    continue;
                }
                description_by_message
                    .entry(*message_id)
                    .and_modify(|existing| {
                        if let (
                            ValidationAmendment::Description {
                                new_description: existing_new,
                                ..
                            },
                            ValidationAmendment::Description {
                                new_description: incoming_new,
                                ..
                            },
                        ) = (&*existing, proposal)
                        {
                            // Prefer the shorter clarification when both rewrite the same message.
                            if incoming_new.chars().count() < existing_new.chars().count() {
                                *existing = proposal.clone();
                            }
                        }
                    })
                    .or_insert_with(|| proposal.clone());
            }
            ValidationAmendment::ReassignValidation { example_id, .. }
            | ValidationAmendment::RemoveValidation { example_id, .. } => {
                validation_by_example.insert(*example_id, proposal.clone());
            }
        }
    }

    // If an example also got a description fix involving its expected/predicted
    // categories, drop the validation mutation — prefer description changes.
    let description_message_ids: std::collections::HashSet<i32> =
        description_by_message.keys().copied().collect();
    validation_by_example.retain(|_, amendment| match amendment {
        ValidationAmendment::ReassignValidation {
            from_message_id,
            to_message_id,
            ..
        } => {
            !description_message_ids.contains(from_message_id)
                && !description_message_ids.contains(to_message_id)
        }
        ValidationAmendment::RemoveValidation { message_id, .. } => {
            !description_message_ids.contains(message_id)
        }
        ValidationAmendment::Description { .. } => true,
    });

    let mut out: Vec<_> = description_by_message
        .into_values()
        .chain(validation_by_example.into_values())
        .collect();
    out.sort_by_key(|a| match a {
        ValidationAmendment::Description { message_id, .. } => (0, *message_id),
        ValidationAmendment::ReassignValidation { example_id, .. } => (1, *example_id),
        ValidationAmendment::RemoveValidation { example_id, .. } => (2, *example_id),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_shorter_description_and_drops_validation_when_desc_exists() {
        let proposals = vec![
            ValidationAmendment::Description {
                message_id: 1,
                category_name: "A".into(),
                old_description: "old".into(),
                new_description: "a much longer rewrite of the category description".into(),
                rationale: "long".into(),
            },
            ValidationAmendment::Description {
                message_id: 1,
                category_name: "A".into(),
                old_description: "old".into(),
                new_description: "short clarify".into(),
                rationale: "short".into(),
            },
            ValidationAmendment::ReassignValidation {
                example_id: 9,
                example_text: "x".into(),
                from_message_id: 1,
                from_category: "A".into(),
                to_message_id: 2,
                to_category: "B".into(),
                rationale: "maybe".into(),
            },
        ];
        let out = collate_minimal_amendments(&proposals);
        assert_eq!(out.len(), 1);
        match &out[0] {
            ValidationAmendment::Description { new_description, .. } => {
                assert_eq!(new_description, "short clarify");
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}
