use std::collections::HashMap;

use inference_types::{
    AnalysisExecutiveSummary, ConfusionCase, FailedAmendmentAttempt, ValidationAmendment,
};
use serde::Deserialize;

use super::dataset::ValidationDataset;
use super::evaluate::ExampleEvalResult;

#[derive(Debug, Deserialize)]
struct BatchAnalysisResponse {
    /// One entry per confusion pair (keyed by example_id).
    analyses: Vec<PerConfusionAnalysis>,
    #[serde(default)]
    executive_summary: Option<ExecutiveSummaryJson>,
}

#[derive(Debug, Deserialize)]
struct ExecutiveSummaryJson {
    #[serde(default)]
    category_priority_assumptions: Vec<String>,
    #[serde(default)]
    differentiation_decisions: Vec<String>,
    #[serde(default)]
    label_challenges: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct PerConfusionAnalysis {
    example_id: i32,
    /// Short explanation of why the local model chose the wrong category.
    diagnosis: String,
    /// Prefer description_gap when categories overlap on the HCP text.
    #[serde(default)]
    verdict: Option<String>,
    /// Required for repeat offenders: why EACH listed prior amendment failed.
    #[serde(default)]
    prior_amendment_failures: Vec<PriorAmendmentFailureJson>,
    #[serde(default)]
    amendments: Vec<AnalysisAmendment>,
}

#[derive(Debug, Deserialize)]
struct PriorAmendmentFailureJson {
    /// 1-based index into the numbered prior-amendment list for this example.
    attempt_index: u32,
    why_failed: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum AnalysisAmendment {
    Description {
        message_id: i32,
        new_description: String,
        rationale: String,
    },
    ReassignValidation {
        example_id: i32,
        to_message_id: i32,
        rationale: String,
    },
    RemoveValidation {
        example_id: i32,
        rationale: String,
    },
}

/// Original descriptions loaded at loop start (before any in-memory edits).
pub type OriginalDescriptions = HashMap<i32, String>;

/// Cross-round memory of every amendment that failed to clear a confusion pair.
#[derive(Debug, Clone, Default)]
pub struct ProgressLedger {
    /// Chronological per-amendment failure records (full history for prompts).
    pub attempts: Vec<FailedAmendmentAttempt>,
}

impl ProgressLedger {
    pub fn format_for_prompt(&self) -> String {
        if self.attempts.is_empty() {
            return "None yet — no failed-amendment history recorded.".into();
        }
        self.attempts
            .iter()
            .enumerate()
            .map(|(i, a)| {
                format!(
                    "{n}. [recorded round {round}] example={ex} {expected} → {predicted:?}\n   amendment: {summary}\n   why it failed: {why}",
                    n = i + 1,
                    round = a.round_observed,
                    ex = a.example_id,
                    expected = a.expected_category,
                    predicted = a.predicted_category,
                    summary = a.amendment_summary,
                    why = a.why_it_failed,
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn for_example(&self, example_id: i32) -> Vec<&FailedAmendmentAttempt> {
        self.attempts
            .iter()
            .filter(|a| a.example_id == example_id)
            .collect()
    }

    pub fn extend(&mut self, new_attempts: Vec<FailedAmendmentAttempt>) {
        for incoming in new_attempts {
            // Update reason if we already tracked the same amendment summary for this example.
            if let Some(existing) = self.attempts.iter_mut().find(|a| {
                a.example_id == incoming.example_id
                    && normalize_desc(&a.amendment_summary)
                        == normalize_desc(&incoming.amendment_summary)
            }) {
                existing.why_it_failed = incoming.why_it_failed;
                existing.round_observed = incoming.round_observed;
            } else {
                self.attempts.push(incoming);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct NumberedPriorAttempt {
    pub index: u32,
    pub amendment_summary: String,
    pub known_why: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RepeatContext {
    pub recurrence_count: u32,
    pub prior_diagnoses: Vec<String>,
    /// Numbered list of ALL prior amendments relevant to this pair (for the model to judge each).
    pub prior_attempts: Vec<NumberedPriorAttempt>,
}

pub fn capture_original_descriptions(dataset: &ValidationDataset) -> OriginalDescriptions {
    dataset
        .messages
        .iter()
        .map(|m| (m.id, m.vc_message.description.clone()))
        .collect()
}

/// Result of a batch analysis: per-example diagnosis + all proposed amendments.
pub struct BatchAnalysisResult {
    /// example_id → diagnosis text
    pub diagnoses: HashMap<i32, String>,
    pub amendments: Vec<ValidationAmendment>,
    /// example_id → per-amendment failure reasons for UI + ledger.
    pub prior_amendment_failures: HashMap<i32, Vec<FailedAmendmentAttempt>>,
    /// New per-amendment failure records to append to the progress ledger.
    pub new_failure_attempts: Vec<FailedAmendmentAttempt>,
    /// Precomputed recurrence metadata for UI tagging.
    pub repeat_contexts: HashMap<i32, RepeatContext>,
    /// Cross-pair executive summary for this analysis call.
    pub executive_summary: Option<AnalysisExecutiveSummary>,
}

/// Build recurrence info for current failures from earlier confusion rounds + ledger.
pub fn build_repeat_contexts(
    failures: &[ExampleEvalResult],
    past_confusions: &[ConfusionCase],
    past_amendments: &[ValidationAmendment],
    ledger: &ProgressLedger,
) -> HashMap<i32, RepeatContext> {
    let mut out = HashMap::new();
    for failure in failures {
        let prior: Vec<&ConfusionCase> = past_confusions
            .iter()
            .filter(|c| c.example_id == failure.example_id)
            .collect();
        let recurrence_count = (prior.len() as u32) + 1;
        if prior.is_empty() {
            continue;
        }
        let prior_diagnoses = prior
            .iter()
            .filter_map(|c| c.analysis.clone())
            .collect::<Vec<_>>();

        // Start from full ledger history for this example, then add any cumulative
        // amendments that are not yet recorded (e.g. applied last round, not yet judged).
        let mut prior_attempts = Vec::new();
        let mut seen_norms = std::collections::HashSet::new();
        for recorded in ledger.for_example(failure.example_id) {
            let norm = normalize_desc(&recorded.amendment_summary);
            if seen_norms.insert(norm) {
                prior_attempts.push(NumberedPriorAttempt {
                    index: 0, // filled below
                    amendment_summary: recorded.amendment_summary.clone(),
                    known_why: Some(recorded.why_it_failed.clone()),
                });
            }
        }
        for a in past_amendments.iter().filter(|a| amendment_touches_failure(a, failure)) {
            let summary = summarize_amendment(a);
            let norm = normalize_desc(&summary);
            if seen_norms.insert(norm) {
                prior_attempts.push(NumberedPriorAttempt {
                    index: 0,
                    amendment_summary: summary,
                    known_why: None,
                });
            }
        }
        for (i, attempt) in prior_attempts.iter_mut().enumerate() {
            attempt.index = (i + 1) as u32;
        }

        out.insert(
            failure.example_id,
            RepeatContext {
                recurrence_count,
                prior_diagnoses,
                prior_attempts,
            },
        );
    }
    out
}

fn amendment_touches_failure(a: &ValidationAmendment, failure: &ExampleEvalResult) -> bool {
    match a {
        ValidationAmendment::Description { message_id, .. } => {
            *message_id == failure.expected_message_id
                || Some(*message_id) == failure.predicted_message_id
        }
        ValidationAmendment::ReassignValidation { example_id, .. }
        | ValidationAmendment::RemoveValidation { example_id, .. } => {
            *example_id == failure.example_id
        }
    }
}

pub fn summarize_amendment(a: &ValidationAmendment) -> String {
    match a {
        ValidationAmendment::Description {
            message_id,
            category_name,
            old_description,
            new_description,
            rationale,
        } => format!(
            "DESC id={message_id} ({category_name}): \"{old}\" → \"{new}\" ({rationale})",
            old = truncate(old_description, 80),
            new = truncate(new_description, 80),
        ),
        ValidationAmendment::ReassignValidation {
            example_id,
            from_category,
            to_category,
            rationale,
            ..
        } => format!("REASSIGN ex={example_id}: {from_category} → {to_category} ({rationale})"),
        ValidationAmendment::RemoveValidation {
            example_id,
            category_name,
            rationale,
            ..
        } => format!("REMOVE ex={example_id} ({category_name}): {rationale}"),
    }
}

/// Analyze all confusion pairs in one OpenAI call so the model can see patterns
/// across overlapping categories (e.g. Virtual Coordinator vs Brand Inquiries).
pub async fn analyze_confusions_batch(
    api_key: &str,
    model: &str,
    dataset: &ValidationDataset,
    originals: &OriginalDescriptions,
    failures: &[ExampleEvalResult],
    past_confusions: &[ConfusionCase],
    past_amendments: &[ValidationAmendment],
    ledger: &ProgressLedger,
    round: u32,
) -> anyhow::Result<BatchAnalysisResult> {
    if failures.is_empty() {
        return Ok(BatchAnalysisResult {
            diagnoses: HashMap::new(),
            amendments: Vec::new(),
            prior_amendment_failures: HashMap::new(),
            new_failure_attempts: Vec::new(),
            repeat_contexts: HashMap::new(),
            executive_summary: None,
        });
    }

    let repeat_contexts =
        build_repeat_contexts(failures, past_confusions, past_amendments, ledger);
    let catalog = format_message_catalog(dataset, originals);
    let revisions = format_revision_history(past_amendments, originals);
    let past_failures = format_past_confusions(past_confusions);
    let lessons = ledger.format_for_prompt();
    let confusion_blocks =
        format_confusion_pairs(dataset, originals, failures, &repeat_contexts);

    let user_prompt = format!(
        r#"You are a critical reviewer debugging a constrained local LLM classifier for a pharma HCP chatbot.

The local model picks exactly one approved message category from descriptions only (no embeddings).
The validation gold labels are NOT always correct — challenge ones that a reasonable reviewer
would reject. Prefer narrow/specific categories over general catch-alls when the HCP text fits.

## Full message catalog (every category the local LLM sees)
{catalog}

## Revision history so far
{revisions}

## Lessons from amendments that DID NOT fix their target pairs (FULL HISTORY — tracked across rounds)
This ledger is the durable memory for the loop. Every failed amendment attempt is listed with
its specific why_it_failed reason. Future prompts include this entire history — do not forget
or compress it into one vague sentence. Do NOT repeat any of these strategies. When judging
REPEAT OFFENDERs, explain why EACH numbered prior attempt for that example failed (or
confirm/refine the known reason), covering ALL of them — not just the latest.
{lessons}

## Past confusion analyses (for context)
{past_failures}

## Confusion pairs this round (analyze ALL of them together)
Pairs marked REPEAT OFFENDER failed again after earlier amendments. For those you MUST
return prior_amendment_failures covering EVERY numbered prior attempt for that example.
{confusion_blocks}

## Category priority (hard rule)
When more than one category could fit, prefer the **narrower / more specific** category.
Example: "who needs this drug?" → Patient Profiles beats General Brand Inquiries.
General catch-alls are residual buckets for questions that do not fit a specific topic.

## How to judge each pair (in this order)
1. **label_error** — Challenge the validation label when a reasonable reviewer would say the
   expected category is wrong for this HCP text. Especially when the predicted category is
   narrower and clearly fits. Use ReassignValidation (or RemoveValidation if no good target).
   Example: expected=General Brand Inquiries, predicted=Patient Profiles, text="who needs this drug?"
   → label_error + reassign to Patient Profiles. This is NOT a description_gap.
2. **description_gap** — Only when the expected label is reasonable, but descriptions fail to
   separate expected vs predicted for this wording. Propose minimal contrastive Description edits.
3. **llm_mistake** — Rare. Descriptions already cleanly separate the text AND the expected label
   is correct, yet the local model still missed. Prefer description_gap or label_error when unsure.

## prior_amendment_failures (REPEAT OFFENDERs only)
Return one entry per numbered prior attempt listed under that confusion (attempt_index 1..N).
Each why_failed MUST be specific and falsifiable. Cite:
- Exact HCP words/phrases that drove the contested reading
- What THAT amendment actually changed
- Why THAT change could not fix this example

Good: "Attempt 1 only added 'brand vs coordinator' wording; the HCP text 'who needs this drug?'
selects patient-eligibility, so a description tweak on General Brand could not fix a mislabeled
gold example."
Bad (forbidden): "The prior edits did not clearly differentiate between the categories."
Bad (forbidden): "Descriptions were still overlapping / ambiguous."
If a known_why is already shown for an attempt, you may refine it — still return an entry.

## Hard constraints
- Prefer ReassignValidation over Description when the gold label is the problem.
- Prefer narrow over general when reassigning or writing contrastive descriptions.
- Description edits: fewest words; contrastive ("this is X, not Y"); no invented clinical niches.
- Do NOT widen a category so it swallows others.
- Use message_ids / example_ids from above only.

Return ONLY JSON:
{{
  "executive_summary": {{
    "category_priority_assumptions": [
      "Prefer narrow/specific categories over general ones when the HCP text fits the narrow category."
    ],
    "differentiation_decisions": [
      "Short bullets of how you separated confused categories this round"
    ],
    "label_challenges": [
      "Which validation labels you challenged and why (example_id + one-line reason)"
    ]
  }},
  "analyses": [
    {{
      "example_id": 123,
      "diagnosis": "evidence-based explanation citing HCP phrasing",
      "verdict": "label_error" | "description_gap" | "llm_mistake",
      "prior_amendment_failures": [
        {{"attempt_index": 1, "why_failed": "evidence-based reason for attempt 1"}},
        {{"attempt_index": 2, "why_failed": "evidence-based reason for attempt 2"}}
      ],
      "amendments": [
        {{"kind":"description","message_id":1,"new_description":"...","rationale":"..."}},
        {{"kind":"reassign_validation","example_id":123,"to_message_id":2,"rationale":"..."}},
        {{"kind":"remove_validation","example_id":123,"rationale":"..."}}
      ]
    }}
  ]
}}

Include exactly one analyses[] entry for every confusion example_id listed above.
"#,
        catalog = catalog,
        revisions = revisions,
        lessons = lessons,
        past_failures = past_failures,
        confusion_blocks = confusion_blocks,
    );

    let content = openai_chat_json(api_key, model, &user_prompt).await?;
    let parsed: BatchAnalysisResponse = serde_json::from_str(&strip_code_fence(&content))
        .map_err(|e| anyhow::anyhow!("failed to parse batch analysis JSON: {e}; body={content}"))?;

    let failure_by_id: HashMap<i32, &ExampleEvalResult> =
        failures.iter().map(|f| (f.example_id, f)).collect();

    let mut diagnoses = HashMap::new();
    let mut amendments = Vec::new();
    let mut prior_amendment_failures: HashMap<i32, Vec<FailedAmendmentAttempt>> = HashMap::new();
    let mut new_failure_attempts = Vec::new();

    for item in parsed.analyses {
        let diagnosis = match item.verdict.as_deref() {
            Some(v) if !v.is_empty() => format!("[{v}] {}", item.diagnosis),
            _ => item.diagnosis.clone(),
        };
        diagnoses.insert(item.example_id, diagnosis);

        let Some(failure) = failure_by_id.get(&item.example_id).copied() else {
            tracing::warn!(
                example_id = item.example_id,
                "batch analysis returned unknown example_id — skipping amendments"
            );
            continue;
        };

        if let Some(ctx) = repeat_contexts.get(&item.example_id) {
            let mut per_example = Vec::new();
            for attempt in &ctx.prior_attempts {
                let why = item
                    .prior_amendment_failures
                    .iter()
                    .find(|f| f.attempt_index == attempt.index)
                    .map(|f| f.why_failed.trim().to_string())
                    .filter(|s| !s.is_empty() && s != "null")
                    .or_else(|| attempt.known_why.clone())
                    .unwrap_or_else(|| {
                        format!(
                            "Model omitted a specific failure reason for attempt {}.",
                            attempt.index
                        )
                    });
                let record = FailedAmendmentAttempt {
                    round_observed: round,
                    example_id: failure.example_id,
                    expected_category: failure.expected_category.clone(),
                    predicted_category: failure.predicted_category.clone(),
                    amendment_summary: attempt.amendment_summary.clone(),
                    why_it_failed: why,
                };
                per_example.push(record.clone());
                new_failure_attempts.push(record);
            }
            // Ensure every attempt is covered even if the model skipped some indexes.
            if per_example.is_empty() && !ctx.prior_attempts.is_empty() {
                for attempt in &ctx.prior_attempts {
                    let record = FailedAmendmentAttempt {
                        round_observed: round,
                        example_id: failure.example_id,
                        expected_category: failure.expected_category.clone(),
                        predicted_category: failure.predicted_category.clone(),
                        amendment_summary: attempt.amendment_summary.clone(),
                        why_it_failed: attempt.known_why.clone().unwrap_or_else(|| {
                            "Prior amendment failed; model returned no per-attempt reason.".into()
                        }),
                    };
                    per_example.push(record.clone());
                    new_failure_attempts.push(record);
                }
            }
            prior_amendment_failures.insert(item.example_id, per_example);
        }

        for amendment in item.amendments {
            if let Some(mapped) = map_amendment(dataset, failure, amendment) {
                if amendment_repeats_failed_lesson(&mapped, ledger) {
                    tracing::info!(
                        example_id = failure.example_id,
                        "dropping amendment that repeats a failed lesson"
                    );
                    continue;
                }
                amendments.push(mapped);
            }
        }
    }

    for f in failures {
        diagnoses
            .entry(f.example_id)
            .or_insert_with(|| "No diagnosis returned for this pair.".into());
    }

    Ok(BatchAnalysisResult {
        diagnoses,
        amendments,
        prior_amendment_failures,
        new_failure_attempts,
        repeat_contexts,
        executive_summary: parsed.executive_summary.map(|s| AnalysisExecutiveSummary {
            category_priority_assumptions: s.category_priority_assumptions,
            differentiation_decisions: s.differentiation_decisions,
            label_challenges: s.label_challenges,
        }),
    })
}

fn amendment_repeats_failed_lesson(
    amendment: &ValidationAmendment,
    ledger: &ProgressLedger,
) -> bool {
    let ValidationAmendment::Description {
        message_id,
        new_description,
        ..
    } = amendment
    else {
        return false;
    };
    let new_norm = normalize_desc(new_description);
    if new_norm.is_empty() {
        return false;
    }
    ledger.attempts.iter().any(|attempt| {
        attempt.amendment_summary.contains(&format!("id={message_id}"))
            && summary_contains_similar_new(&attempt.amendment_summary, &new_norm)
    })
}

fn summary_contains_similar_new(summary: &str, new_norm: &str) -> bool {
    // Summaries look like: DESC id=N (...): "old" → "new" (rationale)
    let Some(arrow) = summary.find('→') else {
        return false;
    };
    let after = &summary[arrow + '→'.len_utf8()..];
    let start = after.find('"').map(|i| i + 1);
    let end = start.and_then(|s| after[s..].find('"').map(|e| s + e));
    match (start, end) {
        (Some(s), Some(e)) if e > s => {
            let prev_new = normalize_desc(&after[s..e]);
            !prev_new.is_empty()
                && (prev_new == *new_norm
                    || prev_new.contains(new_norm)
                    || new_norm.contains(&prev_new))
        }
        _ => false,
    }
}

fn normalize_desc(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn map_amendment(
    dataset: &ValidationDataset,
    failure: &ExampleEvalResult,
    item: AnalysisAmendment,
) -> Option<ValidationAmendment> {
    match item {
        AnalysisAmendment::Description {
            message_id,
            new_description,
            rationale,
        } => {
            let msg = dataset.messages.iter().find(|m| m.id == message_id)?;
            Some(ValidationAmendment::Description {
                message_id,
                category_name: msg.vc_message.category.clone(),
                old_description: msg.vc_message.description.clone(),
                new_description: new_description.trim().to_string(),
                rationale,
            })
        }
        AnalysisAmendment::ReassignValidation {
            example_id,
            to_message_id,
            rationale,
        } => {
            if example_id != failure.example_id {
                return None;
            }
            let to = dataset.messages.iter().find(|m| m.id == to_message_id)?;
            Some(ValidationAmendment::ReassignValidation {
                example_id: failure.example_id,
                example_text: failure.example_text.clone(),
                from_message_id: failure.expected_message_id,
                from_category: failure.expected_category.clone(),
                to_message_id,
                to_category: to.vc_message.category.clone(),
                rationale,
            })
        }
        AnalysisAmendment::RemoveValidation {
            example_id,
            rationale,
        } => {
            if example_id != failure.example_id {
                return None;
            }
            Some(ValidationAmendment::RemoveValidation {
                example_id: failure.example_id,
                example_text: failure.example_text.clone(),
                message_id: failure.expected_message_id,
                category_name: failure.expected_category.clone(),
                rationale,
            })
        }
    }
}

fn format_confusion_pairs(
    dataset: &ValidationDataset,
    originals: &OriginalDescriptions,
    failures: &[ExampleEvalResult],
    repeat_contexts: &HashMap<i32, RepeatContext>,
) -> String {
    let mut blocks = Vec::with_capacity(failures.len());
    for (i, failure) in failures.iter().enumerate() {
        let expected = dataset
            .messages
            .iter()
            .find(|m| m.id == failure.expected_message_id);
        let predicted = failure
            .predicted_message_id
            .and_then(|id| dataset.messages.iter().find(|m| m.id == id));

        let expected_block = expected
            .map(|m| {
                format_message_detail(
                    "EXPECTED",
                    m.id,
                    &m.vc_message.category,
                    &m.vc_message.description,
                    &m.vc_message.message,
                    originals.get(&m.id),
                )
            })
            .unwrap_or_else(|| "EXPECTED message missing".into());

        let predicted_block = predicted
            .map(|m| {
                format_message_detail(
                    "PREDICTED",
                    m.id,
                    &m.vc_message.category,
                    &m.vc_message.description,
                    &m.vc_message.message,
                    originals.get(&m.id),
                )
            })
            .unwrap_or_else(|| {
                format!(
                    "PREDICTED category={:?} (no matching message)",
                    failure.predicted_category
                )
            });

        let repeat_banner = if let Some(ctx) = repeat_contexts.get(&failure.example_id) {
            let attempts = if ctx.prior_attempts.is_empty() {
                "(none recorded)".into()
            } else {
                ctx.prior_attempts
                    .iter()
                    .map(|a| {
                        let known = a
                            .known_why
                            .as_ref()
                            .map(|w| format!("\n      known_why: {w}"))
                            .unwrap_or_default();
                        format!(
                            "   {idx}. {summary}{known}",
                            idx = a.index,
                            summary = a.amendment_summary,
                            known = known
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            format!(
                "\n*** REPEAT OFFENDER (seen {n} times including this round) ***\nPrior diagnoses: {diags}\nPrior amendments for this pair (explain why EACH failed via prior_amendment_failures):\n{attempts}\nYou MUST return prior_amendment_failures for every attempt_index above and propose a DIFFERENT fix.\n",
                n = ctx.recurrence_count,
                diags = if ctx.prior_diagnoses.is_empty() {
                    "(none)".into()
                } else {
                    ctx.prior_diagnoses.join(" || ")
                },
                attempts = attempts,
            )
        } else {
            String::new()
        };

        blocks.push(format!(
            "### Confusion {n} — example_id={id}{repeat}\nHCP text:\n\"\"\"\n{text}\n\"\"\"\n\n{expected}\n\n{predicted}\n",
            n = i + 1,
            id = failure.example_id,
            repeat = repeat_banner,
            text = failure.example_text,
            expected = expected_block,
            predicted = predicted_block,
        ));
    }
    blocks.join("\n")
}

fn format_message_detail(
    label: &str,
    id: i32,
    category: &str,
    description: &str,
    message: &str,
    original: Option<&String>,
) -> String {
    let original_line = match original {
        Some(orig) if orig != description => format!("\noriginal_description:\n{orig}"),
        _ => String::new(),
    };
    format!(
        "{label} message_id={id} category={category}\ncurrent_description:\n{description}{original_line}\nresponse excerpt:\n{}",
        truncate(message, 500)
    )
}

fn format_message_catalog(dataset: &ValidationDataset, originals: &OriginalDescriptions) -> String {
    let mut lines = Vec::with_capacity(dataset.messages.len());
    for m in &dataset.messages {
        let original = originals.get(&m.id);
        let changed = original.is_some_and(|o| o != &m.vc_message.description);
        let original_note = if changed {
            format!(
                "\n  original: {}",
                original.map(|s| s.as_str()).unwrap_or("")
            )
        } else {
            String::new()
        };
        lines.push(format!(
            "- id={id} category={cat}\n  description: {desc}{original_note}\n  response: {resp}",
            id = m.id,
            cat = m.vc_message.category,
            desc = m.vc_message.description,
            original_note = original_note,
            resp = truncate(&m.vc_message.message, 220),
        ));
    }
    if lines.is_empty() {
        "(empty catalog)".into()
    } else {
        lines.join("\n")
    }
}

fn format_revision_history(
    past_amendments: &[ValidationAmendment],
    originals: &OriginalDescriptions,
) -> String {
    if past_amendments.is_empty() {
        return "None yet — descriptions are still at their originals.".into();
    }
    let mut lines = Vec::with_capacity(past_amendments.len());
    for (i, a) in past_amendments.iter().enumerate() {
        match a {
            ValidationAmendment::Description {
                message_id,
                category_name,
                old_description,
                new_description,
                rationale,
            } => {
                let baseline = originals
                    .get(message_id)
                    .map(|s| s.as_str())
                    .unwrap_or("(unknown)");
                lines.push(format!(
                    "{n}. DESCRIPTION id={message_id} category={category_name}\n   baseline: {baseline}\n   from: {old_description}\n   to:   {new_description}\n   why:  {rationale}",
                    n = i + 1,
                ));
            }
            ValidationAmendment::ReassignValidation {
                example_id,
                from_category,
                to_category,
                rationale,
                example_text,
                ..
            } => {
                lines.push(format!(
                    "{n}. REASSIGN example={example_id} {from_category} → {to_category}\n   text: {text}\n   why: {rationale}",
                    n = i + 1,
                    text = truncate(example_text, 160),
                ));
            }
            ValidationAmendment::RemoveValidation {
                example_id,
                category_name,
                rationale,
                example_text,
                ..
            } => {
                lines.push(format!(
                    "{n}. REMOVE example={example_id} category={category_name}\n   text: {text}\n   why: {rationale}",
                    n = i + 1,
                    text = truncate(example_text, 160),
                ));
            }
        }
    }
    lines.join("\n")
}

fn format_past_confusions(past_confusions: &[ConfusionCase]) -> String {
    if past_confusions.is_empty() {
        return "None yet.".into();
    }
    let start = past_confusions.len().saturating_sub(40);
    past_confusions[start..]
        .iter()
        .map(|c| {
            let repeat = if c.is_repeat_offender {
                " [REPEAT]"
            } else {
                ""
            };
            let fails = if c.prior_amendment_failures.is_empty() {
                String::new()
            } else {
                let detail = c
                    .prior_amendment_failures
                    .iter()
                    .map(|f| format!("{{{summary} => {why}}}", summary = f.amendment_summary, why = f.why_it_failed))
                    .collect::<Vec<_>>()
                    .join("; ");
                format!(" | prior_fails: {detail}")
            };
            format!(
                "- ex {}{} expected={} predicted={:?}: {}{}",
                c.example_id,
                repeat,
                c.expected_category,
                c.predicted_category,
                c.analysis.as_deref().unwrap_or("(no analysis)"),
                fails,
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn truncate(s: &str, max: usize) -> String {
    let mut t: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        t.push('…');
    }
    t
}

fn strip_code_fence(s: &str) -> String {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        let rest = rest.trim_start_matches('\n');
        if let Some(end) = rest.rfind("```") {
            return rest[..end].trim().to_string();
        }
    }
    trimmed.to_string()
}

async fn openai_chat_json(api_key: &str, model: &str, user_prompt: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": model,
        "temperature": 0.1,
        "response_format": { "type": "json_object" },
        "messages": [
            {
                "role": "system",
                "content": "You critically review batches of constrained LLM category confusions. Prefer narrow/specific categories over general catch-alls when the HCP text fits. Challenge incorrect validation gold labels with ReassignValidation — do not paper over bad labels with description tweaks. Use description_gap only when the expected label is reasonable but descriptions need contrast. For REPEAT OFFENDERs, explain why EACH prior amendment failed with concrete HCP-phrase evidence — never a single generic 'did not differentiate'. Include full failed-amendment history in prior_amendment_failures. Include an executive_summary of priority assumptions and differentiation decisions. Never invent clinical niche restrictions absent from the original text. Reply with JSON only."
            },
            { "role": "user", "content": user_prompt }
        ]
    });

    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?
        .error_for_status()?;
    let value: serde_json::Value = resp.json().await?;
    let content = value["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing OpenAI content"))?
        .to_string();
    Ok(content)
}
