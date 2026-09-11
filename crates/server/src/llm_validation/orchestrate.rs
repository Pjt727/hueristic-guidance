use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use inference::InferenceEngine;
use inference_types::{
    ClassifierBackend, LlmValidationEvent, ValidationAmendment, ValidationRoundSummary,
};
use sqlx::PgPool;
use tokio::sync::mpsc;

use super::amend::collate_minimal_amendments;
use super::analyze::{
    analyze_confusions_batch, capture_original_descriptions, ProgressLedger,
};
use super::dataset::{
    apply_amendments_in_memory, load_validation_dataset, persist_amendments, ValidationDataset,
};
use super::evaluate::{confusions_from_report, evaluate_dataset, passing_stats_from_report};

/// Initial evaluation + up to this many improvement retries.
const MAX_RETRIES: u32 = 3;

pub struct LlmValidationSession {
    pub dataset: ValidationDataset,
    pub rounds: Vec<ValidationRoundSummary>,
    pub best_round_index: usize,
    pub proposed_amendments: Vec<ValidationAmendment>,
    pub classifier_backend: ClassifierBackend,
}

pub async fn run_validation_loop(
    engine: Arc<InferenceEngine>,
    vc_db: PgPool,
    brand_name: String,
    agent_id: i32,
    version_id: i32,
    classifier_backend: ClassifierBackend,
    cancel: Arc<AtomicBool>,
    tx: mpsc::Sender<LlmValidationEvent>,
) -> Option<LlmValidationSession> {
    let send = |event: LlmValidationEvent| {
        let tx = tx.clone();
        async move {
            let _ = tx.send(event).await;
        }
    };

    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            send(LlmValidationEvent::Error {
                message: "OPENAI_API_KEY is required for confusion analysis".into(),
            })
            .await;
            return None;
        }
    };
    let analysis_model =
        std::env::var("OPENAI_ANALYSIS_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());

    send(LlmValidationEvent::Status {
        message: format!("Loading agent {agent_id} version {version_id}…"),
    })
    .await;

    let mut dataset = match load_validation_dataset(&vc_db, agent_id, version_id).await {
        Ok(d) => d,
        Err(e) => {
            send(LlmValidationEvent::Error {
                message: e.to_string(),
            })
            .await;
            return None;
        }
    };
    let originals = capture_original_descriptions(&dataset);
    // Working best: accuracy measured on this dataset state (amendments already applied).
    let mut best_dataset = dataset.clone();
    let mut best_amendments: Vec<ValidationAmendment> = Vec::new();
    let mut best_accuracy = -1.0_f64;
    let mut best_round_index = 0usize;

    let backend_label = match classifier_backend {
        ClassifierBackend::Local => "local LLM",
        ClassifierBackend::Openai => "OpenAI classifier (concurrency 5)",
    };
    send(LlmValidationEvent::Status {
        message: format!(
            "Loaded {} messages and {} single-link validation examples — classifying with {backend_label}",
            dataset.messages.len(),
            dataset.examples.len()
        ),
    })
    .await;

    let mut rounds = Vec::new();
    let mut cumulative_confusions = Vec::new();
    let mut cumulative_amendments = Vec::new();
    let mut progress_ledger = ProgressLedger::default();
    let max_rounds = 1 + MAX_RETRIES;

    for round in 0..max_rounds {
        if cancel.load(Ordering::SeqCst) {
            send(LlmValidationEvent::Cancelled {
                message: format!(
                    "Cancelled before round {round} ({} round(s) finished)",
                    rounds.len()
                ),
            })
            .await;
            send(LlmValidationEvent::Done).await;
            return None;
        }

        send(LlmValidationEvent::RoundStarted { round }).await;

        let (example_tx, mut example_rx) = mpsc::channel(8);
        let engine_clone = engine.clone();
        let brand = brand_name.clone();
        let dataset_clone = dataset.clone();
        let cancel_clone = cancel.clone();
        let eval_handle = tokio::spawn(async move {
            evaluate_dataset(
                &engine_clone,
                &brand,
                &dataset_clone,
                Some(example_tx),
                round,
                classifier_backend,
                Some(cancel_clone),
            )
            .await
        });

        while let Some((r, result)) = example_rx.recv().await {
            let _ = tx
                .send(LlmValidationEvent::ExampleResult {
                    round: r,
                    example_id: result.example_id,
                    success: result.success,
                    expected_category: result.expected_category,
                    predicted_category: result.predicted_category,
                    close_top_logits: result.close_top_logits,
                    non_grammar_top_logits: result.non_grammar_top_logits,
                })
                .await;
        }

        let report = match eval_handle.await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                send(LlmValidationEvent::Error {
                    message: e.to_string(),
                })
                .await;
                return None;
            }
            Err(e) => {
                send(LlmValidationEvent::Error {
                    message: format!("eval task failed: {e}"),
                })
                .await;
                return None;
            }
        };

        let mut confusions = confusions_from_report(&report);
        let (passing_categories, passing_close_logit_count) = passing_stats_from_report(&report);
        let accuracy = report.accuracy_pct();

        if report.cancelled {
            let summary = ValidationRoundSummary {
                round,
                total: report.total(),
                success_count: report.success_count,
                accuracy_pct: accuracy,
                confusions: confusions.clone(),
                amendments_applied: vec![],
                passing_categories: passing_categories.clone(),
                passing_close_logit_count,
                executive_summary: None,
            };
            send(LlmValidationEvent::RoundCompleted {
                summary: summary.clone(),
            })
            .await;
            rounds.push(summary);
            send(LlmValidationEvent::Cancelled {
                message: format!(
                    "Cancelled mid-round {round} after {}/{} examples ({:.1}%)",
                    report.success_count,
                    report.total(),
                    accuracy
                ),
            })
            .await;
            send(LlmValidationEvent::Done).await;
            return None;
        }

        send(LlmValidationEvent::Status {
            message: format!(
                "Round {round}: {}/{} ({accuracy:.1}%)",
                report.success_count,
                report.total()
            ),
        })
        .await;

        // Track best *evaluated* state. Amendments applied after this round belong to later rounds.
        if accuracy > best_accuracy {
            best_accuracy = accuracy;
            best_round_index = rounds.len();
            best_dataset = dataset.clone();
            best_amendments = cumulative_amendments.clone();
        } else if round > 0 && accuracy < best_accuracy {
            // Revert working dataset to best-so-far so further edits don't compound regressions.
            send(LlmValidationEvent::Status {
                message: format!(
                    "Round {round} regressed ({accuracy:.1}% < {best_accuracy:.1}%) — reverting to best state before more edits"
                ),
            })
            .await;
            dataset = best_dataset.clone();
            cumulative_amendments = best_amendments.clone();
        }

        let mut amendments_this_round = Vec::new();
        let mut last_exec_summary: Option<inference_types::AnalysisExecutiveSummary> = None;
        let can_retry = accuracy < 100.0 && round < max_rounds - 1 && !report.confusions().is_empty();
        if can_retry {
            send(LlmValidationEvent::Analyzing {
                confusion_count: report.confusions().len(),
            })
            .await;

            // Analyze against the best-known descriptions when we just reverted.
            let failures: Vec<_> = report.confusions().into_iter().cloned().collect();
            if cancel.load(Ordering::SeqCst) {
                // fall through to cancel handling below
            } else {
                match analyze_confusions_batch(
                    &api_key,
                    &analysis_model,
                    &dataset,
                    &originals,
                    &failures,
                    &cumulative_confusions,
                    &cumulative_amendments,
                    &progress_ledger,
                    round,
                )
                .await
                {
                    Ok(batch) => {
                        progress_ledger.extend(batch.new_failure_attempts);
                        for failure in &failures {
                            if let Some(c) = confusions
                                .iter_mut()
                                .find(|c| c.example_id == failure.example_id)
                            {
                                if let Some(diagnosis) = batch.diagnoses.get(&failure.example_id) {
                                    c.analysis = Some(diagnosis.clone());
                                }
                                if let Some(ctx) = batch.repeat_contexts.get(&failure.example_id) {
                                    c.is_repeat_offender = true;
                                    c.recurrence_count = ctx.recurrence_count;
                                }
                                if let Some(fails) =
                                    batch.prior_amendment_failures.get(&failure.example_id)
                                {
                                    c.prior_amendment_failures = fails.clone();
                                } else {
                                    // Still attach full ledger history for this example if any.
                                    c.prior_amendment_failures = progress_ledger
                                        .for_example(failure.example_id)
                                        .into_iter()
                                        .cloned()
                                        .collect();
                                }
                                c.suggested_amendments = batch
                                    .amendments
                                    .iter()
                                    .filter(|a| amendment_relevant_to_failure(a, failure))
                                    .cloned()
                                    .collect();
                            }
                        }
                        amendments_this_round.extend(batch.amendments);
                        last_exec_summary = batch.executive_summary;
                        if !progress_ledger.attempts.is_empty() {
                            send(LlmValidationEvent::Status {
                                message: format!(
                                    "Progress ledger: {} failed-amendment record(s)",
                                    progress_ledger.attempts.len()
                                ),
                            })
                            .await;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "batch confusion analysis failed");
                        send(LlmValidationEvent::Status {
                            message: format!("Batch analysis failed: {e}"),
                        })
                        .await;
                    }
                }
            }

            amendments_this_round = collate_minimal_amendments(&amendments_this_round);
            if !amendments_this_round.is_empty() {
                apply_amendments_in_memory(&mut dataset, &amendments_this_round);
                cumulative_amendments.extend(amendments_this_round.clone());
            }
        }

        if cancel.load(Ordering::SeqCst) {
            let summary = ValidationRoundSummary {
                round,
                total: report.total(),
                success_count: report.success_count,
                accuracy_pct: accuracy,
                confusions: confusions.clone(),
                amendments_applied: amendments_this_round.clone(),
                passing_categories: passing_categories.clone(),
                passing_close_logit_count,
                executive_summary: last_exec_summary.clone(),
            };
            send(LlmValidationEvent::RoundCompleted {
                summary: summary.clone(),
            })
            .await;
            rounds.push(summary);
            send(LlmValidationEvent::Cancelled {
                message: format!("Cancelled during round {round} analysis"),
            })
            .await;
            send(LlmValidationEvent::Done).await;
            return None;
        }

        cumulative_confusions.extend(confusions.clone());

        let summary = ValidationRoundSummary {
            round,
            total: report.total(),
            success_count: report.success_count,
            accuracy_pct: accuracy,
            confusions,
            amendments_applied: amendments_this_round.clone(),
            passing_categories,
            passing_close_logit_count,
            executive_summary: last_exec_summary,
        };
        send(LlmValidationEvent::RoundCompleted {
            summary: summary.clone(),
        })
        .await;
        rounds.push(summary);

        if accuracy >= 100.0 {
            send(LlmValidationEvent::Status {
                message: "Reached 100% — stopping".into(),
            })
            .await;
            break;
        }

        if can_retry && amendments_this_round.is_empty() {
            send(LlmValidationEvent::Status {
                message: "No actionable amendments — stopping retries".into(),
            })
            .await;
            break;
        }
    }

    // Amendments that produced the best evaluated accuracy (none if best is baseline).
    let proposed_amendments = best_amendments;

    let best_accuracy_pct = rounds
        .get(best_round_index)
        .map(|r| r.accuracy_pct)
        .unwrap_or(0.0);
    let best_round = rounds
        .get(best_round_index)
        .map(|r| r.round)
        .unwrap_or(0);

    let proposal_exec = rounds
        .get(best_round_index)
        .and_then(|r| r.executive_summary.clone())
        .or_else(|| rounds.iter().rev().find_map(|r| r.executive_summary.clone()));

    send(LlmValidationEvent::ProposalReady {
        best_round,
        best_accuracy_pct,
        amendments: proposed_amendments.clone(),
        rounds: rounds.clone(),
        executive_summary: proposal_exec,
    })
    .await;

    Some(LlmValidationSession {
        dataset: best_dataset,
        rounds,
        best_round_index,
        proposed_amendments,
        classifier_backend,
    })
}

pub async fn apply_amendments(
    vc_db: &PgPool,
    amendments: &[ValidationAmendment],
    tx: mpsc::Sender<LlmValidationEvent>,
) {
    match persist_amendments(vc_db, amendments).await {
        Ok(n) => {
            let _ = tx
                .send(LlmValidationEvent::Applied { applied_count: n })
                .await;
            let _ = tx.send(LlmValidationEvent::Done).await;
        }
        Err(e) => {
            let _ = tx
                .send(LlmValidationEvent::Error {
                    message: e.to_string(),
                })
                .await;
        }
    }
}

fn amendment_relevant_to_failure(
    amendment: &ValidationAmendment,
    failure: &super::evaluate::ExampleEvalResult,
) -> bool {
    match amendment {
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

/// Deny path: hard-limit one baseline retest, no further amendment loop.
pub async fn deny_and_retest(
    engine: Arc<InferenceEngine>,
    vc_db: PgPool,
    brand_name: String,
    agent_id: i32,
    version_id: i32,
    classifier_backend: ClassifierBackend,
    cancel: Arc<AtomicBool>,
    tx: mpsc::Sender<LlmValidationEvent>,
) {
    let send = |event: LlmValidationEvent| {
        let tx = tx.clone();
        async move {
            let _ = tx.send(event).await;
        }
    };

    send(LlmValidationEvent::Status {
        message: "Amendments denied — retesting baseline once…".into(),
    })
    .await;

    let dataset = match load_validation_dataset(&vc_db, agent_id, version_id).await {
        Ok(d) => d,
        Err(e) => {
            send(LlmValidationEvent::Error {
                message: e.to_string(),
            })
            .await;
            return;
        }
    };

    match evaluate_dataset(
        &engine,
        &brand_name,
        &dataset,
        None,
        0,
        classifier_backend,
        Some(cancel.clone()),
    )
    .await
    {
        Ok(report) => {
            if report.cancelled || cancel.load(Ordering::SeqCst) {
                send(LlmValidationEvent::Cancelled {
                    message: "Denied retest cancelled".into(),
                })
                .await;
                send(LlmValidationEvent::Done).await;
                return;
            }
            send(LlmValidationEvent::DeniedRetest {
                total: report.total(),
                success_count: report.success_count,
                accuracy_pct: report.accuracy_pct(),
            })
            .await;
            send(LlmValidationEvent::Done).await;
        }
        Err(e) => {
            send(LlmValidationEvent::Error {
                message: e.to_string(),
            })
            .await;
        }
    }
}
