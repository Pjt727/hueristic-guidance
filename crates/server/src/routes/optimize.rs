use std::collections::HashMap;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use classifiers::calibration;
use inference_types::{
    BiasRegime, CategoryTopToken, ClassifierMethodResult, DecisionDiagnostics, OverrideOutcome,
};
use serde::{Deserialize, Serialize};

use crate::db;
use crate::optimize::{self, CategoryScore, ExampleData};
use crate::state::AppState;

/// Request body for POST /bulk-tests/{run_id}/apply-weights
#[derive(Deserialize)]
pub struct ApplyWeightsRequest {
    /// Per-category kappa values as returned by the optimize endpoint.
    pub weights: HashMap<String, f64>,
}

/// Response body for POST /bulk-tests/{run_id}/apply-weights
#[derive(Serialize)]
pub struct ApplyWeightsResponse {
    /// Number of messages whose kappa was updated.
    pub updated: usize,
    /// Category names that appeared in `weights` but had no matching VC messages.
    pub unmatched_categories: Vec<String>,
}

/// POST /bulk-tests/{run_id}/apply-weights
///
/// Saves the supplied per-category kappa values to SQLite so they are used by
/// future inference runs.  For each category, every VC message belonging to
/// that category (for the agent that owns this run) has its kappa updated.
/// The values are the new absolute kappa — the old kappa is NOT used.
pub async fn apply_weights(
    Path(run_id): Path<i64>,
    State(state): State<AppState>,
    Json(body): Json<ApplyWeightsRequest>,
) -> Result<Json<ApplyWeightsResponse>, StatusCode> {
    // Look up which agent this run belongs to.
    let agent_id = db::get_run_agent_id(&state.db, run_id).await.map_err(|e| {
        tracing::error!(run_id, error = %e, "failed to get agent_id for run");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Load all VC messages for this agent to build category → message_ids.
    let messages = db::load_vc_messages_with_ids(&state.vc_db, agent_id as i32)
        .await
        .map_err(|e| {
            tracing::error!(agent_id, error = %e, "failed to load VC messages for apply-weights");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Build category_name → Vec<message_id>.
    let mut category_to_ids: HashMap<String, Vec<i64>> = HashMap::new();
    for msg in &messages {
        category_to_ids
            .entry(msg.vc_message.category.clone())
            .or_default()
            .push(msg.id as i64);
    }

    let mut updated = 0usize;
    let mut unmatched_categories: Vec<String> = Vec::new();

    for (category_name, kappa) in &body.weights {
        match category_to_ids.get(category_name.as_str()) {
            None => unmatched_categories.push(category_name.clone()),
            Some(ids) => {
                for &message_id in ids {
                    if let Err(e) = db::set_kappa(&state.db, message_id, *kappa).await {
                        tracing::warn!(
                            message_id, category = %category_name, error = %e,
                            "failed to set kappa"
                        );
                    } else {
                        updated += 1;
                    }
                }
            }
        }
    }

    tracing::info!(
        run_id,
        agent_id,
        updated,
        unmatched = unmatched_categories.len(),
        "applied optimised kappa values"
    );

    Ok(Json(ApplyWeightsResponse {
        updated,
        unmatched_categories,
    }))
}

// Re-use the SlimStep definition from bulk_test (private there), so we
// duplicate the minimal serde-only version we need here.
#[derive(Deserialize)]
struct SlimStep {
    #[allow(dead_code)]
    chosen: inference_types::TokenWithProb,
    category_top_tokens: Vec<CategoryTopToken>,
    #[serde(default)]
    decision_diagnostics: Option<DecisionDiagnostics>,
}

#[derive(Debug, Default, Serialize)]
pub struct RoutingMetrics {
    pub examples_analyzed: usize,
    pub force_gate_count: usize,
    pub force_gate_rate_pct: f64,
    pub soft_override_count: usize,
    pub forced_override_count: usize,
    pub override_correct: usize,
    pub override_accuracy_pct: f64,
    pub forced_correct: usize,
    pub forced_accuracy_pct: f64,
    pub raw_correct: usize,
    pub raw_accuracy_pct: f64,
    pub final_correct: usize,
    pub final_accuracy_pct: f64,
}

/// Response body for POST /bulk-tests/{run_id}/optimize
#[derive(Serialize)]
pub struct OptimizeResponse {
    /// Per-category optimal kappa values.
    pub weights: HashMap<String, f64>,
    /// Number of examples used in the optimisation.
    pub examples_used: usize,
    /// Number of examples skipped (no embedding data or no correct categories).
    pub examples_skipped: usize,
    /// Accuracy with the default kappa=10.0 for all categories.
    pub baseline_accuracy: optimize::AccuracyReport,
    /// Accuracy with kappa=0 (LLM logits only, no embedding bias).
    pub no_embedding_accuracy: optimize::AccuracyReport,
    /// Accuracy with the optimised kappa values.
    pub optimized_accuracy: optimize::AccuracyReport,
    /// Classifier ensemble optimization results (TF-IDF vs embedding weights).
    pub ensemble_optimization: Option<calibration::OptimizationResult>,
    /// Effectiveness of embedding routing versus the raw LLM winner.
    pub routing_metrics: RoutingMetrics,
}

/// POST /bulk-tests/{run_id}/optimize
///
/// Reads all results for the given bulk test run, builds per-example
/// per-category (logit, sim_score) matrices, and solves for the
/// minimum-norm ridge-regression weights that maximise classification accuracy.
///
/// Also runs classifier ensemble weight optimization and LLM logit analysis
/// to show alternative approaches for combining embedding and logit data.
///
/// Returns 422 if the run contains no usable embedding data.
pub async fn optimize_weights(
    Path(run_id): Path<i64>,
    State(state): State<AppState>,
) -> Result<Json<OptimizeResponse>, StatusCode> {
    let rows = db::load_bulk_test_results(&state.db, run_id)
        .await
        .map_err(|e| {
            tracing::error!(run_id, error = %e, "failed to load bulk_test_results for optimise");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if rows.is_empty() {
        return Err(StatusCode::NOT_FOUND);
    }

    let mut examples_used = 0usize;
    let mut examples_skipped = 0usize;
    let mut example_data: Vec<ExampleData> = Vec::with_capacity(rows.len());
    let mut calibration_examples: Vec<calibration::ExampleScores> = Vec::with_capacity(rows.len());
    let mut routing_metrics = RoutingMetrics::default();

    for row in &rows {
        // Parse correct categories.
        let correct_categories: Vec<String> =
            match serde_json::from_str(&row.correct_categories_json) {
                Ok(v) => v,
                Err(_) => {
                    examples_skipped += 1;
                    continue;
                }
            };

        if correct_categories.is_empty() {
            examples_skipped += 1;
            continue;
        }

        // Parse steps JSON → find first step with category_top_tokens populated.
        let steps: Vec<SlimStep> = match serde_json::from_str(&row.steps_json) {
            Ok(v) => v,
            Err(_) => {
                examples_skipped += 1;
                continue;
            }
        };

        let first_scored_step = steps
            .into_iter()
            .find(|s| !s.category_top_tokens.is_empty());

        let scored_step = match first_scored_step {
            Some(s) => s,
            None => {
                examples_skipped += 1;
                continue;
            }
        };
        let has_decision_diagnostics = scored_step.decision_diagnostics.is_some();

        if let Some(diagnostics) = &scored_step.decision_diagnostics {
            routing_metrics.examples_analyzed += 1;
            if diagnostics.bias_regime == BiasRegime::Force {
                routing_metrics.force_gate_count += 1;
                if row.success {
                    routing_metrics.forced_correct += 1;
                }
            }
            match diagnostics.override_outcome {
                OverrideOutcome::SoftBias => routing_metrics.soft_override_count += 1,
                OverrideOutcome::ForcedEmbedding => routing_metrics.forced_override_count += 1,
                OverrideOutcome::None => {}
            }
            if diagnostics.override_outcome != OverrideOutcome::None && row.success {
                routing_metrics.override_correct += 1;
            }
            if diagnostics
                .pre_bias
                .winner_category
                .as_ref()
                .is_some_and(|winner| correct_categories.contains(winner))
            {
                routing_metrics.raw_correct += 1;
            }
            if row.success {
                routing_metrics.final_correct += 1;
            }
        }

        let category_scores: HashMap<String, CategoryScore> = scored_step
            .category_top_tokens
            .iter()
            .map(|ct| {
                (
                    ct.category_name.clone(),
                    CategoryScore {
                        logit: if has_decision_diagnostics {
                            ct.pre_bias_logit
                        } else {
                            ct.best_token.logit
                        },
                        sim_score: ct.sim_score,
                        positive_similarity: ct.positive_similarity,
                        force_target: scored_step
                            .decision_diagnostics
                            .as_ref()
                            .and_then(|diagnostics| diagnostics.force_target.as_ref())
                            == Some(&ct.category_name),
                    },
                )
            })
            .collect();

        // Build an unbiased LLM logit map for calibration.
        let llm_logits: HashMap<String, f32> = scored_step
            .category_top_tokens
            .iter()
            .map(|ct| {
                (
                    ct.category_name.clone(),
                    if has_decision_diagnostics {
                        ct.pre_bias_logit
                    } else {
                        ct.best_token.logit
                    },
                )
            })
            .collect();

        let category_sim_scores: HashMap<String, f32> = scored_step
            .category_top_tokens
            .iter()
            .map(|ct| (ct.category_name.clone(), ct.sim_score))
            .collect();

        // Parse classifier results for the calibration module.
        let classifier_results: Vec<ClassifierMethodResult> =
            serde_json::from_str(&row.classifier_results_json).unwrap_or_default();

        let mut tfidf_scores: HashMap<String, f64> = HashMap::new();
        let mut embedding_scores: HashMap<String, f64> = HashMap::new();

        for cr in &classifier_results {
            let target = match cr.method_name.as_str() {
                "tfidf" => &mut tfidf_scores,
                "openai_embedding" => &mut embedding_scores,
                _ => continue,
            };
            for s in &cr.scores {
                target.insert(s.category_name.clone(), s.score);
            }
        }

        calibration_examples.push(calibration::ExampleScores {
            tfidf_scores,
            embedding_scores,
            correct_categories: correct_categories.clone(),
            llm_logits: Some(llm_logits),
            category_sim_scores: Some(category_sim_scores),
        });

        examples_used += 1;
        example_data.push(ExampleData {
            category_scores,
            correct_categories,
        });
    }

    // Run kappa optimization via Adam gradient descent.
    let weights = match optimize::optimize_weights(&example_data) {
        Some(w) => w,
        None => {
            tracing::warn!(
                run_id,
                examples_used,
                "optimise returned None — likely no embedding data in this run"
            );
            return Err(StatusCode::UNPROCESSABLE_ENTITY);
        }
    };

    // Compute accuracy at default kappa=10.0.
    let all_categories: Vec<String> = example_data
        .iter()
        .flat_map(|ex| ex.category_scores.keys().cloned())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let default_kappa: HashMap<String, f64> =
        all_categories.iter().map(|c| (c.clone(), 10.0)).collect();
    let zero_kappa: HashMap<String, f64> =
        all_categories.iter().map(|c| (c.clone(), 0.0)).collect();

    let baseline_accuracy = optimize::eval_accuracy(&example_data, &default_kappa);
    let no_embedding_accuracy = optimize::eval_accuracy(&example_data, &zero_kappa);
    let optimized_accuracy = optimize::eval_accuracy(&example_data, &weights);

    // Run classifier ensemble optimization (TF-IDF vs embedding weights + LLM analysis).
    let ensemble_optimization = if calibration_examples
        .iter()
        .any(|ex| !ex.embedding_scores.is_empty())
    {
        Some(calibration::optimize(&calibration_examples))
    } else {
        None
    };
    finalize_routing_metrics(&mut routing_metrics);

    Ok(Json(OptimizeResponse {
        weights,
        examples_used,
        examples_skipped,
        baseline_accuracy,
        no_embedding_accuracy,
        optimized_accuracy,
        ensemble_optimization,
        routing_metrics,
    }))
}

fn percent(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64 * 100.0
    }
}

fn finalize_routing_metrics(metrics: &mut RoutingMetrics) {
    metrics.force_gate_rate_pct = percent(metrics.force_gate_count, metrics.examples_analyzed);
    let override_count = metrics.soft_override_count + metrics.forced_override_count;
    metrics.override_accuracy_pct = percent(metrics.override_correct, override_count);
    metrics.forced_accuracy_pct = percent(metrics.forced_correct, metrics.force_gate_count);
    metrics.raw_accuracy_pct = percent(metrics.raw_correct, metrics.examples_analyzed);
    metrics.final_accuracy_pct = percent(metrics.final_correct, metrics.examples_analyzed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_percentages_use_their_correct_denominators() {
        let mut metrics = RoutingMetrics {
            examples_analyzed: 10,
            force_gate_count: 4,
            soft_override_count: 2,
            forced_override_count: 2,
            override_correct: 3,
            forced_correct: 3,
            raw_correct: 6,
            final_correct: 8,
            ..RoutingMetrics::default()
        };
        finalize_routing_metrics(&mut metrics);
        assert_eq!(metrics.force_gate_rate_pct, 40.0);
        assert_eq!(metrics.override_accuracy_pct, 75.0);
        assert_eq!(metrics.forced_accuracy_pct, 75.0);
        assert_eq!(metrics.raw_accuracy_pct, 60.0);
        assert_eq!(metrics.final_accuracy_pct, 80.0);
    }

    #[test]
    fn empty_routing_metrics_do_not_produce_nan() {
        let mut metrics = RoutingMetrics::default();
        finalize_routing_metrics(&mut metrics);
        assert_eq!(metrics.override_accuracy_pct, 0.0);
        assert_eq!(metrics.force_gate_rate_pct, 0.0);
    }
}
