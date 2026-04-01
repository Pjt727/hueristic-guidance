use std::collections::HashMap;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use inference_types::CategoryTopToken;
use serde::{Deserialize, Serialize};

use crate::db;
use crate::optimize::{CategoryScore, ExampleData};
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
    let agent_id = db::get_run_agent_id(&state.db, run_id)
        .await
        .map_err(|e| {
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

    Ok(Json(ApplyWeightsResponse { updated, unmatched_categories }))
}

// Re-use the SlimStep definition from bulk_test (private there), so we
// duplicate the minimal serde-only version we need here.
#[derive(Deserialize)]
struct SlimStep {
    category_top_tokens: Vec<CategoryTopToken>,
}

/// Response body for POST /bulk-tests/{run_id}/optimize
#[derive(Serialize)]
pub struct OptimizeResponse {
    /// Per-category optimal weight multipliers.
    /// A value of 1.0 means "same as current kappa"; >1.0 means increase the
    /// embedding influence for this category; <1.0 means decrease it.
    pub weights: HashMap<String, f64>,
    /// Number of examples used in the optimisation.
    pub examples_used: usize,
    /// Number of examples skipped (no embedding data or no correct categories).
    pub examples_skipped: usize,
}

/// POST /bulk-tests/{run_id}/optimize
///
/// Reads all results for the given bulk test run, builds per-example
/// per-category (logit, sim_score) matrices, and solves for the
/// minimum-norm ridge-regression weights that maximise classification accuracy.
///
/// Returns 422 if the run contains no usable embedding data (e.g. an old run
/// recorded before `sim_score` was added to CategoryTopToken).
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

        let category_scores: HashMap<String, CategoryScore> = scored_step
            .category_top_tokens
            .into_iter()
            .map(|ct| {
                (
                    ct.category_name,
                    CategoryScore {
                        logit: ct.best_token.logit,
                        sim_score: ct.sim_score,
                    },
                )
            })
            .collect();

        examples_used += 1;
        example_data.push(ExampleData {
            category_scores,
            correct_categories,
        });
    }

    match crate::optimize::optimize_weights(&example_data) {
        Some(weights) => Ok(Json(OptimizeResponse {
            weights,
            examples_used,
            examples_skipped,
        })),
        None => {
            tracing::warn!(
                run_id,
                examples_used,
                "optimise returned None — likely no embedding data in this run"
            );
            Err(StatusCode::UNPROCESSABLE_ENTITY)
        }
    }
}
