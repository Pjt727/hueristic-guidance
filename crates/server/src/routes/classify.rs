use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use classifiers::{
    CategoryDef, ClassificationResult, Classifier, ConversationContext,
    ensemble::EnsembleClassifier,
    openai_embedding::OpenAiEmbeddingClassifier,
    tfidf::TfIdfClassifier,
};
use serde::{Deserialize, Serialize};

use crate::db;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ClassifyRequest {
    pub prompt: String,
    pub agent_id: i32,
    #[serde(default)]
    pub previous_messages: Vec<String>,
}

#[derive(Serialize)]
pub struct ClassifyResponse {
    pub results: Vec<ClassificationResult>,
}

#[derive(Deserialize)]
pub struct CompareRequest {
    pub agent_id: i32,
}

#[derive(Serialize)]
pub struct CompareResponse {
    pub run_id: i64,
}

#[derive(Serialize)]
pub struct CompareRunSummary {
    pub id: i64,
    pub agent_id: i64,
    pub methods: String,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub total: Option<i64>,
    pub summary: Option<String>,
}

#[derive(Serialize)]
pub struct CompareResultRow {
    pub example_id: i64,
    pub example_text: String,
    pub correct_categories: Vec<String>,
    pub results: HashMap<String, ClassificationResult>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build classifier instances for the given agent's categories.
/// Returns a list of (classifier, categories) ready for classification.
async fn build_classifiers_for_agent(
    state: &AppState,
    agent_id: i32,
) -> Result<(Vec<Arc<dyn Classifier>>, Vec<CategoryDef>), StatusCode> {
    let vc_messages = db::load_vc_messages(&state.vc_db, agent_id)
        .await
        .map_err(|e| {
            tracing::error!(agent_id, error = %e, "failed to load VC messages for classify");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let categories: Vec<CategoryDef> = vc_messages
        .iter()
        .map(|m| CategoryDef {
            name: m.category.clone(),
            description: m.description.clone(),
        })
        .collect();

    if categories.is_empty() {
        tracing::error!(agent_id, "no categories found for agent");
        return Err(StatusCode::BAD_REQUEST);
    }

    let mut classifiers: Vec<Arc<dyn Classifier>> = vec![];

    // TF-IDF — always available
    let tfidf = Arc::new(TfIdfClassifier::new(&categories));
    classifiers.push(tfidf);

    // OpenAI embedding — if API key is set
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        match OpenAiEmbeddingClassifier::new(api_key, &categories).await {
            Ok(c) => classifiers.push(Arc::new(c)),
            Err(e) => {
                tracing::warn!(error = %e, "failed to initialize OpenAI embedding classifier");
            }
        }
    }

    // Ensemble — wraps whatever classifiers are available (if > 1)
    if classifiers.len() > 1 {
        let ensemble = EnsembleClassifier::equal_weight(classifiers.clone());
        classifiers.push(Arc::new(ensemble));
    }

    Ok((classifiers, categories))
}

// ---------------------------------------------------------------------------
// GET /classify/methods — list available classifier method names
// ---------------------------------------------------------------------------

/// Returns the names of classifier methods available on this server instance.
pub async fn list_methods() -> Json<Vec<&'static str>> {
    let mut methods = vec!["tfidf"];
    if std::env::var("OPENAI_API_KEY").is_ok() {
        methods.push("openai_embedding");
        methods.push("ensemble");
    }
    Json(methods)
}

// ---------------------------------------------------------------------------
// POST /classify — single query, all methods
// ---------------------------------------------------------------------------

/// Classify a single query using all available methods.
/// Returns results from each classifier with scores, confidence, and latency.
pub async fn classify_single(
    State(state): State<AppState>,
    Json(body): Json<ClassifyRequest>,
) -> Result<Json<ClassifyResponse>, StatusCode> {
    let (classifiers, categories) = build_classifiers_for_agent(&state, body.agent_id).await?;

    let context = if body.previous_messages.is_empty() {
        None
    } else {
        Some(ConversationContext {
            previous_messages: body.previous_messages,
        })
    };

    let mut results = Vec::new();
    for classifier in &classifiers {
        match classifier
            .classify(&body.prompt, &categories, context.as_ref())
            .await
        {
            Ok(result) => results.push(result),
            Err(e) => {
                tracing::warn!(
                    method = classifier.name(),
                    error = %e,
                    "classifier failed"
                );
            }
        }
    }

    Ok(Json(ClassifyResponse { results }))
}

// ---------------------------------------------------------------------------
// POST /classify/compare — bulk comparison
// ---------------------------------------------------------------------------

/// Run all classifiers on all HCP examples for the given agent.
/// Stores results in SQLite and returns a run_id.
pub async fn start_compare(
    State(state): State<AppState>,
    Json(body): Json<CompareRequest>,
) -> Result<Json<CompareResponse>, StatusCode> {
    let agent_id = body.agent_id;

    // Build classifiers
    let (classifiers, categories) = build_classifiers_for_agent(&state, agent_id).await?;

    let method_names: Vec<String> = classifiers.iter().map(|c| c.name().to_string()).collect();
    let methods_json = serde_json::to_string(&method_names).unwrap_or_else(|_| "[]".to_string());

    // Load HCP examples
    let examples = db::load_hcp_example_messages(&state.vc_db, agent_id)
        .await
        .map_err(|e| {
            tracing::error!(agent_id, error = %e, "failed to load HCP examples for compare");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if examples.is_empty() {
        tracing::warn!(agent_id, "no HCP examples found for compare");
        return Err(StatusCode::BAD_REQUEST);
    }

    // Load correct answer map
    let correct_answers = db::load_correct_answer_map(&state.vc_db, agent_id)
        .await
        .map_err(|e| {
            tracing::error!(agent_id, error = %e, "failed to load correct answers for compare");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Filter out examples with no expected category.
    let examples: Vec<_> = examples
        .into_iter()
        .filter(|e| {
            correct_answers
                .get(&e.id)
                .map(|ids| !ids.is_empty())
                .unwrap_or(false)
        })
        .collect();

    if examples.is_empty() {
        tracing::warn!(agent_id, "no HCP examples with expected categories for compare");
        return Err(StatusCode::BAD_REQUEST);
    }

    // Load VC messages with IDs to map correct answer IDs to category names
    let messages_with_ids = db::load_vc_messages_with_ids(&state.vc_db, agent_id)
        .await
        .map_err(|e| {
            tracing::error!(agent_id, error = %e, "failed to load VC messages with IDs");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    let id_to_category: HashMap<i32, String> = messages_with_ids
        .iter()
        .map(|m| (m.id, m.vc_message.category.clone()))
        .collect();

    // Create comparison run
    let run_id = db::create_comparison_run(&state.db, agent_id as i64, &methods_json)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "failed to create comparison run");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Spawn background task
    let sqlite_db = state.db.clone();
    let total = examples.len();

    tokio::spawn(async move {
        let mut method_correct: HashMap<String, usize> = HashMap::new();

        for example in &examples {
            let correct_categories: Vec<String> = correct_answers
                .get(&example.id)
                .map(|ids| {
                    ids.iter()
                        .filter_map(|id| id_to_category.get(id))
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();

            let mut per_method: HashMap<String, ClassificationResult> = HashMap::new();

            for classifier in &classifiers {
                match classifier
                    .classify(&example.text, &categories, None)
                    .await
                {
                    Ok(result) => {
                        if correct_categories.contains(&result.chosen_category) {
                            *method_correct.entry(result.method_name.clone()).or_default() += 1;
                        }
                        per_method.insert(result.method_name.clone(), result);
                    }
                    Err(e) => {
                        tracing::warn!(
                            method = classifier.name(),
                            example_id = example.id,
                            error = %e,
                            "classifier failed during compare"
                        );
                    }
                }
            }

            let correct_cats_json =
                serde_json::to_string(&correct_categories).unwrap_or_else(|_| "[]".to_string());
            let results_json =
                serde_json::to_string(&per_method).unwrap_or_else(|_| "{}".to_string());

            if let Err(e) = db::insert_comparison_result(
                &sqlite_db,
                run_id,
                example.id as i64,
                &example.text,
                &correct_cats_json,
                &results_json,
            )
            .await
            {
                tracing::warn!(error = %e, "failed to persist comparison result");
            }
        }

        // Build summary
        let summary: HashMap<String, serde_json::Value> = method_correct
            .into_iter()
            .map(|(method, correct)| {
                let acc = if total == 0 {
                    0.0
                } else {
                    correct as f64 / total as f64 * 100.0
                };
                (
                    method,
                    serde_json::json!({
                        "correct": correct,
                        "total": total,
                        "accuracy_pct": acc,
                    }),
                )
            })
            .collect();
        let summary_json = serde_json::to_string(&summary).unwrap_or_else(|_| "{}".to_string());

        if let Err(e) =
            db::complete_comparison_run(&sqlite_db, run_id, total as i64, &summary_json).await
        {
            tracing::warn!(error = %e, "failed to complete comparison run");
        }

        tracing::info!(run_id, total, "comparison run complete");
    });

    Ok(Json(CompareResponse { run_id }))
}

// ---------------------------------------------------------------------------
// GET /classify/compare/{run_id} — fetch comparison results
// ---------------------------------------------------------------------------

pub async fn get_compare_results(
    Path(run_id): Path<i64>,
    State(state): State<AppState>,
) -> Result<Json<Vec<CompareResultRow>>, StatusCode> {
    let rows = db::load_comparison_results(&state.db, run_id)
        .await
        .map_err(|e| {
            tracing::error!(run_id, error = %e, "failed to load comparison results");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let results = rows
        .into_iter()
        .filter_map(|r| {
            let correct_categories: Vec<String> =
                serde_json::from_str(&r.correct_categories_json).ok()?;
            let results: HashMap<String, ClassificationResult> =
                serde_json::from_str(&r.results_json).ok()?;
            Some(CompareResultRow {
                example_id: r.example_id,
                example_text: r.example_text,
                correct_categories,
                results,
            })
        })
        .collect();

    Ok(Json(results))
}

// ---------------------------------------------------------------------------
// GET /classify/compare-runs — list recent comparison runs
// ---------------------------------------------------------------------------

pub async fn list_compare_runs(
    State(state): State<AppState>,
) -> Result<Json<Vec<CompareRunSummary>>, StatusCode> {
    db::list_comparison_runs(&state.db).await.map(Json).map_err(|e| {
        tracing::error!(error = %e, "failed to list comparison runs");
        StatusCode::INTERNAL_SERVER_ERROR
    })
}
