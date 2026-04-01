use std::collections::HashMap;
use std::convert::Infallible;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
};
use inference::{CategoryBias, GrammarFlow, InferenceEvent};
use inference_types::{BulkTestEvent, CategoryTopToken, StepCandidates, TokenWithProb};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::db;
use crate::embedding;
use crate::state::AppState;

const EMBEDDING_BATCH_SIZE: usize = 20;

/// Compact per-step record stored in the database.
/// Only retains `chosen` and `category_top_tokens`; the large `top_alternatives`
/// and `top_constrained` candidate lists are intentionally dropped.
#[derive(Serialize, Deserialize)]
struct SlimStep {
    chosen: TokenWithProb,
    category_top_tokens: Vec<CategoryTopToken>,
}

impl SlimStep {
    fn from_step(s: &StepCandidates) -> Self {
        Self {
            chosen: s.chosen.clone(),
            category_top_tokens: s.category_top_tokens.clone(),
        }
    }

    fn into_step(self) -> StepCandidates {
        StepCandidates {
            chosen: self.chosen,
            top_alternatives: vec![],
            top_constrained: vec![],
            category_top_tokens: self.category_top_tokens,
        }
    }
}

#[derive(Deserialize)]
pub struct BulkTestRequest {
    pub agent_id: i32,
}

#[derive(Serialize)]
pub struct BulkTestResponse {
    pub bulk_test_id: String,
    pub run_id: i64,
}

/// POST /bulk-test
///
/// Loads all HCP example messages for the agent, generates embeddings in
/// batches of 20, spawns an async task that runs inference on every example
/// in parallel, and returns a `bulk_test_id` that the client can stream via
/// GET /bulk-test/stream/{bulk_test_id}.
pub async fn start_bulk_test(
    State(state): State<AppState>,
    Json(body): Json<BulkTestRequest>,
) -> Result<Json<BulkTestResponse>, StatusCode> {
    let agent_id = body.agent_id;

    // Load VC messages with their postgres IDs so we can check success.
    let messages_with_ids = db::load_vc_messages_with_ids(&state.vc_db, agent_id)
        .await
        .map_err(|e| {
            tracing::error!(agent_id, error = %e, "failed to load VC messages with IDs");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if messages_with_ids.is_empty() {
        tracing::error!(agent_id, "no valid VC messages found for agent");
        return Err(StatusCode::BAD_REQUEST);
    }

    // Build the GrammarFlow (system prompt + lark grammar) once for this agent.
    let vc_messages: Vec<_> = messages_with_ids
        .iter()
        .map(|m| m.vc_message.clone())
        .collect();
    let grammar_flow = GrammarFlow::new(&state.brand_name, &vc_messages).map_err(|e| {
        tracing::error!(error = %e, "failed to build GrammarFlow for bulk test");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Build a map: category_name → vcmessage id.
    // The grammar output starts with "Category: {name}\n\n", so we parse the
    // category name directly from the generated text prefix instead of doing a
    // full-string match (which breaks on any whitespace/encoding difference).
    let category_to_id: HashMap<String, i32> = messages_with_ids
        .iter()
        .map(|m| (m.vc_message.category.clone(), m.id))
        .collect();

    // Build a map: vcmessage id → category name (for correct_categories lookup).
    let id_to_category: HashMap<i32, String> = messages_with_ids
        .iter()
        .map(|m| (m.id, m.vc_message.category.clone()))
        .collect();

    // Load HCP example messages (the test prompts).
    let examples = db::load_hcp_example_messages(&state.vc_db, agent_id)
        .await
        .map_err(|e| {
            tracing::error!(agent_id, error = %e, "failed to load HCP example messages");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if examples.is_empty() {
        tracing::warn!(agent_id, "no HCP example messages found — nothing to test");
        return Err(StatusCode::BAD_REQUEST);
    }

    // Load the correct-answer map: example_id → Vec<vcmessage_id>.
    let correct_answers = db::load_correct_answer_map(&state.vc_db, agent_id)
        .await
        .map_err(|e| {
            tracing::error!(agent_id, error = %e, "failed to load correct answer map");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Generate embeddings for all examples in batches of EMBEDDING_BATCH_SIZE.
    let example_embeddings: Vec<Vec<f32>> = match std::env::var("OPENAI_API_KEY") {
        Err(_) => {
            tracing::warn!("OPENAI_API_KEY not set — running bulk test without embedding biases");
            vec![vec![]; examples.len()]
        }
        Ok(api_key) => {
            let texts: Vec<&str> = examples.iter().map(|e| e.text.as_str()).collect();
            let mut all_embeddings = Vec::with_capacity(texts.len());
            for chunk in texts.chunks(EMBEDDING_BATCH_SIZE) {
                match embedding::get_openai_embeddings_batch(chunk, &api_key).await {
                    Ok(batch) => all_embeddings.extend(batch),
                    Err(e) => {
                        tracing::warn!(error = %e, "batch embedding failed — using empty vectors for this chunk");
                        all_embeddings.extend(std::iter::repeat(vec![]).take(chunk.len()));
                    }
                }
            }
            all_embeddings
        }
    };

    // Create the SSE channel and register the receiver.
    let (tx, rx) = mpsc::channel::<BulkTestEvent>(examples.len() + 4);
    let bulk_test_id = Uuid::new_v4().to_string();
    state
        .bulk_test_sessions
        .lock()
        .await
        .insert(bulk_test_id.clone(), rx);

    // Persist a run record so results survive past the SSE connection.
    let run_id = db::create_bulk_test_run(&state.db, agent_id)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "failed to create bulk_test_run row");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Spawn the async task that runs all inference jobs.
    let engine = state.engine.clone();
    let vc_db = state.vc_db.clone();
    let sqlite_db = state.db.clone();
    let total = examples.len();

    // Run inference sequentially: one LlamaContext in memory at a time.
    // Each iteration loads the system-prompt KV cache from disk, processes
    // the user turn, generates, then drops the context before the next test.
    tokio::spawn(async move {
        for (example, embedding_vec) in examples.into_iter().zip(example_embeddings.into_iter()) {
            // Shadow to avoid moving into closure before needed.
            // Compute per-category biases using the pre-fetched embedding.
            let category_biases = if embedding_vec.is_empty() {
                vec![]
            } else {
                compute_category_biases_from_embedding(
                    &vc_db,
                    &sqlite_db,
                    &embedding_vec,
                    agent_id,
                )
                .await
            };

            // Run inference — creates one LlamaContext, awaits completion, then drops it.
            let mut infer_rx = engine
                .generate(example.text.clone(), grammar_flow.clone(), category_biases)
                .await;

            let mut steps = Vec::new();
            let mut full_text: Option<String> = None;

            while let Some(event) = infer_rx.recv().await {
                match event {
                    InferenceEvent::Token(step) => steps.push(step),
                    InferenceEvent::Done { full_text: ft } => {
                        full_text = Some(ft);
                        break;
                    }
                    InferenceEvent::Error { message } => {
                        tracing::warn!(
                            example_id = example.id,
                            error = %message,
                            "inference error during bulk test"
                        );
                        break;
                    }
                }
            }

            // Determine the chosen category and whether this is a success.
            //
            // full_output may start with:
            //   " {name}\n\n{message}"  — when process_prompt() forces "Category: "
            //   "Category: {name}\n\n{message}"  — when the prefix is not forced
            // Both cases may have a leading space from SentencePiece.
            // Normalise by trimming whitespace and stripping "Category:" if present,
            // then use longest-match.  No "\n\n" suffix check is needed because
            // longest-match already prevents false-positive prefix matches
            // (e.g. "Safety" vs "Safety Information").
            let (chosen_category, success) = match &full_text {
                None => (None, false),
                Some(ft) => {
                    let ft_norm = {
                        let s = ft.trim_start();
                        let s = s.strip_prefix("Category:").unwrap_or(s);
                        s.trim_start()
                    };
                    tracing::debug!(
                        example_id = example.id,
                        full_text_prefix = %&ft.chars().take(80).collect::<String>(),
                        normalized_prefix = %&ft_norm.chars().take(80).collect::<String>(),
                        "bulk test full_text prefix"
                    );
                    let cat = category_to_id
                        .keys()
                        .filter(|name| ft_norm.starts_with(name.as_str()))
                        .max_by_key(|name| name.len())
                        .cloned();
                    if cat.is_none() {
                        tracing::warn!(
                            example_id = example.id,
                            full_text_prefix = %&ft.chars().take(80).collect::<String>(),
                            normalized_prefix = %&ft_norm.chars().take(80).collect::<String>(),
                            categories = ?category_to_id.keys().collect::<Vec<_>>(),
                            "no category matched full_text prefix"
                        );
                    }
                    let chosen_id = cat
                        .as_deref()
                        .and_then(|c| category_to_id.get(c))
                        .copied();
                    let ok = match chosen_id {
                        None => false,
                        Some(id) => correct_answers
                            .get(&example.id)
                            .map(|ids| ids.contains(&id))
                            .unwrap_or(false),
                    };
                    (cat, ok)
                }
            };

            let correct_categories: Vec<String> = correct_answers
                .get(&example.id)
                .map(|ids| {
                    ids.iter()
                        .filter_map(|id| id_to_category.get(id))
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();

            // Persist to SQLite before streaming so the result is durable even
            // if the client disconnects mid-run.
            let correct_cats_json =
                serde_json::to_string(&correct_categories).unwrap_or_else(|_| "[]".to_string());
            let slim: Vec<SlimStep> = steps.iter().map(SlimStep::from_step).collect();
            let steps_json = serde_json::to_string(&slim).unwrap_or_else(|_| "[]".to_string());
            if let Err(e) = db::insert_bulk_test_result(
                &sqlite_db,
                run_id,
                example.id,
                &example.text,
                chosen_category.as_deref(),
                &correct_cats_json,
                success,
                &steps_json,
            )
            .await
            {
                tracing::warn!(error = %e, "failed to persist bulk_test_result");
            }

            let result = BulkTestEvent::Result {
                example_id: example.id,
                example_text: example.text,
                chosen_category,
                correct_categories,
                success,
                steps,
            };
            if tx.send(result).await.is_err() {
                break; // client disconnected
            }
        }

        // success_count is tallied in the SSE adapter from the Result events.
        let _ = tx
            .send(BulkTestEvent::Done {
                total,
                success_count: 0,
            })
            .await;

        // Finalise the run row with totals (best-effort).
        let success_count = sqlx::query_scalar!(
            "SELECT COUNT(*) FROM bulk_test_results WHERE run_id = ? AND success = 1",
            run_id,
        )
        .fetch_one(&sqlite_db)
        .await
        .unwrap_or(0);
        if let Err(e) =
            db::complete_bulk_test_run(&sqlite_db, run_id, total as i64, success_count).await
        {
            tracing::warn!(error = %e, "failed to complete bulk_test_run row");
        }
    });

    Ok(Json(BulkTestResponse { bulk_test_id, run_id }))
}

/// Compute per-category embedding biases from a pre-fetched embedding vector.
/// Mirrors `compute_category_biases` in routes/infer.rs but accepts a Vec<f32>
/// directly instead of generating one from an API call.
async fn compute_category_biases_from_embedding(
    vc_db: &sqlx::PgPool,
    sqlite_db: &sqlx::SqlitePool,
    embedding: &[f32],
    agent_id: i32,
) -> Vec<CategoryBias> {
    let margins = match db::compute_embedding_margins(vc_db, embedding, agent_id).await {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(error = ?e, "margin query failed during bulk test — skipping biases");
            return vec![];
        }
    };

    let mut biases = Vec::with_capacity(margins.len());
    for m in &margins {
        let kappa = db::get_or_create_kappa(sqlite_db, m.message_id)
            .await
            .unwrap_or(10.0);
        biases.push(CategoryBias {
            category_name: m.category_name.clone(),
            weighted_margin: (kappa * m.margin) as f32,
            sim_score: m.margin as f32,
        });
    }
    biases
}

/// GET /bulk-test/stream/:bulk_test_id
///
/// Streams `BulkTestEvent` values as Server-Sent Events until all test cases
/// have completed and a `Done` event is emitted.
pub async fn stream_bulk_test_sse(
    Path(bulk_test_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    let rx = state
        .bulk_test_sessions
        .lock()
        .await
        .remove(&bulk_test_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    let mut success_count: usize = 0;

    let stream = ReceiverStream::new(rx).map(move |event| {
        // Tally results so we can patch the Done event with the real count.
        match &event {
            BulkTestEvent::Result { success, .. } => {
                if *success {
                    success_count += 1;
                }
            }
            BulkTestEvent::Done { total, .. } => {
                tracing::info!(
                    bulk_test_id = %"<stream>",
                    total,
                    success_count,
                    "bulk test complete"
                );
            }
            BulkTestEvent::Error { message } => {
                tracing::error!(error = %message, "bulk test error event");
            }
        }

        // Patch the Done event with the real accumulated success count.
        let out_event = match event {
            BulkTestEvent::Done { total, .. } => BulkTestEvent::Done {
                total,
                success_count,
            },
            other => other,
        };

        let data = serde_json::to_string(&out_event).unwrap_or_else(|_| "{}".to_string());
        Ok(Event::default().data(data))
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// GET /bulk-tests
///
/// Returns the 50 most recent bulk test runs as a JSON array, newest first.
pub async fn list_bulk_tests(
    State(state): State<AppState>,
) -> Result<Json<Vec<db::BulkTestRunSummary>>, StatusCode> {
    db::list_bulk_test_runs(&state.db).await.map(Json).map_err(|e| {
        tracing::error!(error = %e, "failed to list bulk test runs");
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

/// One result row as returned by GET /bulk-tests/{run_id}.
#[derive(Serialize)]
pub struct StoredTestResult {
    pub example_id: i32,
    pub example_text: String,
    pub chosen_category: Option<String>,
    pub correct_categories: Vec<String>,
    pub success: bool,
    pub steps: Vec<StepCandidates>,
}

/// GET /bulk-tests/{run_id}
///
/// Returns all results for the given run as a JSON array.
pub async fn get_bulk_test(
    Path(run_id): Path<i64>,
    State(state): State<AppState>,
) -> Result<Json<Vec<StoredTestResult>>, StatusCode> {
    let rows = db::load_bulk_test_results(&state.db, run_id).await.map_err(|e| {
        tracing::error!(run_id, error = %e, "failed to load bulk test results");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let results = rows
        .into_iter()
        .filter_map(|r| {
            let correct_categories: Vec<String> =
                serde_json::from_str(&r.correct_categories_json).ok()?;
            let steps: Vec<StepCandidates> = serde_json::from_str::<Vec<SlimStep>>(&r.steps_json)
                .ok()?
                .into_iter()
                .map(SlimStep::into_step)
                .collect();
            Some(StoredTestResult {
                example_id: r.example_id as i32,
                example_text: r.example_text,
                chosen_category: r.chosen_category,
                correct_categories,
                success: r.success,
                steps,
            })
        })
        .collect();

    Ok(Json(results))
}
