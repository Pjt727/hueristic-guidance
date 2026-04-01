use std::convert::Infallible;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
};
use inference::{CategoryBias, GrammarFlow, InferenceEvent};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

use crate::db;
use crate::embedding;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct InferRequest {
    pub prompt: String,
    pub agent_id: i32,
}

#[derive(Serialize)]
pub struct InferResponse {
    pub session_id: String,
}

/// POST /infer
///
/// Loads VC messages for the requested agent, builds a `GrammarFlow` (which
/// renders the Askama system-prompt and lark-grammar templates), computes
/// per-category embedding logit biases from the user's message similarity,
/// creates an inference session in SQLite, kicks off generation, and returns
/// the public `session_id`.
pub async fn start_infer(
    State(state): State<AppState>,
    Json(body): Json<InferRequest>,
) -> Result<Json<InferResponse>, StatusCode> {
    // Load the latest VC messages for the chosen agent from marketing Postgres
    let vc_messages = db::load_vc_messages(&state.vc_db, body.agent_id)
        .await
        .map_err(|e| {
            tracing::error!(agent_id = body.agent_id, error = %e, "failed to load VC messages");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Render the system prompt and lark grammar via Askama templates
    let grammar_flow = GrammarFlow::new(&state.brand_name, &vc_messages).map_err(|e| {
        tracing::error!(error = %e, "failed to build GrammarFlow");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Compute embedding-based logit biases for the user's prompt.
    let category_biases = compute_category_biases(&state, &body.prompt, body.agent_id).await;

    // Persist session to SQLite
    let session_id = db::create_session(&state.db, &body.prompt)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "failed to create session");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Start generation — non-blocking
    let rx = state.engine.generate(body.prompt, grammar_flow, category_biases).await;

    // Store receiver so the SSE handler can pick it up
    state.sessions.lock().await.insert(session_id.clone(), rx);

    Ok(Json(InferResponse { session_id }))
}

/// Compute per-category logit biases for the given prompt.
///
/// Steps:
/// 1. Get OpenAI embedding of the prompt.
/// 2. Query Postgres for per-category margin scores.
/// 3. Fetch (or create with default 10.0) kappa for each message from SQLite.
/// 4. Return Vec<CategoryBias> with weighted_margin = kappa * margin.
///
/// On any failure the function logs a warning and returns an empty vec so that
/// generation proceeds normally without embedding guidance.
async fn compute_category_biases(
    state: &AppState,
    prompt: &str,
    agent_id: i32,
) -> Vec<CategoryBias> {
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(k) => k,
        Err(_) => {
            tracing::warn!("OPENAI_API_KEY not set — skipping embedding logit biases");
            return vec![];
        }
    };

    let embedding = match embedding::get_openai_embedding(prompt, &api_key).await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "OpenAI embedding failed — skipping logit biases");
            return vec![];
        }
    };

    let margins = match db::compute_embedding_margins(&state.vc_db, &embedding, agent_id).await {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(error = ?e, "margin query failed — skipping logit biases");
            return vec![];
        }
    };

    let mut biases = Vec::with_capacity(margins.len());
    for m in &margins {
        let kappa = db::get_or_create_kappa(&state.db, m.message_id)
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

/// GET /infer/stream/:session_id
///
/// Streams `InferenceEvent` values as Server-Sent Events.
/// Also persists each token and the final result to SQLite.
pub async fn stream_sse(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    let rx = state
        .sessions
        .lock()
        .await
        .remove(&session_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    let db = state.db.clone();
    let sid = session_id.clone();

    // Mark streaming in DB (best-effort, don't fail the SSE on DB error)
    let _ = db::set_session_streaming(&db, &sid).await;

    let db2 = db.clone();
    let sid2 = sid.clone();
    let mut position: i64 = 0;

    let stream = ReceiverStream::new(rx).map(move |event| {
        // Persist token events to SQLite in a fire-and-forget task
        match &event {
            InferenceEvent::Token(step) => {
                let db = db2.clone();
                let sid = sid2.clone();
                let text = step.chosen.text.clone();
                let token_id = step.chosen.token_id as i64;
                let prob = step.chosen.probability as f64;
                let logit = step.chosen.logit as f64;
                let pos = position;
                position += 1;
                tokio::spawn(async move {
                    if let Err(e) =
                        db::insert_token(&db, &sid, pos, &text, token_id, prob, logit).await
                    {
                        tracing::warn!(error = %e, "failed to persist token");
                    }
                });
            }
            InferenceEvent::Done { full_text } => {
                let db = db2.clone();
                let sid = sid2.clone();
                let text = full_text.clone();
                tokio::spawn(async move {
                    if let Err(e) = db::complete_session(&db, &sid, &text).await {
                        tracing::warn!(error = %e, "failed to complete session");
                    }
                });
            }
            InferenceEvent::Error { message } => {
                tracing::error!(session_id = %sid2, error = %message, "inference error");
            }
        }

        let data = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
        Ok(Event::default().data(data))
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}
