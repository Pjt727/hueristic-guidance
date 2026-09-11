use std::convert::Infallible;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
};
use inference_types::{ClassifierBackend, LlmValidationEvent};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::llm_validation::{
    apply_amendments, deny_and_retest, load_agent_versions, run_validation_loop,
};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct StartValidationRequest {
    pub agent_id: i32,
    pub version_id: i32,
    #[serde(default)]
    pub classifier_backend: ClassifierBackend,
}

#[derive(Serialize)]
pub struct StartValidationResponse {
    pub session_id: String,
}

#[derive(Deserialize)]
pub struct DecisionRequest {
    pub approve: bool,
}

/// GET /agent-versions
pub async fn list_versions(
    State(state): State<AppState>,
) -> Result<Json<Vec<inference_types::AgentVersionInfo>>, StatusCode> {
    load_agent_versions(&state.vc_db)
        .await
        .map(Json)
        .map_err(|e| {
            tracing::error!(error = %e, "list agent versions failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })
}

/// POST /llm-validation
pub async fn start_validation(
    State(state): State<AppState>,
    Json(body): Json<StartValidationRequest>,
) -> Result<Json<StartValidationResponse>, StatusCode> {
    let (tx, rx) = mpsc::channel::<LlmValidationEvent>(64);
    let session_id = Uuid::new_v4().to_string();
    state
        .llm_validation_sessions
        .lock()
        .await
        .insert(session_id.clone(), rx);

    let cancel = Arc::new(AtomicBool::new(false));
    state
        .llm_validation_cancel
        .lock()
        .await
        .insert(session_id.clone(), cancel.clone());

    let engine = state.engine.clone();
    let vc_db = state.vc_db.clone();
    let brand = state.brand_name.clone();
    let pending = state.llm_validation_pending.clone();
    let cancel_map = state.llm_validation_cancel.clone();
    let sid = session_id.clone();

    let classifier_backend = body.classifier_backend;
    tokio::spawn(async move {
        let session = run_validation_loop(
            engine,
            vc_db,
            brand,
            body.agent_id,
            body.version_id,
            classifier_backend,
            cancel,
            tx,
        )
        .await;
        cancel_map.lock().await.remove(&sid);
        if let Some(session) = session {
            pending.lock().await.insert(sid, session);
        }
    });

    Ok(Json(StartValidationResponse { session_id }))
}

/// POST /llm-validation/:session_id/cancel
pub async fn cancel_validation(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
) -> StatusCode {
    let map = state.llm_validation_cancel.lock().await;
    if let Some(flag) = map.get(&session_id) {
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

/// GET /llm-validation/stream/:session_id
pub async fn stream_validation(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    let rx = state
        .llm_validation_sessions
        .lock()
        .await
        .remove(&session_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    let stream = ReceiverStream::new(rx).map(|event| {
        let data = serde_json::to_string(&event).unwrap_or_else(|_| "{}".into());
        Ok(Event::default().data(data))
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// POST /llm-validation/:session_id/decision
pub async fn decide_validation(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
    Json(body): Json<DecisionRequest>,
) -> Result<Json<StartValidationResponse>, StatusCode> {
    let session = state
        .llm_validation_pending
        .lock()
        .await
        .remove(&session_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    let (tx, rx) = mpsc::channel::<LlmValidationEvent>(16);
    let stream_id = Uuid::new_v4().to_string();
    state
        .llm_validation_sessions
        .lock()
        .await
        .insert(stream_id.clone(), rx);

    let cancel = Arc::new(AtomicBool::new(false));
    state
        .llm_validation_cancel
        .lock()
        .await
        .insert(stream_id.clone(), cancel.clone());

    let engine = state.engine.clone();
    let vc_db = state.vc_db.clone();
    let brand = state.brand_name.clone();
    let cancel_map = state.llm_validation_cancel.clone();
    let sid = stream_id.clone();

    if body.approve {
        let amendments = session.proposed_amendments;
        tokio::spawn(async move {
            apply_amendments(&vc_db, &amendments, tx).await;
            cancel_map.lock().await.remove(&sid);
        });
    } else {
        let agent_id = session.dataset.agent_id;
        let version_id = session.dataset.version_id;
        let classifier_backend = session.classifier_backend;
        tokio::spawn(async move {
            deny_and_retest(
                engine,
                vc_db,
                brand,
                agent_id,
                version_id,
                classifier_backend,
                cancel,
                tx,
            )
            .await;
            cancel_map.lock().await.remove(&sid);
        });
    }

    Ok(Json(StartValidationResponse {
        session_id: stream_id,
    }))
}
