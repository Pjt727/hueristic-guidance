use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use classifiers::Classifier;
use inference::InferenceEngine;
use inference_types::{BulkTestEvent, InferenceEvent, LlmValidationEvent};
use sqlx::{PgPool, SqlitePool};
use tokio::sync::{mpsc, Mutex};

use crate::llm_validation::LlmValidationSession;

/// Shared application state threaded through every Axum handler.
#[derive(Clone)]
pub struct AppState {
    /// Inference engine (model loaded once at startup).
    pub engine: Arc<InferenceEngine>,
    /// Brand name used in the system prompt (from `BRAND_NAME` env var).
    pub brand_name: String,
    /// SQLite pool — application-owned tables (inference sessions, tokens).
    /// Compile-time checked via `sqlx::query!` with `DATABASE_URL=sqlite:./app.db`.
    pub db: SqlitePool,
    /// Postgres pool — marketing VC database.
    pub vc_db: PgPool,
    /// In-memory map of session obfuscated_id → live receiver for that stream.
    /// Removed and owned by the SSE handler when the client connects.
    pub sessions: Arc<Mutex<HashMap<String, mpsc::Receiver<InferenceEvent>>>>,
    /// In-memory map of bulk_test_id → live receiver for that bulk test stream.
    /// Removed and owned by the SSE handler when the client connects.
    pub bulk_test_sessions: Arc<Mutex<HashMap<String, mpsc::Receiver<BulkTestEvent>>>>,
    /// Live SSE receivers for LLM validation runs.
    pub llm_validation_sessions: Arc<Mutex<HashMap<String, mpsc::Receiver<LlmValidationEvent>>>>,
    /// Completed validation loops awaiting approve/deny.
    pub llm_validation_pending: Arc<Mutex<HashMap<String, LlmValidationSession>>>,
    /// Cancel flags for in-flight validation / decision streams.
    pub llm_validation_cancel: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    /// Standalone classifiers (OpenAI embedding).
    /// Initialized at startup based on available models/API keys.
    pub classifiers: Vec<Arc<dyn Classifier>>,
}
