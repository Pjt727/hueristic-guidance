use std::collections::HashMap;
use std::sync::Arc;

use inference::InferenceEngine;
use inference_types::InferenceEvent;
use sqlx::{PgPool, SqlitePool};
use tokio::sync::{mpsc, Mutex};

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
    /// Postgres pool — marketing VC database, read-only queries only.
    pub vc_db: PgPool,
    /// In-memory map of session obfuscated_id → live receiver for that stream.
    /// Removed and owned by the SSE handler when the client connects.
    pub sessions: Arc<Mutex<HashMap<String, mpsc::Receiver<InferenceEvent>>>>,
}
