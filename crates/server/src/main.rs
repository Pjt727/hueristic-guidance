mod db;
mod embedding;
mod routes;
mod state;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    Router,
    routing::{get, post},
};
use inference::{InferenceConfig, InferenceEngine};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "server=info,tower_http=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // --- Application SQLite DB ----------------------------------------------
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite:./app.db".to_string());
    let db = sqlx::SqlitePool::connect(&database_url).await?;
    sqlx::migrate!("../../migrations").run(&db).await?;
    tracing::info!("SQLite migrations applied");

    // --- Marketing Postgres DB (read-only) ----------------------------------
    let vc_database_url = std::env::var("VC_DATABASE_URL")
        .unwrap_or_else(|_| "postgres://localhost:5432/marketing?sslmode=disable".to_string());
    let vc_db = sqlx::PgPool::connect(&vc_database_url).await?;
    tracing::info!("Connected to marketing Postgres DB");

    // --- Inference engine ---------------------------------------------------
    let model_path = PathBuf::from(
        std::env::var("MODEL_PATH")
            .unwrap_or_else(|_| "models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf".to_string()),
    );
    let context_cache_dir = PathBuf::from(
        std::env::var("CONTEXT_CACHE_DIR").unwrap_or_else(|_| "context_cache".to_string()),
    );

    let config = InferenceConfig {
        model_path,
        context_cache_dir,
        max_tokens: 200,
        top_candidate_count: 10,
    };

    tracing::info!("Loading inference engine (this may take a moment)…");
    let engine = tokio::task::spawn_blocking(|| InferenceEngine::new(config)).await??;
    let engine = Arc::new(engine);
    tracing::info!("Inference engine ready");

    let brand_name = std::env::var("BRAND_NAME").unwrap_or_else(|_| "Pemazyre".to_string());

    // --- App state ----------------------------------------------------------
    let state = AppState {
        engine,
        brand_name,
        db,
        vc_db,
        sessions: Arc::new(Mutex::new(HashMap::new())),
        bulk_test_sessions: Arc::new(Mutex::new(HashMap::new())),
    };

    // --- Router -------------------------------------------------------------
    let app = Router::new()
        .route("/health", get(routes::health::handler))
        .route("/agents", get(routes::agents::list_agents))
        .route("/agents/{agent_id}/system-prompt", get(routes::agents::get_system_prompt))
        .route("/infer", post(routes::infer::start_infer))
        .route("/infer/stream/{session_id}", get(routes::infer::stream_sse))
        .route("/bulk-test", post(routes::bulk_test::start_bulk_test))
        .route("/bulk-test/stream/{bulk_test_id}", get(routes::bulk_test::stream_bulk_test_sse))
        .route("/bulk-tests", get(routes::bulk_test::list_bulk_tests))
        .route("/bulk-tests/{run_id}", get(routes::bulk_test::get_bulk_test))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // --- Serve --------------------------------------------------------------
    let addr: SocketAddr = std::env::var("BIND_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:3000".to_string())
        .parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Listening on {addr}");
    axum::serve(listener, app).await?;

    Ok(())
}
