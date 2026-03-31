use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use inference::GrammarFlow;

use crate::db;
use crate::state::AppState;

/// GET /agents
///
/// Returns the sorted list of agent IDs that have at least one valid VC message.
pub async fn list_agents(State(state): State<AppState>) -> Result<Json<Vec<i32>>, StatusCode> {
    let ids = db::list_agent_ids(&state.vc_db).await.map_err(|e| {
        tracing::error!(error = %e, "failed to list agent IDs");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok(Json(ids))
}

/// GET /agents/:agent_id/system-prompt
///
/// Loads VC messages for the agent, renders the Askama system-prompt template,
/// and returns the plain text result. Used by the frontend's prompt preview modal.
pub async fn get_system_prompt(
    Path(agent_id): Path<i32>,
    State(state): State<AppState>,
) -> Result<String, StatusCode> {
    let vc_messages = db::load_vc_messages(&state.vc_db, agent_id)
        .await
        .map_err(|e| {
            tracing::error!(agent_id, error = %e, "failed to load VC messages for system prompt");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let grammar_flow = GrammarFlow::new(&state.brand_name, &vc_messages).map_err(|e| {
        tracing::error!(error = %e, "failed to render system prompt template");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(grammar_flow.system_prompt.clone())
}
