// ---------------------------------------------------------------------------
// Application database — SQLite
// All queries use sqlx::query! for compile-time verification.
// DATABASE_URL must point to sqlite:./app.db at build time.
// ---------------------------------------------------------------------------

use anyhow::Context;
use sqlx::SqlitePool;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Inference sessions
// ---------------------------------------------------------------------------

/// Insert a new pending session. Returns the `obfuscated_id` (the public UUID
/// string used as the session key in all API responses).
pub async fn create_session(db: &SqlitePool, prompt: &str) -> anyhow::Result<String> {
    let obfuscated_id = Uuid::new_v4().to_string();
    sqlx::query!(
        "INSERT INTO inference_sessions (obfuscated_id, prompt) VALUES (?, ?)",
        obfuscated_id,
        prompt,
    )
    .execute(db)
    .await
    .context("failed to insert inference_session")?;
    Ok(obfuscated_id)
}

/// Mark a session as streaming.
pub async fn set_session_streaming(db: &SqlitePool, obfuscated_id: &str) -> anyhow::Result<()> {
    sqlx::query!(
        "UPDATE inference_sessions SET status = 'streaming' WHERE obfuscated_id = ?",
        obfuscated_id,
    )
    .execute(db)
    .await
    .context("failed to update session to streaming")?;
    Ok(())
}

/// Mark a session complete and store the full generated text.
pub async fn complete_session(
    db: &SqlitePool,
    obfuscated_id: &str,
    result_text: &str,
) -> anyhow::Result<()> {
    sqlx::query!(
        r#"UPDATE inference_sessions
           SET status = 'complete', result_text = ?,
               completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
           WHERE obfuscated_id = ?"#,
        result_text,
        obfuscated_id,
    )
    .execute(db)
    .await
    .context("failed to complete session")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Inference tokens
// ---------------------------------------------------------------------------

/// Append one token event to the session's token log.
pub async fn insert_token(
    db: &SqlitePool,
    obfuscated_session_id: &str,
    position: i64,
    token_text: &str,
    token_id: i64,
    probability: f64,
    logit: f64,
) -> anyhow::Result<()> {
    sqlx::query!(
        r#"INSERT INTO inference_tokens
               (session_id, position, token_text, token_id, probability, logit)
           VALUES (
               (SELECT id FROM inference_sessions WHERE obfuscated_id = ?),
               ?, ?, ?, ?, ?
           )"#,
        obfuscated_session_id,
        position,
        token_text,
        token_id,
        probability,
        logit,
    )
    .execute(db)
    .await
    .context("failed to insert inference_token")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Embedding constants — SQLite (kappa scaling per VC message)
// ---------------------------------------------------------------------------

/// Retrieve the kappa constant for a given message_id, inserting the default
/// (10.0) row if this message has not been seen before.
pub async fn get_or_create_kappa(db: &SqlitePool, message_id: i64) -> anyhow::Result<f64> {
    sqlx::query!(
        "INSERT OR IGNORE INTO vc_message_constants (message_id) VALUES (?)",
        message_id,
    )
    .execute(db)
    .await
    .context("failed to upsert vc_message_constants row")?;

    let row = sqlx::query!(
        "SELECT kappa FROM vc_message_constants WHERE message_id = ?",
        message_id,
    )
    .fetch_one(db)
    .await
    .context("failed to fetch kappa for message")?;

    Ok(row.kappa)
}

// ---------------------------------------------------------------------------
// VC database — Postgres (read-only)
// Uses sqlx::query_as with typed structs — no query! macro because this is an
// external schema we do not own and DATABASE_URL points to SQLite.
// ---------------------------------------------------------------------------

use inference::VCmessage;
use sqlx::PgPool;

/// Raw row returned from the marketing `vcmessages` table.
#[derive(sqlx::FromRow)]
struct VcMessageRow {
    categoryname: Option<String>,
    categorydescription: Option<String>,
    textcontent: Option<String>,
}

/// List all agent IDs that have at least one valid VC message.
pub async fn list_agent_ids(vc_db: &PgPool) -> anyhow::Result<Vec<i32>> {
    let ids = sqlx::query_scalar::<_, i32>(
        r#"SELECT DISTINCT agentid
           FROM vcmessages
           WHERE textcontent IS NOT NULL
             AND categoryname IS NOT NULL
             AND categoryname != 'conversation_flow'
             AND textcontent NOT LIKE '%{{conversation_flow}}%'
             AND textcontent != 'N/A'
           ORDER BY agentid"#,
    )
    .fetch_all(vc_db)
    .await
    .context("failed to list agent IDs from marketing DB")?;
    Ok(ids)
}

/// Load approved VC messages for a given agent from the marketing Postgres DB.
/// Filters out placeholder-only rows and cleans `{{conversation_continuer}}` tags.
pub async fn load_vc_messages(
    vc_db: &PgPool,
    agent_id: i32,
) -> anyhow::Result<Vec<VCmessage>> {
    let rows = sqlx::query_as::<_, VcMessageRow>(
        r#"SELECT categoryname, categorydescription, textcontent
           FROM vcmessages
           WHERE agentid    = $1
             AND textcontent   IS NOT NULL
             AND categoryname  IS NOT NULL
             AND categoryname  != 'conversation_flow'
             AND textcontent NOT LIKE '%{{conversation_flow}}%'
             AND textcontent   != 'N/A'"#,
    )
    .bind(agent_id)
    .fetch_all(vc_db)
    .await
    .context("failed to load vcmessages from marketing DB")?;

    let messages: Vec<VCmessage> = rows
        .into_iter()
        .filter_map(|r| {
            let category = r.categoryname?.trim().to_string();
            let description = r
                .categorydescription
                .unwrap_or_default()
                .trim()
                .to_string();
            let raw_text = r.textcontent?;
            let message = raw_text
                .replace("{{conversation_continuer}}", "")
                .trim()
                .to_string();
            if message.is_empty() || category.starts_with("qpharma.") {
                return None;
            }
            Some(VCmessage {
                category,
                kind: String::new(),
                description,
                mlr_message: message.clone(),
                message,
            })
        })
        .collect();

    anyhow::ensure!(!messages.is_empty(), "no valid VC messages found for agent {agent_id}");
    Ok(messages)
}

// ---------------------------------------------------------------------------
// Embedding margin scores — Postgres
// ---------------------------------------------------------------------------

/// Per-message cosine-similarity margin from the embedding query.
pub struct MessageMargin {
    /// Postgres integer (int4) cast to i64 for SQLite compatibility.
    pub message_id: i64,
    pub category_name: String,
    /// pgvector cosine distance returns float8 (f64).
    pub margin: f64,
}

#[derive(sqlx::FromRow)]
struct MessageMarginRow {
    /// Postgres integer (int4) → i32 in sqlx.
    message_id: Option<i32>,
    message_identifier: Option<String>,
    /// pgvector cosine distance result is double precision (float8) → f64.
    margin: Option<f64>,
}

/// Query cosine-similarity margins between `embedding` and the positive/negative
/// example embeddings stored in `vcembeddingmessagefinalparameters` for the
/// latest version of the given agent.
///
/// Returns one row per message category, ordered by descending margin.
pub async fn compute_embedding_margins(
    vc_db: &PgPool,
    embedding: &[f32],
    agent_id: i32,
) -> anyhow::Result<Vec<MessageMargin>> {
    let vec = pgvector::Vector::from(embedding.to_vec());

    let rows = sqlx::query_as::<_, MessageMarginRow>(
        r#"WITH similarities AS (
            SELECT
              f.messageid AS id,
              m.categoryName AS message_identifier,
              MAX(CASE WHEN f.kind = 'pos'
                       THEN 1 - (f.embeddingData <=> $1)
                  END) AS pos_similarity,
              MAX(CASE WHEN f.kind = 'neg'
                       THEN 1 - (f.embeddingData <=> $1)
                  END) AS neg_similarity
            FROM vcembeddingmessagefinalparameters f
            JOIN vcmessages m ON f.messageid = m.id
            WHERE m.agentid = $2
              AND m.versionid = (SELECT MAX(versionid) FROM vcmessages WHERE agentid = $2)
            GROUP BY f.messageid, m.categoryName
        )
        SELECT
            id        AS message_id,
            message_identifier,
            (pos_similarity - neg_similarity) AS margin
        FROM similarities
        ORDER BY (pos_similarity - neg_similarity) DESC"#,
    )
    .bind(vec)
    .bind(agent_id)
    .fetch_all(vc_db)
    .await
    .with_context(|| {
        format!("failed to compute embedding margins for agent {agent_id}")
    })?;

    let margins = rows
        .into_iter()
        .filter_map(|r| {
            Some(MessageMargin {
                message_id: r.message_id? as i64,
                category_name: r.message_identifier?,
                margin: r.margin?,
            })
        })
        .collect();

    Ok(margins)
}
