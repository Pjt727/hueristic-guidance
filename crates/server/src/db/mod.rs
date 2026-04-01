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

/// Overwrite the kappa value for a single message.
pub async fn set_kappa(db: &SqlitePool, message_id: i64, kappa: f64) -> anyhow::Result<()> {
    sqlx::query!(
        "INSERT INTO vc_message_constants (message_id, kappa) VALUES (?, ?)
         ON CONFLICT(message_id) DO UPDATE SET kappa = excluded.kappa",
        message_id,
        kappa,
    )
    .execute(db)
    .await
    .context("failed to set kappa for message")?;
    Ok(())
}

/// Return the agent_id stored in a bulk_test_run row.
pub async fn get_run_agent_id(db: &SqlitePool, run_id: i64) -> anyhow::Result<i64> {
    let row = sqlx::query!(
        "SELECT agent_id FROM bulk_test_runs WHERE id = ?",
        run_id,
    )
    .fetch_one(db)
    .await
    .context("failed to fetch agent_id for run")?;
    Ok(row.agent_id)
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
// Bulk test — Postgres (read-only)
// ---------------------------------------------------------------------------

/// One HCP example message used as a test prompt.
pub struct HcpExample {
    pub id: i32,
    pub text: String,
}

/// A VC message with its Postgres primary key.
pub struct VcMessageWithId {
    pub id: i32,
    pub vc_message: VCmessage,
}

/// Load all HCP example messages for the latest version of `agent_id`.
pub async fn load_hcp_example_messages(
    vc_db: &PgPool,
    agent_id: i32,
) -> anyhow::Result<Vec<HcpExample>> {
    #[derive(sqlx::FromRow)]
    struct Row {
        id: Option<i32>,
        textcontent: Option<String>,
    }

    let rows = sqlx::query_as::<_, Row>(
        r#"SELECT id, textcontent
           FROM vchcpexamplemessages
           WHERE agentid  = $1
             AND versionid = (SELECT MAX(versionid) FROM vchcpexamplemessages WHERE agentid = $1)
             AND textcontent IS NOT NULL"#,
    )
    .bind(agent_id)
    .fetch_all(vc_db)
    .await
    .with_context(|| format!("failed to load HCP example messages for agent {agent_id}"))?;

    let examples = rows
        .into_iter()
        .filter_map(|r| {
            Some(HcpExample {
                id: r.id?,
                text: r.textcontent?.trim().to_string(),
            })
        })
        .filter(|e| !e.text.is_empty())
        .collect();

    Ok(examples)
}

/// Load the correct-answer map for all HCP examples of the latest version of
/// `agent_id`. Returns a HashMap from `examplemessageid` → `Vec<messageid>`.
pub async fn load_correct_answer_map(
    vc_db: &PgPool,
    agent_id: i32,
) -> anyhow::Result<std::collections::HashMap<i32, Vec<i32>>> {
    #[derive(sqlx::FromRow)]
    struct Row {
        examplemessageid: Option<i32>,
        messageid: Option<i32>,
    }

    let rows = sqlx::query_as::<_, Row>(
        r#"SELECT j.examplemessageid, j.messageid
           FROM vcmessagestohcpexamplemessages j
           JOIN vchcpexamplemessages h ON h.id = j.examplemessageid
           WHERE h.agentid   = $1
             AND h.versionid = (SELECT MAX(versionid) FROM vchcpexamplemessages WHERE agentid = $1)"#,
    )
    .bind(agent_id)
    .fetch_all(vc_db)
    .await
    .with_context(|| format!("failed to load correct answer map for agent {agent_id}"))?;

    let mut map: std::collections::HashMap<i32, Vec<i32>> = std::collections::HashMap::new();
    for r in rows {
        if let (Some(eid), Some(mid)) = (r.examplemessageid, r.messageid) {
            map.entry(eid).or_default().push(mid);
        }
    }
    Ok(map)
}

/// Like `load_vc_messages` but also returns the Postgres `id` for each row.
pub async fn load_vc_messages_with_ids(
    vc_db: &PgPool,
    agent_id: i32,
) -> anyhow::Result<Vec<VcMessageWithId>> {
    #[derive(sqlx::FromRow)]
    struct Row {
        id: Option<i32>,
        categoryname: Option<String>,
        categorydescription: Option<String>,
        textcontent: Option<String>,
    }

    let rows = sqlx::query_as::<_, Row>(
        r#"SELECT id, categoryname, categorydescription, textcontent
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
    .context("failed to load vcmessages with IDs from marketing DB")?;

    let messages = rows
        .into_iter()
        .filter_map(|r| {
            let id = r.id?;
            let category = r.categoryname?.trim().to_string();
            let description = r.categorydescription.unwrap_or_default().trim().to_string();
            let raw_text = r.textcontent?;
            let message = raw_text
                .replace("{{conversation_continuer}}", "")
                .trim()
                .to_string();
            if message.is_empty() || category.starts_with("qpharma.") {
                return None;
            }
            Some(VcMessageWithId {
                id,
                vc_message: VCmessage {
                    category,
                    kind: String::new(),
                    description,
                    mlr_message: message.clone(),
                    message,
                },
            })
        })
        .collect();

    Ok(messages)
}

// ---------------------------------------------------------------------------
// Bulk test persistence — SQLite
// Uses sqlx::query (no !) to avoid offline cache regeneration for new tables.
// ---------------------------------------------------------------------------

/// Create a new bulk test run row and return its SQLite row ID.
pub async fn create_bulk_test_run(db: &SqlitePool, agent_id: i32) -> anyhow::Result<i64> {
    let aid = agent_id as i64;
    let result = sqlx::query!(
        "INSERT INTO bulk_test_runs (agent_id) VALUES (?)",
        aid,
    )
    .execute(db)
    .await
    .context("failed to insert bulk_test_run")?;
    Ok(result.last_insert_rowid())
}

/// Persist one example result within a bulk test run.
pub async fn insert_bulk_test_result(
    db: &SqlitePool,
    run_id: i64,
    example_id: i32,
    example_text: &str,
    chosen_category: Option<&str>,
    correct_categories_json: &str,
    success: bool,
    steps_json: &str,
) -> anyhow::Result<()> {
    let eid = example_id as i64;
    let ok = success as i64;
    sqlx::query!(
        "INSERT INTO bulk_test_results \
         (run_id, example_id, example_text, chosen_category, correct_categories, success, steps) \
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        run_id,
        eid,
        example_text,
        chosen_category,
        correct_categories_json,
        ok,
        steps_json,
    )
    .execute(db)
    .await
    .context("failed to insert bulk_test_result")?;
    Ok(())
}

/// Mark a run as finished and store the final totals.
pub async fn complete_bulk_test_run(
    db: &SqlitePool,
    run_id: i64,
    total: i64,
    success_count: i64,
) -> anyhow::Result<()> {
    sqlx::query!(
        "UPDATE bulk_test_runs \
         SET completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), \
             total = ?, success_count = ? \
         WHERE id = ?",
        total,
        success_count,
        run_id,
    )
    .execute(db)
    .await
    .context("failed to complete bulk_test_run")?;
    Ok(())
}

#[derive(serde::Serialize)]
pub struct BulkTestRunSummary {
    pub id: i64,
    pub agent_id: i64,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub total: Option<i64>,
    pub success_count: Option<i64>,
}

/// List the 50 most recent bulk test runs (newest first).
pub async fn list_bulk_test_runs(db: &SqlitePool) -> anyhow::Result<Vec<BulkTestRunSummary>> {
    let rows = sqlx::query!(
        "SELECT id, agent_id, started_at, completed_at, total, success_count \
         FROM bulk_test_runs ORDER BY started_at DESC LIMIT 50"
    )
    .fetch_all(db)
    .await
    .context("failed to list bulk_test_runs")?;

    Ok(rows
        .into_iter()
        .map(|r| BulkTestRunSummary {
            id: r.id,
            agent_id: r.agent_id,
            started_at: r.started_at,
            completed_at: r.completed_at,
            total: r.total,
            success_count: r.success_count,
        })
        .collect())
}

pub struct StoredBulkTestResult {
    pub example_id: i64,
    pub example_text: String,
    pub chosen_category: Option<String>,
    pub correct_categories_json: String,
    pub success: bool,
    pub steps_json: String,
}

/// Load all results for a given run, ordered by insertion.
pub async fn load_bulk_test_results(
    db: &SqlitePool,
    run_id: i64,
) -> anyhow::Result<Vec<StoredBulkTestResult>> {
    let rows = sqlx::query!(
        "SELECT example_id, example_text, chosen_category, correct_categories, success, steps \
         FROM bulk_test_results WHERE run_id = ? ORDER BY id",
        run_id,
    )
    .fetch_all(db)
    .await
    .context("failed to load bulk_test_results")?;

    Ok(rows
        .into_iter()
        .map(|r| StoredBulkTestResult {
            example_id: r.example_id,
            example_text: r.example_text,
            chosen_category: r.chosen_category,
            correct_categories_json: r.correct_categories,
            success: r.success != 0,
            steps_json: r.steps,
        })
        .collect())
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
