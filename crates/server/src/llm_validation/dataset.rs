use std::collections::HashMap;

use anyhow::Context;
use inference::VCmessage;
use inference_types::{AgentVersionInfo, ValidationAmendment};
use sqlx::PgPool;

use crate::db::{HcpExample, VcMessageWithId};

/// Version-scoped working copy used by the validation loop.
#[derive(Clone)]
pub struct ValidationDataset {
    pub agent_id: i32,
    pub version_id: i32,
    pub agent_name: String,
    pub messages: Vec<VcMessageWithId>,
    /// Examples linked to exactly one VC message.
    pub examples: Vec<HcpExample>,
    /// example_id → single expected message_id
    pub expected: HashMap<i32, i32>,
}

pub async fn load_agent_versions(vc_db: &PgPool) -> anyhow::Result<Vec<AgentVersionInfo>> {
    #[derive(sqlx::FromRow)]
    struct Row {
        agentid: i32,
        agentname: Option<String>,
        versionid: i32,
        deploymentstatus: Option<String>,
        messagecount: Option<i32>,
    }

    let rows = sqlx::query_as::<_, Row>(
        r#"SELECT v.agentid,
                  a.agentname,
                  v.versionid,
                  v.deploymentstatus,
                  v.messagecount
           FROM vcagentversions v
           LEFT JOIN vcagents a ON a.id = v.agentid
           WHERE COALESCE(v.isdeleted, false) = false
             AND EXISTS (
               SELECT 1 FROM vcmessages m
               WHERE m.agentid = v.agentid
                 AND m.versionid = v.versionid
                 AND m.textcontent IS NOT NULL
                 AND m.categoryname IS NOT NULL
                 AND m.categoryname != 'conversation_flow'
                 AND (
                   EXISTS (SELECT 1 FROM vcembeddingmessages em WHERE em.messageid = m.id)
                   OR EXISTS (SELECT 1 FROM vcllmmessages lm WHERE lm.messageid = m.id)
                 )
             )
           ORDER BY a.agentname NULLS LAST, v.agentid, v.versionid DESC"#,
    )
    .fetch_all(vc_db)
    .await
    .context("failed to list agent versions")?;

    Ok(rows
        .into_iter()
        .map(|r| AgentVersionInfo {
            agent_id: r.agentid,
            agent_name: r
                .agentname
                .unwrap_or_else(|| format!("Agent {}", r.agentid)),
            version_id: r.versionid,
            deployment_status: r.deploymentstatus,
            message_count: r.messagecount,
        })
        .collect())
}

pub async fn load_validation_dataset(
    vc_db: &PgPool,
    agent_id: i32,
    version_id: i32,
) -> anyhow::Result<ValidationDataset> {
    #[derive(sqlx::FromRow)]
    struct NameRow {
        agentname: Option<String>,
    }
    let name_row = sqlx::query_as::<_, NameRow>(
        r#"SELECT agentname FROM vcagents WHERE id = $1"#,
    )
    .bind(agent_id)
    .fetch_optional(vc_db)
    .await
    .context("failed to load agent name")?;
    let agent_name = name_row
        .and_then(|r| r.agentname)
        .unwrap_or_else(|| format!("Agent {agent_id}"));

    let messages = load_version_messages(vc_db, agent_id, version_id).await?;
    anyhow::ensure!(
        !messages.is_empty(),
        "no valid VC messages for agent {agent_id} version {version_id}"
    );

    let (examples, expected) =
        load_single_link_examples(vc_db, agent_id, version_id, &messages).await?;
    anyhow::ensure!(
        !examples.is_empty(),
        "no single-link validation examples for agent {agent_id} version {version_id}"
    );

    Ok(ValidationDataset {
        agent_id,
        version_id,
        agent_name,
        messages,
        examples,
        expected,
    })
}

async fn load_version_messages(
    vc_db: &PgPool,
    agent_id: i32,
    version_id: i32,
) -> anyhow::Result<Vec<VcMessageWithId>> {
    #[derive(sqlx::FromRow)]
    struct Row {
        id: Option<i32>,
        categoryname: Option<String>,
        categorydescription: Option<String>,
        textcontent: Option<String>,
    }

    // Only include LLM / embedding subtype messages in the classifier catalog.
    let rows = sqlx::query_as::<_, Row>(
        r#"SELECT v.id, v.categoryname, v.categorydescription, v.textcontent
           FROM vcmessages v
           WHERE v.agentid = $1
             AND v.versionid = $2
             AND v.categoryisactive = true
             AND v.textcontent IS NOT NULL
             AND v.categoryname IS NOT NULL
             AND v.categoryname != 'conversation_flow'
             AND v.textcontent NOT LIKE '%{{conversation_flow}}%'
             AND v.textcontent != 'N/A'
             AND (
               EXISTS (SELECT 1 FROM vcembeddingmessages em WHERE em.messageid = v.id)
               OR EXISTS (SELECT 1 FROM vcllmmessages lm WHERE lm.messageid = v.id)
             )"#,
    )
    .bind(agent_id)
    .bind(version_id)
    .fetch_all(vc_db)
    .await
    .with_context(|| {
        format!("failed to load vcmessages for agent {agent_id} version {version_id}")
    })?;

    Ok(rows
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
        .collect())
}

async fn load_single_link_examples(
    vc_db: &PgPool,
    agent_id: i32,
    version_id: i32,
    messages: &[VcMessageWithId],
) -> anyhow::Result<(Vec<HcpExample>, HashMap<i32, i32>)> {
    let message_ids: HashMap<i32, ()> = messages.iter().map(|m| (m.id, ())).collect();

    #[derive(sqlx::FromRow)]
    struct Row {
        example_id: Option<i32>,
        example_text: Option<String>,
        message_id: Option<i32>,
    }

    let rows = sqlx::query_as::<_, Row>(
        r#"SELECT h.id AS example_id,
                  h.textcontent AS example_text,
                  j.messageid AS message_id
           FROM vchcpexamplemessages h
           JOIN vcmessagestohcpexamplemessages j ON j.examplemessageid = h.id
           WHERE h.agentid = $1
             AND h.versionid = $2
             AND h.textcontent IS NOT NULL
             AND h.id IN (
               SELECT examplemessageid
               FROM vcmessagestohcpexamplemessages
               GROUP BY examplemessageid
               HAVING COUNT(*) = 1
             )"#,
    )
    .bind(agent_id)
    .bind(version_id)
    .fetch_all(vc_db)
    .await
    .with_context(|| {
        format!("failed to load single-link examples for agent {agent_id} version {version_id}")
    })?;

    let mut examples = Vec::new();
    let mut expected = HashMap::new();
    for r in rows {
        let example_id = match r.example_id {
            Some(id) => id,
            None => continue,
        };
        let message_id = match r.message_id {
            Some(id) => id,
            None => continue,
        };
        if !message_ids.contains_key(&message_id) {
            continue;
        }
        let text = r.example_text.unwrap_or_default().trim().to_string();
        if text.is_empty() {
            continue;
        }
        examples.push(HcpExample {
            id: example_id,
            text,
        });
        expected.insert(example_id, message_id);
    }

    Ok((examples, expected))
}

pub fn apply_amendments_in_memory(
    dataset: &mut ValidationDataset,
    amendments: &[ValidationAmendment],
) {
    for amendment in amendments {
        match amendment {
            ValidationAmendment::Description {
                message_id,
                new_description,
                ..
            } => {
                if let Some(msg) = dataset.messages.iter_mut().find(|m| m.id == *message_id) {
                    msg.vc_message.description = new_description.clone();
                }
            }
            ValidationAmendment::ReassignValidation {
                example_id,
                to_message_id,
                ..
            } => {
                if dataset.messages.iter().any(|m| m.id == *to_message_id) {
                    dataset.expected.insert(*example_id, *to_message_id);
                }
            }
            ValidationAmendment::RemoveValidation { example_id, .. } => {
                dataset.expected.remove(example_id);
                dataset.examples.retain(|e| e.id != *example_id);
            }
        }
    }
}

pub async fn persist_amendments(
    vc_db: &PgPool,
    amendments: &[ValidationAmendment],
) -> anyhow::Result<usize> {
    let mut applied = 0usize;
    for amendment in amendments {
        match amendment {
            ValidationAmendment::Description {
                message_id,
                new_description,
                ..
            } => {
                sqlx::query(
                    r#"UPDATE vcmessages
                       SET categorydescription = $1,
                           updatedate = now()
                       WHERE id = $2"#,
                )
                .bind(new_description)
                .bind(message_id)
                .execute(vc_db)
                .await
                .with_context(|| format!("failed to update description for message {message_id}"))?;
                applied += 1;
            }
            ValidationAmendment::ReassignValidation {
                example_id,
                from_message_id,
                to_message_id,
                ..
            } => {
                let mut tx = vc_db.begin().await?;
                sqlx::query(
                    r#"DELETE FROM vcmessagestohcpexamplemessages
                       WHERE examplemessageid = $1 AND messageid = $2"#,
                )
                .bind(example_id)
                .bind(from_message_id)
                .execute(&mut *tx)
                .await?;
                sqlx::query(
                    r#"INSERT INTO vcmessagestohcpexamplemessages (examplemessageid, messageid)
                       VALUES ($1, $2)
                       ON CONFLICT DO NOTHING"#,
                )
                .bind(example_id)
                .bind(to_message_id)
                .execute(&mut *tx)
                .await?;
                tx.commit().await?;
                applied += 1;
            }
            ValidationAmendment::RemoveValidation {
                example_id,
                message_id,
                ..
            } => {
                sqlx::query(
                    r#"DELETE FROM vcmessagestohcpexamplemessages
                       WHERE examplemessageid = $1 AND messageid = $2"#,
                )
                .bind(example_id)
                .bind(message_id)
                .execute(vc_db)
                .await?;
                applied += 1;
            }
        }
    }
    Ok(applied)
}
