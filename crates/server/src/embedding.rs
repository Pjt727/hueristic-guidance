use anyhow::Context;
use serde::Deserialize;

const OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";
const EMBEDDING_MODEL: &str = "text-embedding-3-large";

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingObject>,
}

#[derive(Deserialize)]
struct EmbeddingObject {
    embedding: Vec<f32>,
    index: usize,
}

/// Normalize text before embedding: lowercase and strip punctuation.
fn normalize(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect::<String>()
        .to_lowercase()
}

/// Call the OpenAI embeddings API for a batch of texts and return one embedding
/// vector per input, in the same order as `texts`.
///
/// All texts are normalized before sending. The OpenAI response `index` field
/// is used to reorder results correctly regardless of response ordering.
pub async fn get_openai_embeddings_batch(
    texts: &[&str],
    api_key: &str,
) -> anyhow::Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    let normalized: Vec<String> = texts.iter().map(|t| normalize(t)).collect();
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "input": normalized,
        "model": EMBEDDING_MODEL,
    });

    let resp = client
        .post(OPENAI_EMBEDDINGS_URL)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .context("failed to send OpenAI batch embedding request")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("OpenAI batch embedding API returned {status}: {text}");
    }

    let mut parsed: EmbeddingResponse = resp
        .json()
        .await
        .context("failed to parse OpenAI batch embedding response")?;

    // Sort by index so output order matches input order.
    parsed.data.sort_by_key(|o| o.index);

    anyhow::ensure!(
        parsed.data.len() == texts.len(),
        "OpenAI returned {} embeddings for {} inputs",
        parsed.data.len(),
        texts.len()
    );

    Ok(parsed.data.into_iter().map(|o| o.embedding).collect())
}

/// Call the OpenAI embeddings API and return the embedding vector for `text`.
/// The text is normalized (lowercased, punctuation removed) before sending.
/// Uses model `text-embedding-3-large` (3072 dimensions).
pub async fn get_openai_embedding(text: &str, api_key: &str) -> anyhow::Result<Vec<f32>> {
    let normalized = normalize(text);
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "input": normalized,
        "model": EMBEDDING_MODEL,
    });

    let resp = client
        .post(OPENAI_EMBEDDINGS_URL)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .context("failed to send OpenAI embedding request")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("OpenAI embedding API returned {status}: {text}");
    }

    let parsed: EmbeddingResponse = resp
        .json()
        .await
        .context("failed to parse OpenAI embedding response")?;

    parsed
        .data
        .into_iter()
        .next()
        .map(|o| o.embedding)
        .context("OpenAI embedding response contained no data")
}
