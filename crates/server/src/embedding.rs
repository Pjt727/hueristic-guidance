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
}

/// Normalize text before embedding: lowercase and strip punctuation.
fn normalize(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect::<String>()
        .to_lowercase()
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
