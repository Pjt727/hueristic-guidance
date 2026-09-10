//! OpenAI Embedding classifier — uses text-embedding-3-large for cosine similarity.
//!
//! Precomputes category description embeddings at construction time,
//! then classifies queries by cosine similarity to the nearest category.

use std::time::Instant;

use anyhow::Context;
use serde::Deserialize;

use crate::{
    CategoryDef, CategoryScore, ClassificationResult, Classifier, ConversationContext,
    cosine_similarity, timed_classify,
};

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

/// OpenAI embedding-based classifier.
///
/// Precomputes category description embeddings at construction time.
/// Classification is a single embedding call + cosine similarity.
pub struct OpenAiEmbeddingClassifier {
    api_key: String,
    /// (category_name, embedding_vector) — precomputed at construction.
    category_embeddings: Vec<(String, Vec<f32>)>,
}

impl OpenAiEmbeddingClassifier {
    /// Create a new classifier by embedding all category descriptions.
    ///
    /// This makes one batch API call to OpenAI to precompute embeddings.
    pub async fn new(api_key: String, categories: &[CategoryDef]) -> anyhow::Result<Self> {
        let texts: Vec<&str> = categories.iter().map(|c| c.description.as_str()).collect();
        let embeddings = get_openai_embeddings_batch(&texts, &api_key).await?;

        let category_embeddings: Vec<(String, Vec<f32>)> = categories
            .iter()
            .zip(embeddings)
            .map(|(cat, emb)| (cat.name.clone(), emb))
            .collect();

        Ok(Self {
            api_key,
            category_embeddings,
        })
    }

    /// Create from pre-existing category embeddings (avoids API call).
    pub fn from_precomputed(api_key: String, category_embeddings: Vec<(String, Vec<f32>)>) -> Self {
        Self {
            api_key,
            category_embeddings,
        }
    }
}

#[async_trait::async_trait]
impl Classifier for OpenAiEmbeddingClassifier {
    fn name(&self) -> &str {
        "openai_embedding"
    }

    async fn classify(
        &self,
        query: &str,
        _categories: &[CategoryDef],
        context: Option<&ConversationContext>,
    ) -> anyhow::Result<ClassificationResult> {
        let start = Instant::now();

        let effective_query = match context {
            Some(ctx) => ctx.enrich_query(query),
            None => query.to_string(),
        };

        let query_embedding = get_openai_embedding(&effective_query, &self.api_key).await?;

        let scores: Vec<CategoryScore> = self
            .category_embeddings
            .iter()
            .map(|(name, cat_emb)| {
                let sim = cosine_similarity(&query_embedding, cat_emb);
                // Normalize from [-1, 1] to [0, 1]
                let score = (sim + 1.0) / 2.0;
                CategoryScore {
                    category_name: name.clone(),
                    score,
                }
            })
            .collect();

        Ok(timed_classify(self.name(), start, scores))
    }
}
