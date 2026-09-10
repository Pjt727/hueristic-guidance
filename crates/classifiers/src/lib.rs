pub mod calibration;
pub mod ensemble;
pub mod openai_embedding;
pub mod tfidf;

#[cfg(feature = "onnx")]
pub mod local_embedding;

use std::time::Instant;

use serde::{Deserialize, Serialize};

/// A category definition passed to classifiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryDef {
    pub name: String,
    pub description: String,
}

/// Optional conversation history for context-sensitive classification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConversationContext {
    pub previous_messages: Vec<String>,
}

impl ConversationContext {
    /// Build a context-enriched query by prepending previous messages.
    /// Returns the original query if there is no history.
    pub fn enrich_query(&self, query: &str) -> String {
        if self.previous_messages.is_empty() {
            return query.to_string();
        }
        let mut enriched = String::new();
        for msg in &self.previous_messages {
            enriched.push_str(msg);
            enriched.push(' ');
        }
        enriched.push_str(query);
        enriched
    }
}

/// One classification score for a category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryScore {
    pub category_name: String,
    /// Score in [0, 1] where higher = more confident.
    pub score: f64,
}

/// Full result from a classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub method_name: String,
    /// All category scores, sorted by score descending.
    pub scores: Vec<CategoryScore>,
    /// The chosen category (highest score).
    pub chosen_category: String,
    /// Confidence of the chosen category (its score).
    pub confidence: f64,
    /// Wall-clock time for classification in milliseconds.
    pub latency_ms: u64,
}

/// All classifiers implement this trait.
#[async_trait::async_trait]
pub trait Classifier: Send + Sync {
    fn name(&self) -> &str;

    async fn classify(
        &self,
        query: &str,
        categories: &[CategoryDef],
        context: Option<&ConversationContext>,
    ) -> anyhow::Result<ClassificationResult>;
}

/// Cosine similarity between two f32 vectors. Returns 0.0 for zero-length vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Cosine similarity between f64 vectors.
pub fn cosine_similarity_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Helper: measure classification time and build the result.
pub fn timed_classify(
    method_name: &str,
    start: Instant,
    mut scores: Vec<CategoryScore>,
) -> ClassificationResult {
    scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let chosen_category = scores
        .first()
        .map(|s| s.category_name.clone())
        .unwrap_or_default();
    // Confidence via margin sigmoid: how much does the winner beat the runner-up?
    // sigmoid(k * margin) — independent of category count and score scale.
    // k=20 maps: margin 0.02→60%, 0.05→73%, 0.10→88%, 0.15→95%.
    let confidence = match scores.as_slice() {
        [] => 0.0,
        [_] => 1.0,
        [winner, runner_up, ..] => {
            let margin = winner.score - runner_up.score;
            let k = 20.0_f64;
            1.0 / (1.0 + (-k * margin).exp())
        }
    };
    let latency_ms = start.elapsed().as_millis() as u64;

    ClassificationResult {
        method_name: method_name.to_string(),
        scores,
        chosen_category,
        confidence,
        latency_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_identical_vectors() {
        let a = vec![1.0f32, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors should have similarity 1.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors should have similarity 0.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6, "opposite vectors should have similarity -1.0, got {sim}");
    }

    #[test]
    fn conversation_context_enriches_query() {
        let ctx = ConversationContext {
            previous_messages: vec!["What is Pemazyre?".to_string()],
        };
        let enriched = ctx.enrich_query("What about dosing?");
        assert!(enriched.starts_with("What is Pemazyre?"));
        assert!(enriched.ends_with("What about dosing?"));
    }

    #[test]
    fn conversation_context_empty_returns_original() {
        let ctx = ConversationContext::default();
        let enriched = ctx.enrich_query("test query");
        assert_eq!(enriched, "test query");
    }
}
