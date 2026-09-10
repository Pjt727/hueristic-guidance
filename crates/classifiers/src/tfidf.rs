//! TF-IDF + Cosine Similarity classifier.
//!
//! Pure Rust, zero ML dependencies. Builds TF-IDF vectors from category
//! descriptions and classifies queries by cosine similarity.

use std::collections::HashMap;
use std::time::Instant;

use crate::{
    CategoryDef, CategoryScore, ClassificationResult, Classifier, ConversationContext,
    cosine_similarity_f64, timed_classify,
};

/// TF-IDF classifier — deterministic, offline, sub-millisecond.
pub struct TfIdfClassifier {
    /// Term → vocabulary index.
    vocabulary: HashMap<String, usize>,
    /// IDF values indexed by vocabulary position.
    idf: Vec<f64>,
    /// Pre-computed L2-normalized TF-IDF vectors per category.
    category_vectors: Vec<(String, Vec<f64>)>,
}

impl TfIdfClassifier {
    /// Build a TF-IDF model from category definitions.
    ///
    /// Each category's description is treated as a document. The vocabulary,
    /// IDF weights, and per-category TF-IDF vectors are precomputed.
    pub fn new(categories: &[CategoryDef]) -> Self {
        let docs: Vec<Vec<String>> = categories.iter().map(|c| tokenize(&c.description)).collect();

        // Build vocabulary from all terms across all documents.
        let mut vocab_map: HashMap<String, usize> = HashMap::new();
        for doc in &docs {
            for term in doc {
                let next_idx = vocab_map.len();
                vocab_map.entry(term.clone()).or_insert(next_idx);
            }
        }

        let vocab_size = vocab_map.len();
        let n_docs = docs.len() as f64;

        // Compute IDF: log(N / df(t)) where df(t) is the number of documents containing term t.
        let mut df = vec![0usize; vocab_size];
        for doc in &docs {
            let mut seen = HashMap::new();
            for term in doc {
                if let Some(&idx) = vocab_map.get(term) {
                    seen.entry(idx).or_insert(true);
                }
            }
            for &idx in seen.keys() {
                df[idx] += 1;
            }
        }

        let idf: Vec<f64> = df
            .iter()
            .map(|&d| {
                if d == 0 {
                    0.0
                } else {
                    (n_docs / d as f64).ln() + 1.0 // smoothed IDF
                }
            })
            .collect();

        // Build per-category TF-IDF vectors (L2-normalized).
        let category_vectors: Vec<(String, Vec<f64>)> = categories
            .iter()
            .zip(docs.iter())
            .map(|(cat, doc)| {
                let vec = build_tfidf_vector(doc, &vocab_map, &idf, vocab_size);
                (cat.name.clone(), vec)
            })
            .collect();

        Self {
            vocabulary: vocab_map,
            idf,
            category_vectors,
        }
    }

    fn classify_text(&self, text: &str) -> Vec<CategoryScore> {
        let query_tokens = tokenize(text);
        let query_vec =
            build_tfidf_vector(&query_tokens, &self.vocabulary, &self.idf, self.idf.len());

        self.category_vectors
            .iter()
            .map(|(name, cat_vec)| {
                let sim = cosine_similarity_f64(&query_vec, cat_vec);
                // Normalize from [-1, 1] to [0, 1]
                let score = (sim + 1.0) / 2.0;
                CategoryScore {
                    category_name: name.clone(),
                    score,
                }
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl Classifier for TfIdfClassifier {
    fn name(&self) -> &str {
        "tfidf"
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

        let scores = self.classify_text(&effective_query);
        Ok(timed_classify(self.name(), start, scores))
    }
}

/// Simple whitespace tokenizer: lowercase, strip punctuation, split on whitespace.
fn tokenize(text: &str) -> Vec<String> {
    text.chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() > 1) // drop single-char tokens
        .map(String::from)
        .collect()
}

/// Build a TF-IDF vector for a document, L2-normalized.
fn build_tfidf_vector(
    tokens: &[String],
    vocab: &HashMap<String, usize>,
    idf: &[f64],
    vocab_size: usize,
) -> Vec<f64> {
    let mut tf = vec![0.0f64; vocab_size];
    let n_tokens = tokens.len() as f64;

    if n_tokens == 0.0 {
        return tf;
    }

    for token in tokens {
        if let Some(&idx) = vocab.get(token) {
            tf[idx] += 1.0;
        }
    }

    // TF = count / total_tokens, then multiply by IDF
    let mut norm_sq = 0.0f64;
    for (i, val) in tf.iter_mut().enumerate() {
        *val = (*val / n_tokens) * idf[i];
        norm_sq += *val * *val;
    }

    // L2 normalize
    let norm = norm_sq.sqrt();
    if norm > 0.0 {
        for val in tf.iter_mut() {
            *val /= norm;
        }
    }

    tf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_categories() -> Vec<CategoryDef> {
        vec![
            CategoryDef {
                name: "Dosing".to_string(),
                description: "Information about drug dosing, administration, and schedule".to_string(),
            },
            CategoryDef {
                name: "Safety".to_string(),
                description: "Safety information including adverse events and side effects".to_string(),
            },
            CategoryDef {
                name: "Efficacy".to_string(),
                description: "Clinical trial results and effectiveness data".to_string(),
            },
        ]
    }

    #[test]
    fn classifies_dosing_query() {
        let cats = test_categories();
        let classifier = TfIdfClassifier::new(&cats);
        let scores = classifier.classify_text("What is the recommended dosing schedule?");
        let best = scores.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
        assert_eq!(best.category_name, "Dosing");
    }

    #[test]
    fn classifies_safety_query() {
        let cats = test_categories();
        let classifier = TfIdfClassifier::new(&cats);
        let scores = classifier.classify_text("What are the side effects and adverse events?");
        let best = scores.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
        assert_eq!(best.category_name, "Safety");
    }

    #[test]
    fn classifies_efficacy_query() {
        let cats = test_categories();
        let classifier = TfIdfClassifier::new(&cats);
        let scores = classifier.classify_text("Show me the clinical trial results");
        let best = scores.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
        assert_eq!(best.category_name, "Efficacy");
    }

    #[test]
    fn is_deterministic() {
        let cats = test_categories();
        let classifier = TfIdfClassifier::new(&cats);
        let query = "Tell me about dosing";
        let scores1 = classifier.classify_text(query);
        let scores2 = classifier.classify_text(query);
        for (a, b) in scores1.iter().zip(scores2.iter()) {
            assert_eq!(a.category_name, b.category_name);
            assert!((a.score - b.score).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn scores_are_in_valid_range() {
        let cats = test_categories();
        let classifier = TfIdfClassifier::new(&cats);
        let scores = classifier.classify_text("any query here");
        for s in &scores {
            assert!(s.score >= 0.0 && s.score <= 1.0, "score {} out of [0,1]", s.score);
        }
    }

    #[tokio::test]
    async fn trait_classify_with_context() {
        let cats = test_categories();
        let classifier = TfIdfClassifier::new(&cats);
        let ctx = ConversationContext {
            previous_messages: vec!["Tell me about the drug.".to_string()],
        };
        let result = classifier
            .classify("What is the dosing?", &cats, Some(&ctx))
            .await
            .unwrap();
        assert_eq!(result.method_name, "tfidf");
        assert!(!result.chosen_category.is_empty());
        assert!(result.latency_ms < 100); // should be sub-ms
    }
}
