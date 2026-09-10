//! Ensemble classifier — combines scores from multiple classifiers.
//!
//! Runs all inner classifiers, normalizes scores per-method to [0, 1],
//! computes weighted sum per category, and returns the top category.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::{
    CategoryDef, CategoryScore, ClassificationResult, Classifier, ConversationContext,
    timed_classify,
};

/// Ensemble classifier combining multiple classification methods.
pub struct EnsembleClassifier {
    /// (classifier, weight) pairs. Weights are relative — they are normalized
    /// internally so they sum to 1.0.
    classifiers: Vec<(Arc<dyn Classifier>, f64)>,
}

impl EnsembleClassifier {
    /// Create an ensemble from a list of (classifier, weight) pairs.
    /// Weights are relative and will be normalized to sum to 1.0.
    pub fn new(classifiers: Vec<(Arc<dyn Classifier>, f64)>) -> Self {
        Self { classifiers }
    }

    /// Create an equally-weighted ensemble from a list of classifiers.
    pub fn equal_weight(classifiers: Vec<Arc<dyn Classifier>>) -> Self {
        let weighted = classifiers.into_iter().map(|c| (c, 1.0)).collect();
        Self::new(weighted)
    }
}

#[async_trait::async_trait]
impl Classifier for EnsembleClassifier {
    fn name(&self) -> &str {
        "ensemble"
    }

    async fn classify(
        &self,
        query: &str,
        categories: &[CategoryDef],
        context: Option<&ConversationContext>,
    ) -> anyhow::Result<ClassificationResult> {
        let start = Instant::now();

        // Run all classifiers sequentially (could be parallelized with tokio::join
        // but classifiers may share resources like API rate limits).
        let mut sub_results: Vec<(f64, ClassificationResult)> = Vec::new();
        for (classifier, weight) in &self.classifiers {
            match classifier.classify(query, categories, context).await {
                Ok(result) => sub_results.push((*weight, result)),
                Err(e) => {
                    tracing::warn!(
                        classifier = classifier.name(),
                        error = %e,
                        "ensemble: sub-classifier failed, skipping"
                    );
                }
            }
        }

        if sub_results.is_empty() {
            anyhow::bail!("all sub-classifiers failed");
        }

        // Normalize weights to sum to 1.0.
        let total_weight: f64 = sub_results.iter().map(|(w, _)| w).sum();

        // Aggregate scores: weighted sum per category.
        let mut combined: HashMap<String, f64> = HashMap::new();
        for (weight, result) in &sub_results {
            let normalized_weight = weight / total_weight;
            for score in &result.scores {
                *combined.entry(score.category_name.clone()).or_default() +=
                    normalized_weight * score.score;
            }
        }

        let scores: Vec<CategoryScore> = combined
            .into_iter()
            .map(|(name, score)| CategoryScore {
                category_name: name,
                score,
            })
            .collect();

        Ok(timed_classify(self.name(), start, scores))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tfidf::TfIdfClassifier;

    #[tokio::test]
    async fn ensemble_with_single_classifier_matches() {
        let categories = vec![
            CategoryDef {
                name: "Dosing".to_string(),
                description: "Drug dosing and administration schedule".to_string(),
            },
            CategoryDef {
                name: "Safety".to_string(),
                description: "Safety adverse events side effects".to_string(),
            },
        ];

        let tfidf = Arc::new(TfIdfClassifier::new(&categories));
        let ensemble = EnsembleClassifier::equal_weight(vec![tfidf.clone()]);

        let query = "What is the dosing schedule?";
        let single = tfidf.classify(query, &categories, None).await.unwrap();
        let combined = ensemble.classify(query, &categories, None).await.unwrap();

        assert_eq!(single.chosen_category, combined.chosen_category);
    }
}
