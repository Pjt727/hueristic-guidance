//! Weight & temperature optimization using historical bulk-test data.
//!
//! Given per-example classifier scores and correct categories from a bulk test
//! run, this module finds:
//! 1. Optimal ensemble weights (grid search over TF-IDF vs embedding)
//! 2. Optimal embedding softmax temperature (better confidence calibration)
//! 3. Per-category bias offsets derived from LLM logit residuals

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Per-example data needed for optimization.
#[derive(Debug, Clone)]
pub struct ExampleScores {
    /// Category name → raw score from TF-IDF classifier.
    pub tfidf_scores: HashMap<String, f64>,
    /// Category name → raw score from OpenAI embedding classifier.
    pub embedding_scores: HashMap<String, f64>,
    /// Category names that are acceptable correct answers.
    pub correct_categories: Vec<String>,
    /// Category name → LLM adjusted logit (logit + embedding_logit) from the
    /// first decision step, if available.
    pub llm_logits: Option<HashMap<String, f32>>,
    /// Category name → raw embedding sim_score from CategoryTopToken, if available.
    pub category_sim_scores: Option<HashMap<String, f32>>,
}

/// One row in the weight search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightPoint {
    pub tfidf_weight: f64,
    pub embedding_weight: f64,
    pub correct: usize,
    pub total: usize,
    pub accuracy_pct: f64,
}

/// One row in the temperature search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperaturePoint {
    pub temperature: f64,
    pub correct: usize,
    pub total: usize,
    pub accuracy_pct: f64,
}

/// Per-category LLM logit stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryLogitStats {
    pub avg_logit: f64,
    pub avg_probability: f64,
    pub times_chosen: usize,
    pub times_correct: usize,
    pub sample_count: usize,
}

/// LLM logit-derived category bias offsets for embedding score correction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryBiasOffset {
    pub category_name: String,
    /// Average residual: mean(softmax(llm_logit)) - mean(embed_score).
    /// Positive means the LLM favors this category more than embeddings do.
    pub bias_offset: f64,
    /// How many examples this was computed from.
    pub sample_count: usize,
}

/// Full optimization result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub examples_analyzed: usize,
    /// Per-method standalone accuracy.
    pub baseline_accuracy: HashMap<String, f64>,
    /// Grid search over ensemble weights.
    pub weight_search: Vec<WeightPoint>,
    /// Optimal weight configuration.
    pub optimal_weights: WeightPoint,
    /// Temperature calibration results for embedding classifier.
    pub temperature_search: Vec<TemperaturePoint>,
    /// Best embedding temperature.
    pub optimal_temperature: TemperaturePoint,
    /// LLM logit accuracy (when available).
    pub llm_accuracy_pct: Option<f64>,
    /// Per-category LLM logit statistics.
    pub llm_category_stats: HashMap<String, CategoryLogitStats>,
    /// Category bias offsets derived from LLM logits vs embedding scores.
    pub category_bias_offsets: Vec<CategoryBiasOffset>,
    /// Accuracy when applying bias offsets to embedding scores.
    pub bias_corrected_accuracy_pct: Option<f64>,
}

/// Run the full optimization pipeline on a set of example scores.
pub fn optimize(examples: &[ExampleScores]) -> OptimizationResult {
    let total = examples.len();

    // Baseline accuracy for each standalone method.
    let tfidf_correct = count_correct(examples, |ex| &ex.tfidf_scores);
    let embed_correct = count_correct(examples, |ex| &ex.embedding_scores);

    let mut baseline_accuracy = HashMap::new();
    baseline_accuracy.insert(
        "tfidf".to_string(),
        pct(tfidf_correct, total),
    );
    baseline_accuracy.insert(
        "openai_embedding".to_string(),
        pct(embed_correct, total),
    );

    // 1. Weight grid search.
    let weight_search = weight_grid_search(examples, 21); // 0.00, 0.05, ..., 1.00
    let optimal_weights = weight_search
        .iter()
        .max_by(|a, b| {
            a.accuracy_pct
                .partial_cmp(&b.accuracy_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
        .unwrap_or(WeightPoint {
            tfidf_weight: 0.5,
            embedding_weight: 0.5,
            correct: 0,
            total,
            accuracy_pct: 0.0,
        });

    // 2. Embedding temperature search — tests softmax(embed/T) + TF-IDF ensemble.
    // Small T amplifies tiny embedding score differences; large T flattens them.
    let temperature_search = temperature_grid_search(examples, &[
        0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0,
    ]);
    let optimal_temperature = temperature_search
        .iter()
        .max_by(|a, b| {
            a.accuracy_pct
                .partial_cmp(&b.accuracy_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
        .unwrap_or(TemperaturePoint {
            temperature: 1.0,
            correct: 0,
            total,
            accuracy_pct: 0.0,
        });

    // 3. LLM logit analysis.
    let (llm_accuracy_pct, llm_category_stats) = analyze_llm_logits(examples);

    // 4. Category bias offsets.
    let category_bias_offsets = compute_bias_offsets(examples);
    let bias_corrected_accuracy_pct = if category_bias_offsets.is_empty() {
        None
    } else {
        Some(eval_bias_corrected(examples, &category_bias_offsets))
    };

    OptimizationResult {
        examples_analyzed: total,
        baseline_accuracy,
        weight_search,
        optimal_weights,
        temperature_search,
        optimal_temperature,
        llm_accuracy_pct,
        llm_category_stats,
        category_bias_offsets,
        bias_corrected_accuracy_pct,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn pct(correct: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64 * 100.0
    }
}

fn count_correct(
    examples: &[ExampleScores],
    get_scores: impl Fn(&ExampleScores) -> &HashMap<String, f64>,
) -> usize {
    examples
        .iter()
        .filter(|ex| {
            let scores = get_scores(ex);
            let chosen = scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(name, _)| name.as_str());
            match chosen {
                Some(c) => ex.correct_categories.iter().any(|cc| cc == c),
                None => false,
            }
        })
        .count()
}

/// Grid search over tfidf_weight ∈ [0, 1] with `steps` evenly spaced points.
fn weight_grid_search(examples: &[ExampleScores], steps: usize) -> Vec<WeightPoint> {
    let total = examples.len();
    (0..steps)
        .map(|i| {
            let w_tfidf = i as f64 / (steps - 1) as f64;
            let w_embed = 1.0 - w_tfidf;
            let correct = examples
                .iter()
                .filter(|ex| {
                    let combined = combine_scores(&ex.tfidf_scores, &ex.embedding_scores, w_tfidf, w_embed);
                    let chosen = combined
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(name, _)| name.as_str());
                    match chosen {
                        Some(c) => ex.correct_categories.iter().any(|cc| cc == c),
                        None => false,
                    }
                })
                .count();
            WeightPoint {
                tfidf_weight: (w_tfidf * 100.0).round() / 100.0,
                embedding_weight: (w_embed * 100.0).round() / 100.0,
                correct,
                total,
                accuracy_pct: pct(correct, total),
            }
        })
        .collect()
}

fn combine_scores(
    tfidf: &HashMap<String, f64>,
    embed: &HashMap<String, f64>,
    w_tfidf: f64,
    w_embed: f64,
) -> HashMap<String, f64> {
    let mut combined: HashMap<String, f64> = HashMap::new();
    for (name, &score) in tfidf {
        *combined.entry(name.clone()).or_default() += w_tfidf * score;
    }
    for (name, &score) in embed {
        *combined.entry(name.clone()).or_default() += w_embed * score;
    }
    combined
}

/// Search over different softmax temperatures for embedding scores combined
/// with equal-weight TF-IDF. Temperature-scaled embedding scores are
/// `softmax(raw / T)`, which changes the score magnitudes before ensemble
/// combination — unlike raw scores where the tight clustering makes the
/// embedding signal negligible.
fn temperature_grid_search(
    examples: &[ExampleScores],
    temperatures: &[f64],
) -> Vec<TemperaturePoint> {
    let total = examples.len();
    temperatures
        .iter()
        .map(|&temp| {
            let correct = examples
                .iter()
                .filter(|ex| {
                    // Apply softmax(embedding_scores / T), then equal-weight
                    // ensemble with raw TF-IDF scores.
                    let scaled_embed = softmax_scale(&ex.embedding_scores, temp);
                    let combined = combine_with_tfidf(&ex.tfidf_scores, &scaled_embed, 0.5, 0.5);
                    let chosen = combined
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(name, _)| name.as_str());
                    match chosen {
                        Some(c) => ex.correct_categories.iter().any(|cc| cc.as_str() == c),
                        None => false,
                    }
                })
                .count();
            TemperaturePoint {
                temperature: temp,
                correct,
                total,
                accuracy_pct: pct(correct, total),
            }
        })
        .collect()
}

/// Apply softmax temperature scaling to scores: `exp(s / T) / sum(exp(s / T))`.
/// Lower T → more peaked distribution (amplifies small differences).
fn softmax_scale(scores: &HashMap<String, f64>, temperature: f64) -> HashMap<String, f64> {
    if scores.is_empty() || temperature == 0.0 {
        return scores.clone();
    }
    let max_s = scores
        .values()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: HashMap<&str, f64> = scores
        .iter()
        .map(|(name, &s)| (name.as_str(), ((s - max_s) / temperature).exp()))
        .collect();
    let sum: f64 = exps.values().sum();
    if sum == 0.0 {
        return scores.clone();
    }
    exps.into_iter()
        .map(|(name, e)| (name.to_string(), e / sum))
        .collect()
}

fn combine_with_tfidf(
    tfidf: &HashMap<String, f64>,
    scaled_embed: &HashMap<String, f64>,
    w_tfidf: f64,
    w_embed: f64,
) -> HashMap<String, f64> {
    let mut combined: HashMap<String, f64> = HashMap::new();
    for (name, &score) in tfidf {
        *combined.entry(name.clone()).or_default() += w_tfidf * score;
    }
    for (name, &score) in scaled_embed {
        *combined.entry(name.clone()).or_default() += w_embed * score;
    }
    combined
}

/// Analyze LLM logit data from the first decision step.
fn analyze_llm_logits(
    examples: &[ExampleScores],
) -> (Option<f64>, HashMap<String, CategoryLogitStats>) {
    let examples_with_logits: Vec<_> = examples
        .iter()
        .filter(|ex| ex.llm_logits.is_some())
        .collect();

    if examples_with_logits.is_empty() {
        return (None, HashMap::new());
    }

    let total = examples_with_logits.len();
    let mut correct = 0usize;
    let mut stats: HashMap<String, (f64, f64, usize, usize, usize)> = HashMap::new();

    for ex in &examples_with_logits {
        let logits = ex.llm_logits.as_ref().unwrap();
        if logits.is_empty() {
            continue;
        }

        // Compute softmax probabilities from LLM logits.
        let max_logit = logits
            .values()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: HashMap<&str, f32> = logits
            .iter()
            .map(|(name, &logit)| (name.as_str(), (logit - max_logit).exp()))
            .collect();
        let sum: f32 = exps.values().sum();

        let probs: HashMap<&str, f32> = exps
            .iter()
            .map(|(&name, &exp_val)| (name, if sum > 0.0 { exp_val / sum } else { 0.0 }))
            .collect();

        // Argmax
        let chosen = logits
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, _)| name.as_str());

        let is_correct = match chosen {
            Some(c) => ex.correct_categories.iter().any(|cc| cc == c),
            None => false,
        };
        if is_correct {
            correct += 1;
        }

        for (name, &logit) in logits {
            let prob = probs.get(name.as_str()).copied().unwrap_or(0.0);
            let (sum_logit, sum_prob, chosen_cnt, correct_cnt, count) =
                stats.entry(name.clone()).or_insert((0.0, 0.0, 0, 0, 0));
            *sum_logit += logit as f64;
            *sum_prob += prob as f64;
            *count += 1;
            if chosen == Some(name.as_str()) {
                *chosen_cnt += 1;
                if is_correct {
                    *correct_cnt += 1;
                }
            }
        }
    }

    let cat_stats: HashMap<String, CategoryLogitStats> = stats
        .into_iter()
        .map(|(name, (sum_logit, sum_prob, chosen, cor, count))| {
            (
                name,
                CategoryLogitStats {
                    avg_logit: if count > 0 { sum_logit / count as f64 } else { 0.0 },
                    avg_probability: if count > 0 { sum_prob / count as f64 } else { 0.0 },
                    times_chosen: chosen,
                    times_correct: cor,
                    sample_count: count,
                },
            )
        })
        .collect();

    (Some(pct(correct, total)), cat_stats)
}

/// Compute per-category bias offsets: mean(llm_softmax_prob) - mean(embed_score).
/// These can be added to embedding scores to "nudge" them toward the LLM's
/// category preferences.
fn compute_bias_offsets(examples: &[ExampleScores]) -> Vec<CategoryBiasOffset> {
    // Accumulate: category → (sum_llm_prob, sum_embed_score, count)
    let mut accum: HashMap<String, (f64, f64, usize)> = HashMap::new();

    for ex in examples {
        let logits = match &ex.llm_logits {
            Some(l) if !l.is_empty() => l,
            _ => continue,
        };

        // LLM softmax probabilities.
        let max_logit = logits
            .values()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<(&str, f32)> = logits
            .iter()
            .map(|(name, &logit)| (name.as_str(), (logit - max_logit).exp()))
            .collect();
        let sum: f32 = exps.iter().map(|(_, e)| e).sum();

        for (name, exp_val) in &exps {
            let llm_prob = if sum > 0.0 { *exp_val as f64 / sum as f64 } else { 0.0 };
            let embed_score = ex
                .embedding_scores
                .get(*name)
                .copied()
                .unwrap_or(0.5); // fallback: neutral

            let (s_llm, s_embed, cnt) = accum.entry(name.to_string()).or_insert((0.0, 0.0, 0));
            *s_llm += llm_prob;
            *s_embed += embed_score;
            *cnt += 1;
        }
    }

    let mut offsets: Vec<CategoryBiasOffset> = accum
        .into_iter()
        .map(|(name, (s_llm, s_embed, count))| {
            let avg_llm = s_llm / count as f64;
            let avg_embed = s_embed / count as f64;
            CategoryBiasOffset {
                category_name: name,
                bias_offset: avg_llm - avg_embed,
                sample_count: count,
            }
        })
        .collect();

    offsets.sort_by(|a, b| {
        b.bias_offset
            .abs()
            .partial_cmp(&a.bias_offset.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    offsets
}

/// Evaluate accuracy when adding bias offsets to embedding scores.
fn eval_bias_corrected(
    examples: &[ExampleScores],
    offsets: &[CategoryBiasOffset],
) -> f64 {
    let offset_map: HashMap<&str, f64> = offsets
        .iter()
        .map(|o| (o.category_name.as_str(), o.bias_offset))
        .collect();

    let total = examples.len();
    let correct = examples
        .iter()
        .filter(|ex| {
            let corrected: HashMap<String, f64> = ex
                .embedding_scores
                .iter()
                .map(|(name, &score)| {
                    let offset = offset_map.get(name.as_str()).copied().unwrap_or(0.0);
                    (name.clone(), score + offset)
                })
                .collect();
            let chosen = corrected
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(name, _)| name.as_str());
            match chosen {
                Some(c) => ex.correct_categories.iter().any(|cc| cc == c),
                None => false,
            }
        })
        .count();

    pct(correct, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_example(
        tfidf: &[(&str, f64)],
        embed: &[(&str, f64)],
        correct: &[&str],
    ) -> ExampleScores {
        ExampleScores {
            tfidf_scores: tfidf.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            embedding_scores: embed.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            correct_categories: correct.iter().map(|s| s.to_string()).collect(),
            llm_logits: None,
            category_sim_scores: None,
        }
    }

    #[test]
    fn weight_search_finds_best() {
        let examples = vec![
            // TF-IDF picks A (correct), embedding picks B (wrong)
            make_example(
                &[("A", 0.8), ("B", 0.2)],
                &[("A", 0.3), ("B", 0.7)],
                &["A"],
            ),
            // TF-IDF picks B (wrong), embedding picks A (correct)
            make_example(
                &[("A", 0.3), ("B", 0.7)],
                &[("A", 0.8), ("B", 0.2)],
                &["A"],
            ),
        ];

        let result = optimize(&examples);
        // Equal weights should get both right (0.55 for A in both cases)
        assert_eq!(result.optimal_weights.correct, 2);
    }

    #[test]
    fn empty_examples() {
        let result = optimize(&[]);
        assert_eq!(result.examples_analyzed, 0);
    }
}
