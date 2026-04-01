//! Per-category embedding weight optimisation via gradient descent.
//!
//! ## Problem
//! The inference engine scores each category as:
//!   `total_score[c] = logit[c] + kappa[c] * sim_score[c]`
//!
//! This module finds per-category kappa values that maximise correct
//! classifications over a validation set, subject to kappa ≥ 0 (so we
//! never *penalise* a category for having a high embedding similarity).
//!
//! ## Algorithm — Gradient Descent with Cross-Entropy Loss
//!
//! For each example, compute a softmax distribution over category scores and
//! minimise the cross-entropy between that distribution and the ground-truth
//! one-hot for the correct category.  This is a smooth surrogate for
//! maximising classification accuracy.
//!
//! Optimiser: **Adam** (`β₁=0.9, β₂=0.999, ε=1e-8`) for stable convergence.
//!
//! After each Adam update: clamp `kappa[i] = kappa[i].max(0.0)` to enforce
//! non-negativity, so the embedding can only *help* a category.
//!
//! Initialise kappa = 0 (minimum starting point per minimum-norm requirement).

use std::collections::HashMap;

/// Logit and embedding similarity score for one category in one example.
pub struct CategoryScore {
    pub logit: f32,
    /// Raw embedding margin for this category (before kappa multiplication).
    /// 0.0 if embedding biases were not active for this run.
    pub sim_score: f32,
}

/// All per-category scores for one validation example.
pub struct ExampleData {
    /// Keyed by category name.
    pub category_scores: HashMap<String, CategoryScore>,
    pub correct_categories: Vec<String>,
}

// ---------------------------------------------------------------------------
// Hyper-parameters
// ---------------------------------------------------------------------------

const N_ITERS: usize = 3000;
const LR: f64 = 0.05;
const LAMBDA: f64 = 1e-4; // L2 regularisation — encourages minimum kappa
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;

/// Find per-category kappa values that maximise classification accuracy.
///
/// Returns a map from category name → optimal kappa (≥ 0).  Categories with
/// `sim_score = 0.0` on every example are omitted (no embedding data).
///
/// Returns `None` when there are fewer than 2 categories or no usable examples.
pub fn optimize_weights(examples: &[ExampleData]) -> Option<HashMap<String, f64>> {
    // Collect the ordered set of categories that have at least one non-zero sim_score.
    let categories: Vec<String> = {
        let mut seen: HashMap<String, bool> = HashMap::new();
        for ex in examples {
            for (name, score) in &ex.category_scores {
                let entry = seen.entry(name.clone()).or_insert(false);
                if score.sim_score != 0.0 {
                    *entry = true;
                }
            }
        }
        let mut cats: Vec<String> = seen
            .into_iter()
            .filter(|(_, has_data)| *has_data)
            .map(|(name, _)| name)
            .collect();
        cats.sort();
        cats
    };

    let n_cats = categories.len();
    if n_cats < 2 {
        return None;
    }

    let cat_index: HashMap<&str, usize> = categories
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    // Filter to examples that have at least one correct category in our set
    // and at least one incorrect category in our set.
    let usable: Vec<&ExampleData> = examples
        .iter()
        .filter(|ex| {
            let has_correct = ex
                .correct_categories
                .iter()
                .any(|g| cat_index.contains_key(g.as_str()));
            let has_incorrect = ex
                .category_scores
                .keys()
                .any(|c| cat_index.contains_key(c.as_str()) && !ex.correct_categories.contains(c));
            has_correct && has_incorrect
        })
        .collect();

    if usable.is_empty() {
        return None;
    }

    // Pre-extract per-example data into flat arrays for efficient iteration.
    // For each example we store all (cat_idx, logit, sim_score) triples and
    // the set of correct category indices.
    struct ExFlat {
        scores: Vec<(usize, f64, f64)>, // (cat_idx, logit, sim)
        correct_mask: Vec<bool>,        // indexed by position in `scores`
    }

    let flat: Vec<ExFlat> = usable
        .iter()
        .map(|ex| {
            let mut scores: Vec<(usize, f64, f64)> = ex
                .category_scores
                .iter()
                .filter_map(|(name, cs)| {
                    let idx = *cat_index.get(name.as_str())?;
                    Some((idx, cs.logit as f64, cs.sim_score as f64))
                })
                .collect();
            scores.sort_by_key(|(idx, _, _)| *idx);
            let correct_mask: Vec<bool> = scores
                .iter()
                .map(|(_, _, _)| false) // placeholder; fill below
                .collect();
            let _ = correct_mask; // suppress warning
            let correct_mask: Vec<bool> = scores
                .iter()
                .map(|(idx, _, _)| {
                    categories
                        .get(*idx)
                        .map(|name| ex.correct_categories.contains(name))
                        .unwrap_or(false)
                })
                .collect();
            ExFlat { scores, correct_mask }
        })
        .collect();

    // ---------------------------------------------------------------------------
    // Adam optimiser state
    // ---------------------------------------------------------------------------
    let mut kappa = vec![0.0f64; n_cats];
    let mut m = vec![0.0f64; n_cats]; // first moment
    let mut v = vec![0.0f64; n_cats]; // second moment

    for iter in 1..=N_ITERS {
        let mut grad = vec![0.0f64; n_cats];

        for ex in &flat {
            // Compute scores: s[k] = logit[k] + kappa[cat_idx[k]] * sim[k]
            let n = ex.scores.len();
            let mut s: Vec<f64> = ex
                .scores
                .iter()
                .map(|(idx, logit, sim)| logit + kappa[*idx] * sim)
                .collect();

            // Numerically stable softmax.
            let s_max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            for x in s.iter_mut() {
                *x = (*x - s_max).exp();
            }
            let sum: f64 = s.iter().sum();
            let prob: Vec<f64> = s.iter().map(|x| x / sum).collect();

            // Target: uniform over correct categories.
            let n_correct = ex.correct_mask.iter().filter(|&&b| b).count();
            if n_correct == 0 {
                continue;
            }
            let target_val = 1.0 / n_correct as f64;

            // Cross-entropy gradient w.r.t. kappa:
            // ∂L/∂kappa[c] = Σ_k sim[k] * (prob[k] - target[k])   (where k iterates over positions)
            // but only for k where cat_idx[k] == c.
            for pos in 0..n {
                let (cat_idx, _, sim) = ex.scores[pos];
                let target = if ex.correct_mask[pos] { target_val } else { 0.0 };
                let delta = prob[pos] - target;
                grad[cat_idx] += sim * delta;
            }
        }

        let n_ex = flat.len() as f64;

        // Add L2 regularisation gradient: λ * kappa
        for i in 0..n_cats {
            grad[i] = grad[i] / n_ex + LAMBDA * kappa[i];
        }

        // Adam update.
        let lr_t = LR * (1.0 - BETA2.powi(iter as i32)).sqrt() / (1.0 - BETA1.powi(iter as i32));
        for i in 0..n_cats {
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * grad[i];
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * grad[i] * grad[i];
            kappa[i] -= lr_t * m[i] / (v[i].sqrt() + EPS);
            // Non-negativity constraint.
            kappa[i] = kappa[i].max(0.0);
        }
    }

    let result: HashMap<String, f64> = categories.into_iter().zip(kappa).collect();
    Some(result)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_score(logit: f32, sim: f32) -> CategoryScore {
        CategoryScore { logit, sim_score: sim }
    }

    /// With two categories and an example where the incorrect category has a
    /// higher logit but the correct one has higher sim_score, the optimiser
    /// should return kappa_A > kappa_B.
    #[test]
    fn returns_positive_weight_for_easy_case() {
        let ex = ExampleData {
            category_scores: HashMap::from([
                ("A".to_string(), make_score(1.0, 2.0)), // correct, low logit, high sim
                ("B".to_string(), make_score(2.0, 0.5)), // incorrect, high logit, low sim
            ]),
            correct_categories: vec!["A".to_string()],
        };
        let weights = optimize_weights(&[ex]).expect("should return weights");
        assert!(
            weights["A"] > weights["B"],
            "kappa_A ({}) should exceed kappa_B ({})",
            weights["A"],
            weights["B"]
        );
    }

    /// All returned kappas must be non-negative.
    #[test]
    fn weights_are_non_negative() {
        let examples: Vec<ExampleData> = vec![
            ExampleData {
                category_scores: HashMap::from([
                    ("A".to_string(), make_score(3.0, 0.1)),
                    ("B".to_string(), make_score(1.0, 0.9)),
                    ("C".to_string(), make_score(2.0, 0.5)),
                ]),
                correct_categories: vec!["B".to_string()],
            },
            ExampleData {
                category_scores: HashMap::from([
                    ("A".to_string(), make_score(1.0, 0.8)),
                    ("B".to_string(), make_score(2.0, 0.2)),
                    ("C".to_string(), make_score(3.0, 0.1)),
                ]),
                correct_categories: vec!["A".to_string()],
            },
        ];
        let weights = optimize_weights(&examples).expect("should return weights");
        for (cat, w) in &weights {
            assert!(*w >= 0.0, "kappa for {cat} is negative: {w}");
        }
    }

    /// With no sim_score data (all zeros), the function should return None.
    #[test]
    fn returns_none_when_no_sim_data() {
        let ex = ExampleData {
            category_scores: HashMap::from([
                ("A".to_string(), make_score(2.0, 0.0)),
                ("B".to_string(), make_score(1.0, 0.0)),
            ]),
            correct_categories: vec!["A".to_string()],
        };
        assert!(optimize_weights(&[ex]).is_none());
    }
}
