use std::collections::HashMap;

use llguidance::toktrie::SimpleVob;

pub type TokenID = u32;

#[derive(Clone)]
pub struct Canidate {
    pub token_id: TokenID,
    pub probability: f32,
    pub logit: f32,
    /// Additive logit bias from embedding similarity (w(v)). Zero until apply_biases() is called.
    pub embedding_logit: f32,
}

pub struct Canidates {
    canidates: Vec<Canidate>,
    /// O(1) lookup: token_id → index in `canidates`. Rebuilt after every sort.
    by_id: HashMap<TokenID, usize>,
}

impl Canidates {
    pub fn new(mut canidates: Vec<Canidate>) -> Self {
        canidates.sort_by(|c1, c2| c2.logit.total_cmp(&c1.logit));
        let by_id = build_index(&canidates);
        Self { canidates, by_id }
    }

    /// Look up a candidate by token ID in O(1).
    pub fn get_by_id(&self, token_id: TokenID) -> Option<&Canidate> {
        self.by_id.get(&token_id).map(|&i| &self.canidates[i])
    }

    /// Apply per-token logit biases from the embedding similarity algorithm.
    /// Sets each candidate's `embedding_logit` to the precomputed w(v) value.
    /// The raw `logit` field is left unchanged; adjusted logit = logit + embedding_logit.
    /// Re-sorts by adjusted logit so that top_n() reflects the new ordering.
    pub fn apply_biases(&mut self, biases: &HashMap<TokenID, f32>) {
        for c in &mut self.canidates {
            c.embedding_logit = biases.get(&c.token_id).copied().unwrap_or(0.0);
        }
        self.canidates
            .sort_by(|a, b| (b.logit + b.embedding_logit).total_cmp(&(a.logit + a.embedding_logit)));
        self.by_id = build_index(&self.canidates);
    }

    pub fn constrain(&mut self, mask: &SimpleVob) {
        let new_canidates: Vec<_> = self
            .canidates
            .iter()
            .filter(|&c| mask.is_allowed(c.token_id))
            .cloned()
            .collect();
        self.by_id = build_index(&new_canidates);
        self.canidates = new_canidates;
    }

    /// Returns the top-N candidates with probabilities renormalized via softmax
    /// over the adjusted logit (logit + embedding_logit) for this subset.
    pub fn top_n(&self, n: usize) -> Vec<Canidate> {
        let mut top: Vec<Canidate> = self.canidates.iter().take(n).cloned().collect();
        if top.is_empty() {
            return top;
        }
        let max_logit = top
            .iter()
            .map(|c| c.logit + c.embedding_logit)
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top
            .iter()
            .map(|c| (c.logit + c.embedding_logit - max_logit).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        for (c, exp) in top.iter_mut().zip(exps.iter()) {
            c.probability = exp / sum;
        }
        top
    }
}

fn build_index(canidates: &[Canidate]) -> HashMap<TokenID, usize> {
    canidates
        .iter()
        .enumerate()
        .map(|(i, c)| (c.token_id, i))
        .collect()
}
