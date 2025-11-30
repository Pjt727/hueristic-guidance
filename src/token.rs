use llama_cpp_2::token::LlamaToken;
use llguidance::{toktrie::SimpleVob, Constraint};

pub type TokenID = u32;

#[derive(Clone)]
pub struct Canidate {
    pub token_id: TokenID,
    pub probability: f32,
    pub logit: f32,
}

pub struct Canidates {
    canidates: Vec<Canidate>,
}

impl Canidates {
    pub fn new(mut canidates: Vec<Canidate>) -> Self {
        canidates.sort_by(|c1, c2| c2.probability.total_cmp(&c1.probability));
        Self { canidates }
    }

    pub fn constrain(&mut self, mask: &SimpleVob) {
        let new_canidates: Vec<_> = self
            .canidates
            .iter()
            .filter(|c| mask.is_allowed(c.token_id))
            .collect();

        // normalize the probability distribution
        let total: f32 = new_canidates.iter().map(|c| c.probability).sum();
        for c in &mut self.canidates {
            c.probability /= total;
        }
    }

    pub fn top_n(&self, n: usize) -> Vec<Canidate> {
        self.canidates.iter().take(n).cloned().collect()
    }

    pub fn top_n_by_logits(&mut self, n: usize) -> Vec<Canidate> {
        self.canidates
            .sort_by(|c1, c2| c2.logit.total_cmp(&c1.logit));
        self.canidates.iter().take(n).cloned().collect()
    }
}
