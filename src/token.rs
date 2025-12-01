use llguidance::toktrie::SimpleVob;

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
        canidates.sort_by(|c1, c2| c2.logit.total_cmp(&c1.logit));
        Self { canidates }
    }

    pub fn constrain(&mut self, mask: &SimpleVob) {
        let new_canidates: Vec<_> = self
            .canidates
            .iter()
            .filter(|&c| mask.is_allowed(c.token_id))
            .cloned()
            .collect();

        self.canidates = new_canidates;
    }

    pub fn top_n(&mut self, n: usize) -> Vec<Canidate> {
        self.canidates.iter().take(n).cloned().collect()
    }
}
