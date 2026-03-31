pub type TokenID = u32;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenWithProb {
    pub text: String,
    pub token_id: TokenID,
    pub probability: f32,
    /// Raw logit from the language model (before embedding adjustment).
    pub logit: f32,
    /// Additive logit adjustment from embedding similarity (w(v) in the algorithm).
    /// 0.0 for grammar-forced fast-forward tokens.
    pub embedding_logit: f32,
}

/// The highest-probability token in `top_constrained` that is a prefix of a
/// given category name. Used to render the "best token per category" table.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CategoryTopToken {
    pub category_name: String,
    pub best_token: TokenWithProb,
}

/// Candidates at a single decoding step.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepCandidates {
    /// The token that was committed.
    pub chosen: TokenWithProb,
    /// Top-N candidates before the grammar mask was applied.
    pub top_alternatives: Vec<TokenWithProb>,
    /// Top-N candidates after the grammar mask was applied.
    pub top_constrained: Vec<TokenWithProb>,
    /// Best prefix-matching token for each category, from `top_constrained`.
    /// Empty for grammar-forced fast-forward tokens.
    pub category_top_tokens: Vec<CategoryTopToken>,
}

/// Events streamed from the inference engine to the server and frontend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceEvent {
    /// A single token was committed during generation.
    Token(StepCandidates),
    /// Generation finished; full assembled text is included.
    Done { full_text: String },
    /// An error occurred during generation.
    Error { message: String },
}
