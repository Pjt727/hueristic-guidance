pub type TokenID = u32;

/// An agent entry returned by GET /agents.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentInfo {
    pub id: i32,
    pub name: String,
}

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
    /// Raw embedding similarity margin for this category (before kappa multiplication
    /// and before token-level averaging). Used for per-category weight optimization.
    /// 0.0 for steps without embedding biases, and for data saved before this field existed.
    #[serde(default)]
    pub sim_score: f32,
    /// Maximum cosine similarity to a positive example for this category.
    #[serde(default)]
    pub positive_similarity: f32,
    /// Maximum cosine similarity to a negative example for this category.
    #[serde(default)]
    pub negative_similarity: f32,
    /// Best raw LLM score for this category at this decision step.
    #[serde(default)]
    pub pre_bias_logit: f32,
    /// Best score after adding the curved embedding bias.
    #[serde(default)]
    pub post_bias_logit: f32,
    /// Probability among category candidates before embedding guidance.
    #[serde(default)]
    pub pre_bias_probability: f32,
    /// Probability among category candidates after embedding guidance.
    #[serde(default)]
    pub post_bias_probability: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BiasRegime {
    #[default]
    None,
    Soft,
    Force,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverrideOutcome {
    #[default]
    None,
    SoftBias,
    ForcedEmbedding,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CategoryDistributionMetrics {
    pub winner_category: Option<String>,
    /// Difference between the highest and second-highest category scores.
    pub top_two_margin: f32,
    /// Entropy divided by ln(category_count), in the range 0..=1.
    pub normalized_entropy: f32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DecisionDiagnostics {
    pub pre_bias: CategoryDistributionMetrics,
    pub post_bias: CategoryDistributionMetrics,
    pub bias_regime: BiasRegime,
    pub force_target: Option<String>,
    pub override_outcome: OverrideOutcome,
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
    /// Category-level decision evidence. Present only at category decision steps.
    #[serde(default)]
    pub decision_diagnostics: Option<DecisionDiagnostics>,
}

/// Events streamed from the inference engine to the server and frontend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceEvent {
    /// A single token was committed during generation.
    Token(StepCandidates),
    /// Generation finished; full assembled text is included.
    Done {
        full_text: String,
        /// Wall time for the full generation loop (ms).
        #[serde(default)]
        latency_ms: u64,
        /// Wall time until the first category-decision step (ms), when available.
        #[serde(default)]
        category_latency_ms: Option<u64>,
    },
    /// An error occurred during generation.
    Error { message: String },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClassifierScore {
    pub category_name: String,
    pub score: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClassifierMethodResult {
    pub method_name: String,
    pub chosen_category: String,
    pub confidence: f64,
    pub latency_ms: u64,
    pub scores: Vec<ClassifierScore>,
}

/// A single test case result streamed during a bulk test run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BulkTestEvent {
    /// One test case has completed.
    Result {
        example_id: i32,
        example_text: String,
        /// The chosen category name, or None if inference errored.
        chosen_category: Option<String>,
        /// All category names that are acceptable correct answers for this example.
        correct_categories: Vec<String>,
        success: bool,
        steps: Vec<StepCandidates>,
        #[serde(default)]
        classifier_results: Vec<ClassifierMethodResult>,
    },
    /// All test cases have finished.
    Done { total: usize, success_count: usize },
    /// A fatal error aborted the bulk test.
    Error { message: String },
}

// ---------------------------------------------------------------------------
// LLM-only validation loop (description / validation association debugging)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentVersionInfo {
    pub agent_id: i32,
    pub agent_name: String,
    pub version_id: i32,
    pub deployment_status: Option<String>,
    pub message_count: Option<i32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ValidationAmendment {
    Description {
        message_id: i32,
        category_name: String,
        old_description: String,
        new_description: String,
        rationale: String,
    },
    ReassignValidation {
        example_id: i32,
        example_text: String,
        from_message_id: i32,
        from_category: String,
        to_message_id: i32,
        to_category: String,
        rationale: String,
    },
    RemoveValidation {
        example_id: i32,
        example_text: String,
        message_id: i32,
        category_name: String,
        rationale: String,
    },
}

/// Which model classifies examples during LLM validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassifierBackend {
    #[default]
    Local,
    Openai,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConfusionCase {
    pub example_id: i32,
    pub example_text: String,
    pub expected_category: String,
    pub expected_message_id: i32,
    pub predicted_category: Option<String>,
    pub predicted_message_id: Option<i32>,
    pub analysis: Option<String>,
    pub suggested_amendments: Vec<ValidationAmendment>,
    /// Local-LLM category decision steps with pre/post grammar logits.
    #[serde(default)]
    pub decision_steps: Vec<StepCandidates>,
    /// True when the top two unconstrained logits are close (ambiguous).
    #[serde(default)]
    pub close_top_logits: bool,
    /// True when high-logit unconstrained tokens are not grammar-allowed.
    #[serde(default)]
    pub non_grammar_top_logits: bool,
    /// Logit gap between #1 and #2 unconstrained candidates (when available).
    #[serde(default)]
    pub top_logit_margin: Option<f32>,
    /// True when this example (or same expected→predicted pair) failed in an earlier round.
    #[serde(default)]
    pub is_repeat_offender: bool,
    /// How many times this example has appeared as a confusion (including this round).
    #[serde(default)]
    pub recurrence_count: u32,
    /// Full history of prior amendments that failed to fix this pair, with per-attempt reasons.
    #[serde(default)]
    pub prior_amendment_failures: Vec<FailedAmendmentAttempt>,
}

/// One prior amendment that failed to clear a confusion, with an evidence-based reason.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FailedAmendmentAttempt {
    /// Round in which we recorded that this amendment had failed.
    pub round_observed: u32,
    pub example_id: i32,
    pub expected_category: String,
    pub predicted_category: Option<String>,
    pub amendment_summary: String,
    pub why_it_failed: String,
}

/// @deprecated alias kept for clarity in call sites that still say "lesson".
pub type AmendmentLesson = FailedAmendmentAttempt;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PassingCategoryStat {
    pub category_name: String,
    pub pass_count: usize,
    /// How many of those passes had close top-2 unconstrained logits.
    pub close_logit_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnalysisExecutiveSummary {
    /// How the reviewer prioritizes categories when several could fit.
    #[serde(default)]
    pub category_priority_assumptions: Vec<String>,
    /// Concrete decisions used to separate confused categories this round.
    #[serde(default)]
    pub differentiation_decisions: Vec<String>,
    /// Validation labels the reviewer challenged as likely wrong.
    #[serde(default)]
    pub label_challenges: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationRoundSummary {
    pub round: u32,
    pub total: usize,
    pub success_count: usize,
    pub accuracy_pct: f64,
    pub confusions: Vec<ConfusionCase>,
    pub amendments_applied: Vec<ValidationAmendment>,
    /// Per expected-category pass counts and how often those passes had close logits.
    #[serde(default)]
    pub passing_categories: Vec<PassingCategoryStat>,
    /// Total successful examples whose decision step had close top logits.
    #[serde(default)]
    pub passing_close_logit_count: usize,
    /// Batch-analysis executive summary for this round (if analysis ran).
    #[serde(default)]
    pub executive_summary: Option<AnalysisExecutiveSummary>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LlmValidationEvent {
    Status { message: String },
    RoundStarted { round: u32 },
    ExampleResult {
        round: u32,
        example_id: i32,
        success: bool,
        expected_category: String,
        predicted_category: Option<String>,
        #[serde(default)]
        close_top_logits: bool,
        #[serde(default)]
        non_grammar_top_logits: bool,
    },
    RoundCompleted { summary: ValidationRoundSummary },
    Analyzing { confusion_count: usize },
    ProposalReady {
        best_round: u32,
        best_accuracy_pct: f64,
        amendments: Vec<ValidationAmendment>,
        rounds: Vec<ValidationRoundSummary>,
        #[serde(default)]
        executive_summary: Option<AnalysisExecutiveSummary>,
    },
    Applied { applied_count: usize },
    DeniedRetest {
        total: usize,
        success_count: usize,
        accuracy_pct: f64,
    },
    Cancelled { message: String },
    Done,
    Error { message: String },
}

