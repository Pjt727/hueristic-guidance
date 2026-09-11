//! LLM-only validation loop: find description / association errors without embeddings.
//!
//! Pipeline stages (kept in separate modules on purpose):
//! 1. `dataset`  — version-scoped messages + single-link validation examples
//! 2. `evaluate` — classify with local LLM or OpenAI (empty embedding biases)
//! 3. `analyze`  — OpenAI batch-analyzes all confusions and proposes amendments
//! 4. `amend`    — merge proposals into a *minimal* change set
//! 5. `orchestrate` — rounds, stop rules, approve/deny apply

mod amend;
mod analyze;
mod dataset;
mod evaluate;
mod orchestrate;

pub use dataset::load_agent_versions;
pub use orchestrate::{
    apply_amendments, deny_and_retest, run_validation_loop, LlmValidationSession,
};
