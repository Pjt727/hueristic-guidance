use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{Special, params::LlamaModelParams, LlamaModel},
};
use llguidance::toktrie::TokenizerEnv;
use tokio::sync::mpsc;

use crate::constraints::new_default_constraint;
use crate::grammar::GrammarFlow;
use crate::inference::{LlamaLlm, Llm};
use crate::llama_tokenizer::{END_TURN_TOKEN, ID_END_TOKEN, ID_START_TOKEN, LlamaTokenizerEnv};
use crate::token::TokenID;
use inference_types::{CategoryTopToken, InferenceEvent, StepCandidates, TokenWithProb};

/// Scaling constant and category name used to adjust token logits based on
/// embedding cosine similarity between the user message and VC message examples.
pub struct CategoryBias {
    /// The name of the VC message category (e.g. "referral").
    pub category_name: String,
    /// Pre-scaled margin: kappa * (max_pos_similarity - max_neg_similarity).
    pub weighted_margin: f32,
    /// Raw margin before kappa multiplication: max_pos_similarity - max_neg_similarity.
    /// Stored in CategoryTopToken.sim_score for per-category weight optimization.
    pub sim_score: f32,
}

#[derive(Clone)]
pub struct InferenceConfig {
    pub model_path: PathBuf,
    pub context_cache_dir: PathBuf,
    pub max_tokens: usize,
    pub top_candidate_count: usize,
}

struct InferenceEngineInner {
    backend: LlamaBackend,
    model: Arc<LlamaModel>,
    config: InferenceConfig,
}

/// The main inference engine. Cheap to clone — internally reference-counted.
#[derive(Clone)]
pub struct InferenceEngine(Arc<InferenceEngineInner>);

impl InferenceEngine {
    /// Load the model. Blocking — call from `tokio::task::spawn_blocking` or
    /// before the async runtime starts.
    pub fn new(config: InferenceConfig) -> anyhow::Result<Self> {
        let backend = LlamaBackend::init()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)?;
        let model = Arc::new(model);

        Ok(Self(Arc::new(InferenceEngineInner {
            backend,
            model,
            config,
        })))
    }

    /// Start generating tokens for `prompt` using the provided `grammar_flow`.
    /// `category_biases` contains per-category embedding similarity margins used
    /// to adjust token logits toward contextually relevant message categories.
    /// Returns immediately; generation runs on a blocking thread pool thread.
    /// Events are sent until `InferenceEvent::Done` or `InferenceEvent::Error`.
    pub async fn generate(
        &self,
        prompt: String,
        grammar_flow: GrammarFlow,
        category_biases: Vec<CategoryBias>,
    ) -> mpsc::Receiver<InferenceEvent> {
        let (tx, rx) = mpsc::channel(64);
        let inner = Arc::clone(&self.0);
        tokio::task::spawn_blocking(move || {
            run_generation_blocking(&inner, prompt, grammar_flow, category_biases, tx);
        });
        rx
    }
}

// ---------------------------------------------------------------------------
// Blocking generation — runs on the tokio blocking thread pool
// ---------------------------------------------------------------------------

fn run_generation_blocking(
    inner: &InferenceEngineInner,
    prompt: String,
    grammar_flow: GrammarFlow,
    category_biases: Vec<CategoryBias>,
    tx: mpsc::Sender<InferenceEvent>,
) {
    // Build per-request tokenizer (just wraps Arc<LlamaModel>, cheap)
    let tokenizer = Arc::new(LlamaTokenizerEnv::new(inner.model.clone()));

    // Precompute per-token logit bias map, category name/text pairs, and
    // per-category token ID lists for per-step "best token per category" lookup.
    let (logit_bias_map, category_info, category_token_ids) =
        build_logit_bias_map(&category_biases, &tokenizer, &inner.model);

    // Build a map of category_name → raw sim_score for populating CategoryTopToken.
    let category_sim_scores: HashMap<String, f32> = category_biases
        .iter()
        .map(|b| (b.category_name.clone(), b.sim_score))
        .collect();

    // Tokenize the system prompt and build the LLM context (cached to disk)
    let system_prompt = grammar_flow.get_system_prompt();
    let initial_tokens: Vec<_> = tokenizer.tokenize(&system_prompt);
    let mut llm = LlamaLlm::new(
        &inner.backend,
        inner.model.clone(),
        &initial_tokens,
        &inner.config.context_cache_dir,
    );

    // Build a fresh constraint for this request
    let tok_env: Arc<dyn TokenizerEnv + Sync + 'static> =
        Arc::new(LlamaTokenizerEnv::new(inner.model.clone()));
    let mut constraint = new_default_constraint(&grammar_flow, &tok_env);

    // Process the grammar prompt prefix (may return tokens the LLM should see first)
    let prefix_tokens = constraint.process_prompt(vec![]);
    let prefix_text = tokenizer.tokens_to_string(&prefix_tokens);
    llm.feed_tokens(&prefix_tokens);

    // Format the user turn including any prefix from the grammar
    let user_turn = format!(
        "{ID_START_TOKEN}user{ID_END_TOKEN}{prompt}{END_TURN_TOKEN}{ID_START_TOKEN}assistant{ID_END_TOKEN}{prefix_text}"
    );
    let user_tokens: Vec<_> = tokenizer.tokenize(&user_turn);
    llm.feed_tokens(&user_tokens);

    let mut full_output = String::new();

    for _ in 0..inner.config.max_tokens {
        let mut candidates = llm.get_canidates();

        // Apply embedding-based logit biases before any top_n sampling.
        candidates.apply_biases(&logit_bias_map);

        // Capture top-N before applying the grammar mask (using adjusted logits)
        let top_alternatives: Vec<TokenWithProb> = candidates
            .top_n(inner.config.top_candidate_count)
            .iter()
            .map(|c| TokenWithProb {
                text: tokenizer.tokens_to_string(&[c.token_id]),
                token_id: c.token_id,
                probability: c.probability,
                logit: c.logit,
                embedding_logit: c.embedding_logit,
            })
            .collect();

        // Compute and apply the grammar mask
        let mask = match constraint.compute_mask() {
            Ok(m) => m,
            Err(e) => {
                let _ = tx.blocking_send(InferenceEvent::Error {
                    message: e.to_string(),
                });
                return;
            }
        };
        let sample_mask = match &mask.sample_mask {
            Some(m) => m,
            None => {
                let _ = tx.blocking_send(InferenceEvent::Done {
                    full_text: full_output,
                });
                return;
            }
        };
        candidates.constrain(sample_mask);

        // Top-N after mask (adjusted logits)
        let top_constrained: Vec<TokenWithProb> = candidates
            .top_n(inner.config.top_candidate_count)
            .iter()
            .map(|c| TokenWithProb {
                text: tokenizer.tokens_to_string(&[c.token_id]),
                token_id: c.token_id,
                probability: c.probability,
                logit: c.logit,
                embedding_logit: c.embedding_logit,
            })
            .collect();

        let chosen = match top_constrained.first() {
            Some(c) => c.clone(),
            None => {
                let _ = tx.blocking_send(InferenceEvent::Done {
                    full_text: full_output,
                });
                return;
            }
        };
        let chosen_token_id = chosen.token_id;

        // For every category find the best-scoring prefix token using the full
        // pre-mask candidate list (O(1) per token via the HashMap index).
        // This shows all categories, not just those whose tokens happen to be in top-N.
        let category_top_tokens: Vec<CategoryTopToken> = category_info
            .iter()
            .zip(category_token_ids.iter())
            .filter_map(|((cat_name, _), token_ids)| {
                token_ids
                    .iter()
                    .filter_map(|&tid| candidates.get_by_id(tid))
                    .max_by(|a, b| {
                        (a.logit + a.embedding_logit)
                            .partial_cmp(&(b.logit + b.embedding_logit))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|best| CategoryTopToken {
                        category_name: cat_name.clone(),
                        best_token: TokenWithProb {
                            text: tokenizer.tokens_to_string(&[best.token_id]),
                            token_id: best.token_id,
                            probability: best.probability,
                            logit: best.logit,
                            embedding_logit: best.embedding_logit,
                        },
                        sim_score: category_sim_scores
                            .get(cat_name.as_str())
                            .copied()
                            .unwrap_or(0.0),
                    })
            })
            .collect();

        if tx
            .blocking_send(InferenceEvent::Token(StepCandidates {
                chosen,
                top_alternatives,
                top_constrained,
                category_top_tokens,
            }))
            .is_err()
        {
            return; // receiver dropped (client disconnected)
        }

        // Commit chosen token to the constraint
        let commit = match constraint.commit_token(Some(chosen_token_id)) {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.blocking_send(InferenceEvent::Error {
                    message: e.to_string(),
                });
                return;
            }
        };
        let ff_tokens = commit.ff_tokens;

        // Emit Token events for grammar-forced fast-forward tokens (ff_tokens[0]
        // is the chosen token already sent above; start from index 1).
        // These are not sampled so they carry probability=1.0 and no alternatives.
        let mut generation_done = false;
        for &ff_id in ff_tokens.iter().skip(1) {
            let ff_text = tokenizer.tokens_to_string(&[ff_id]);
            let ff_token = TokenWithProb {
                text: ff_text,
                token_id: ff_id,
                probability: 1.0,
                logit: 0.0,
                embedding_logit: 0.0,
            };
            if tx
                .blocking_send(InferenceEvent::Token(StepCandidates {
                    chosen: ff_token.clone(),
                    top_alternatives: vec![],
                    top_constrained: vec![ff_token],
                    category_top_tokens: vec![],
                }))
                .is_err()
            {
                return;
            }
            match constraint.commit_token(Some(ff_id)) {
                Ok(r) if r.stop => {
                    generation_done = true;
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    let _ = tx.blocking_send(InferenceEvent::Error {
                        message: e.to_string(),
                    });
                    return;
                }
            }
        }

        // Feed all committed tokens to the LLM KV cache
        if ff_tokens.is_empty() {
            llm.feed_tokens(&[chosen_token_id]);
            full_output += &tokenizer.tokens_to_string(&[chosen_token_id]);
        } else {
            llm.feed_tokens(&ff_tokens);
            full_output += &tokenizer.tokens_to_string(&ff_tokens);
        }

        if generation_done {
            let _ = tx.blocking_send(InferenceEvent::Done {
                full_text: full_output,
            });
            return;
        }
    }

    let _ = tx.blocking_send(InferenceEvent::Done {
        full_text: full_output,
    });
}

// ---------------------------------------------------------------------------
// Logit bias precomputation
// ---------------------------------------------------------------------------

/// Build a map from token ID → logit adjustment w(v).
///
/// For each category c with `weighted_margin` = kappa * s_c, the full decoded
/// category name text is built with a leading space (as it appears mid-generation).
///
/// For each vocab token v:
///   text_v = decoded text of token v
///   C_v = { c | text_v is a non-empty prefix of the full category name text }
///   w(v) = (1 / |C_v|) * sum_{c in C_v} weighted_margin_c
///
/// Matching against the full name string (not individual tokenization units) means
/// every prefix token of a category name receives the bias. For example, if a
/// category is "Dosing Administration", then " D", " Do", " Dos", " Dosi", " Dosing"
/// all match — regardless of how the tokenizer happened to split "Dosing".
///
/// Tokens with no matching categories are omitted from the map.
/// Returns `(bias_map, category_info, category_token_ids)`:
/// - `bias_map`: token_id → w(v) logit adjustment
/// - `category_info`: Vec<(category_name, full_text_with_leading_space)>
/// - `category_token_ids`: parallel to `category_info`; for each category, the
///   token IDs whose decoded text is a non-empty prefix of that category's full text.
///   Used for O(1)-per-token per-step "best token per category" lookup.
fn build_logit_bias_map(
    biases: &[CategoryBias],
    tokenizer: &LlamaTokenizerEnv,
    model: &LlamaModel,
) -> (HashMap<TokenID, f32>, Vec<(String, String)>, Vec<Vec<TokenID>>) {
    if biases.is_empty() {
        return (HashMap::new(), vec![], vec![]);
    }

    // (category_name, full_text_with_leading_space)
    let category_info: Vec<(String, String)> = biases
        .iter()
        .map(|b| (b.category_name.clone(), format!(" {}", b.category_name)))
        .collect();

    // (weighted_margin, full_text) — parallel to category_info
    let category_texts: Vec<(f32, &str)> = biases
        .iter()
        .zip(category_info.iter())
        .map(|(b, (_, text))| (b.weighted_margin, text.as_str()))
        .collect();

    let mut bias_map: HashMap<TokenID, f32> = HashMap::new();
    let mut category_token_ids: Vec<Vec<TokenID>> = vec![vec![]; biases.len()];

    for (token, _) in model.tokens(Special::Tokenize) {
        let vid = token.0 as TokenID;
        let text_v = tokenizer.tokens_to_string(&[vid]);
        if text_v.is_empty() {
            continue;
        }

        let mut sum = 0.0f32;
        let mut count = 0usize;

        for (i, (weighted_margin, category_text)) in category_texts.iter().enumerate() {
            if category_text.starts_with(text_v.as_str()) {
                sum += weighted_margin;
                count += 1;
                category_token_ids[i].push(vid);
            }
        }

        if count > 0 {
            bias_map.insert(vid, sum / count as f32);
        }
    }

    (bias_map, category_info, category_token_ids)
}
