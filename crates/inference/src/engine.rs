use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{LlamaModel, params::LlamaModelParams},
};
use llguidance::toktrie::TokenizerEnv;
use tokio::sync::mpsc;

use crate::constraints::{new_constraint, new_parser_factory};
use crate::embedding_bias::ResolvedCategoryBias;
use crate::grammar::GrammarFlow;
use crate::inference::{LlamaLlm, Llm};
use crate::llama_tokenizer::LlamaTokenizerEnv;
use crate::token::{Canidates, TokenID};
use inference_types::{
    BiasRegime, CategoryDistributionMetrics, CategoryTopToken, DecisionDiagnostics, InferenceEvent,
    OverrideOutcome, StepCandidates, TokenWithProb,
};
use llguidance::ParserFactory;

/// Scaling constant and category name used to adjust token logits based on
/// embedding cosine similarity between the user message and VC message examples.
#[derive(Debug, Clone)]
pub struct CategoryBias {
    /// The name of the VC message category (e.g. "referral").
    pub category_name: String,
    /// Pre-scaled margin: kappa * (max_pos_similarity - max_neg_similarity).
    pub weighted_margin: f32,
    /// Raw margin before kappa multiplication: max_pos_similarity - max_neg_similarity.
    /// Stored in CategoryTopToken.sim_score for per-category weight optimization.
    pub sim_score: f32,
    pub positive_similarity: f32,
    pub negative_similarity: f32,
    pub regime: BiasRegime,
    pub force_target: bool,
}

impl From<ResolvedCategoryBias> for CategoryBias {
    fn from(value: ResolvedCategoryBias) -> Self {
        Self {
            category_name: value.category_name,
            weighted_margin: value.weighted_margin,
            sim_score: value.sim_score,
            positive_similarity: value.positive_similarity,
            negative_similarity: value.negative_similarity,
            regime: value.regime,
            force_target: value.force_target,
        }
    }
}

#[derive(Clone)]
pub struct InferenceConfig {
    pub model_path: PathBuf,
    pub context_cache_dir: PathBuf,
    pub max_tokens: usize,
    pub top_candidate_count: usize,
}

struct HotSystemKv {
    hash: String,
    llm: LlamaLlm,
}

// SAFETY: `LlamaContext` is not `Send`, but we only ever touch `llm` while holding
// `InferenceEngineInner::hot_system_kv`. That mutex serializes all access across
// spawn_blocking workers, so no concurrent use of the context occurs.
unsafe impl Send for HotSystemKv {}

struct InferenceEngineInner {
    backend: LlamaBackend,
    model: Arc<LlamaModel>,
    config: InferenceConfig,
    /// Built once at engine load — vocab trie construction is expensive (~248k tokens).
    tokenizer: Arc<LlamaTokenizerEnv>,
    /// Shared llguidance factory; only the per-request grammar parser is rebuilt.
    parser_factory: ParserFactory,
    /// Last system-prompt KV kept in memory. Evicted when the system prompt hash changes.
    hot_system_kv: Mutex<Option<HotSystemKv>>,
}

/// Restores the system-prompt SeqState snapshot when dropped so the next
/// request can reuse in-memory attention without a disk reload.
struct RestoreOnDrop<'a> {
    llm: &'a mut LlamaLlm,
}

impl Drop for RestoreOnDrop<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.llm.restore_system_prompt_state() {
            tracing::warn!(error = %e, "failed to restore system prompt state on drop");
        }
    }
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

        tracing::info!("building tokenizer / toktrie (one-time)…");
        let tokenizer = Arc::new(LlamaTokenizerEnv::new(model.clone())?);
        let tok_env: Arc<dyn TokenizerEnv + Sync + 'static> = tokenizer.clone();
        let parser_factory = new_parser_factory(&tok_env);

        Ok(Self(Arc::new(InferenceEngineInner {
            backend,
            model,
            config,
            tokenizer,
            parser_factory,
            hot_system_kv: Mutex::new(None),
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
    let started = std::time::Instant::now();
    let mut category_latency_ms: Option<u64> = None;
    let send_done = |tx: &mpsc::Sender<InferenceEvent>,
                     full_text: String,
                     category_latency_ms: Option<u64>| {
        let _ = tx.blocking_send(InferenceEvent::Done {
            full_text,
            latency_ms: started.elapsed().as_millis() as u64,
            category_latency_ms,
        });
    };

    let tokenizer = Arc::clone(&inner.tokenizer);

    // Precompute per-token logit bias map, category name/text pairs, and
    // per-category token ID lists for per-step "best token per category" lookup.
    let (logit_bias_map, category_info, category_token_ids) =
        build_logit_bias_map(&category_biases, &tokenizer, &inner.model);

    let category_bias_details: HashMap<String, CategoryBias> = category_biases
        .iter()
        .cloned()
        .map(|bias| (bias.category_name.clone(), bias))
        .collect();
    let force_target = category_biases
        .iter()
        .find(|bias| bias.force_target)
        .map(|bias| bias.category_name.clone());

    // Tokenize the system prompt and reuse in-memory KV when the hash matches.
    let system_prompt = tokenizer
        .chat_format
        .wrap_system(grammar_flow.get_system_prompt());
    let initial_tokens: Vec<_> = tokenizer.tokenize(&system_prompt);
    let system_hash = LlamaLlm::system_token_hash(&initial_tokens);

    let mut hot_guard = inner
        .hot_system_kv
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let reuse = hot_guard
        .as_ref()
        .is_some_and(|hot| hot.hash == system_hash);
    if reuse {
        if let Err(e) = hot_guard
            .as_mut()
            .expect("checked above")
            .llm
            .restore_system_prompt_state()
        {
            tracing::warn!(error = %e, %system_hash, "hot KV restore failed — rebuilding");
            *hot_guard = None;
        }
    }
    if hot_guard
        .as_ref()
        .is_none_or(|hot| hot.hash != system_hash)
    {
        tracing::info!(%system_hash, "loading system prompt KV (disk or compute)");
        let llm = match LlamaLlm::new(
            &inner.backend,
            inner.model.clone(),
            &initial_tokens,
            &inner.config.context_cache_dir,
        ) {
            Ok(llm) => llm,
            Err(e) => {
                let _ = tx.blocking_send(InferenceEvent::Error {
                    message: e.to_string(),
                });
                return;
            }
        };
        *hot_guard = Some(HotSystemKv {
            hash: system_hash,
            llm,
        });
    }
    let llm = &mut hot_guard.as_mut().expect("hot KV just ensured").llm;
    let llm = RestoreOnDrop { llm };

    // Fresh grammar parser per request; factory + tokenizer are reused.
    let mut constraint = new_constraint(&inner.parser_factory, &grammar_flow);

    // Process the grammar prompt prefix (may return tokens the LLM should see first)
    let prefix_tokens = constraint.process_prompt(vec![]);
    let prefix_text = tokenizer.tokens_to_string(&prefix_tokens);
    let force_tokens = force_target
        .as_ref()
        .map(|target| remaining_force_tokens(target, &prefix_text, &tokenizer))
        .unwrap_or_default();
    let mut force_token_index = 0usize;
    if let Err(e) = llm.llm.feed_tokens(&prefix_tokens) {
        let _ = tx.blocking_send(InferenceEvent::Error {
            message: e.to_string(),
        });
        return;
    }

    // Format the user turn including any prefix from the grammar
    let user_turn = tokenizer
        .chat_format
        .wrap_user_turn(&prompt, &prefix_text);
    let user_tokens: Vec<_> = tokenizer.tokenize(&user_turn);
    if let Err(e) = llm.llm.feed_tokens(&user_tokens) {
        let _ = tx.blocking_send(InferenceEvent::Error {
            message: e.to_string(),
        });
        return;
    }

    // Accumulate token IDs and decode once for Done. Per-token UTF-8 decode
    // fails on Qwen byte-fallback pieces and used to poison full_text with
    // the literal "Invalid utf-8", breaking bulk category matching.
    let mut output_tokens: Vec<TokenID> = Vec::new();
    let full_text = |tokens: &[TokenID]| tokenizer.tokens_to_string(tokens);

    for _ in 0..inner.config.max_tokens {
        let mut raw_candidates = llm.llm.get_canidates();
        let mut candidates = raw_candidates.clone();

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
                send_done(&tx, full_text(&output_tokens), category_latency_ms);
                return;
            }
        };
        raw_candidates.constrain(sample_mask);
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

        let normal_chosen = match top_constrained.first() {
            Some(c) => c.clone(),
            None => {
                send_done(&tx, full_text(&output_tokens), category_latency_ms);
                return;
            }
        };
        let forced_candidate = forced_token_id(&force_tokens, force_token_index, &candidates)
            .and_then(|token_id| candidates.get_with_probability(token_id))
            .map(|candidate| TokenWithProb {
                text: tokenizer.tokens_to_string(&[candidate.token_id]),
                token_id: candidate.token_id,
                probability: candidate.probability,
                logit: candidate.logit,
                embedding_logit: candidate.embedding_logit,
            });
        let force_applied = forced_candidate.is_some();
        let chosen = forced_candidate.unwrap_or(normal_chosen);
        let chosen_token_id = chosen.token_id;

        // For every category find the best-scoring prefix token using the full
        // pre-mask candidate list (O(1) per token via the HashMap index).
        // This shows all categories, not just those whose tokens happen to be in top-N.
        let mut category_top_tokens = build_category_top_tokens(
            &category_info,
            &category_token_ids,
            &raw_candidates,
            &candidates,
            &category_bias_details,
            &tokenizer,
        );
        populate_category_probabilities(&mut category_top_tokens);
        if category_latency_ms.is_none() && !category_top_tokens.is_empty() {
            category_latency_ms = Some(started.elapsed().as_millis() as u64);
        }
        let decision_diagnostics = if category_top_tokens.is_empty() {
            None
        } else {
            let pre_bias = distribution_metrics(
                category_top_tokens
                    .iter()
                    .map(|category| (&category.category_name, category.pre_bias_logit)),
            );
            let post_bias = distribution_metrics(
                category_top_tokens
                    .iter()
                    .map(|category| (&category.category_name, category.post_bias_logit)),
            );
            let bias_regime = if force_target.is_some() {
                BiasRegime::Force
            } else if category_biases.is_empty() {
                BiasRegime::None
            } else {
                BiasRegime::Soft
            };
            let override_outcome = if force_applied {
                if force_target.as_ref() != pre_bias.winner_category.as_ref() {
                    OverrideOutcome::ForcedEmbedding
                } else {
                    OverrideOutcome::None
                }
            } else if pre_bias.winner_category != post_bias.winner_category {
                OverrideOutcome::SoftBias
            } else {
                OverrideOutcome::None
            };
            Some(DecisionDiagnostics {
                pre_bias,
                post_bias,
                bias_regime,
                force_target: force_target.clone(),
                override_outcome,
            })
        };

        if tx
            .blocking_send(InferenceEvent::Token(StepCandidates {
                chosen,
                top_alternatives,
                top_constrained,
                category_top_tokens,
                decision_diagnostics,
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
        if ff_tokens.is_empty() {
            if force_tokens.get(force_token_index) == Some(&chosen_token_id) {
                force_token_index += 1;
            }
        } else {
            for token_id in &ff_tokens {
                if force_tokens.get(force_token_index) == Some(token_id) {
                    force_token_index += 1;
                }
            }
        }

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
                    decision_diagnostics: None,
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
            if let Err(e) = llm.llm.feed_tokens(&[chosen_token_id]) {
                let _ = tx.blocking_send(InferenceEvent::Error {
                    message: e.to_string(),
                });
                return;
            }
            output_tokens.push(chosen_token_id);
        } else {
            if let Err(e) = llm.llm.feed_tokens(&ff_tokens) {
                let _ = tx.blocking_send(InferenceEvent::Error {
                    message: e.to_string(),
                });
                return;
            }
            output_tokens.extend_from_slice(&ff_tokens);
        }

        if generation_done {
            send_done(&tx, full_text(&output_tokens), category_latency_ms);
            return;
        }
    }

    send_done(&tx, full_text(&output_tokens), category_latency_ms);
}

// ---------------------------------------------------------------------------
// Logit bias precomputation
// ---------------------------------------------------------------------------

fn build_category_top_tokens(
    category_info: &[(String, String)],
    category_token_ids: &[Vec<TokenID>],
    raw_candidates: &Canidates,
    adjusted_candidates: &Canidates,
    bias_details: &HashMap<String, CategoryBias>,
    tokenizer: &LlamaTokenizerEnv,
) -> Vec<CategoryTopToken> {
    category_info
        .iter()
        .zip(category_token_ids)
        .filter_map(|((category_name, _), token_ids)| {
            let raw_best = token_ids
                .iter()
                .filter_map(|token_id| raw_candidates.get_by_id(*token_id))
                .max_by(|a, b| a.logit.total_cmp(&b.logit))?;
            let adjusted_best = token_ids
                .iter()
                .filter_map(|token_id| adjusted_candidates.get_by_id(*token_id))
                .max_by(|a, b| {
                    (a.logit + a.embedding_logit).total_cmp(&(b.logit + b.embedding_logit))
                })?;
            let details = bias_details.get(category_name);
            Some(CategoryTopToken {
                category_name: category_name.clone(),
                best_token: TokenWithProb {
                    text: tokenizer.tokens_to_string(&[adjusted_best.token_id]),
                    token_id: adjusted_best.token_id,
                    probability: adjusted_best.probability,
                    logit: adjusted_best.logit,
                    embedding_logit: adjusted_best.embedding_logit,
                },
                sim_score: details.map(|bias| bias.sim_score).unwrap_or(0.0),
                positive_similarity: details.map(|bias| bias.positive_similarity).unwrap_or(0.0),
                negative_similarity: details.map(|bias| bias.negative_similarity).unwrap_or(0.0),
                pre_bias_logit: raw_best.logit,
                post_bias_logit: adjusted_best.logit + adjusted_best.embedding_logit,
                pre_bias_probability: 0.0,
                post_bias_probability: 0.0,
            })
        })
        .collect()
}

fn populate_category_probabilities(categories: &mut [CategoryTopToken]) {
    if categories.is_empty() {
        return;
    }
    let raw_max = categories
        .iter()
        .map(|category| category.pre_bias_logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let post_max = categories
        .iter()
        .map(|category| category.post_bias_logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let raw_sum: f32 = categories
        .iter()
        .map(|category| (category.pre_bias_logit - raw_max).exp())
        .sum();
    let post_sum: f32 = categories
        .iter()
        .map(|category| (category.post_bias_logit - post_max).exp())
        .sum();
    for category in categories {
        category.pre_bias_probability = (category.pre_bias_logit - raw_max).exp() / raw_sum;
        category.post_bias_probability = (category.post_bias_logit - post_max).exp() / post_sum;
    }
}

fn distribution_metrics<'a>(
    scores: impl Iterator<Item = (&'a String, f32)>,
) -> CategoryDistributionMetrics {
    let mut scores: Vec<(&String, f32)> = scores.collect();
    if scores.is_empty() {
        return CategoryDistributionMetrics::default();
    }
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    let max_score = scores[0].1;
    let exps: Vec<f32> = scores
        .iter()
        .map(|(_, score)| (*score - max_score).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    let entropy = exps
        .iter()
        .map(|value| value / sum)
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum::<f32>();
    let normalized_entropy = if scores.len() > 1 {
        entropy / (scores.len() as f32).ln()
    } else {
        0.0
    };
    CategoryDistributionMetrics {
        winner_category: Some(scores[0].0.clone()),
        top_two_margin: scores
            .get(1)
            .map(|runner_up| scores[0].1 - runner_up.1)
            .unwrap_or(0.0),
        normalized_entropy,
    }
}

fn forced_token_id(
    force_tokens: &[TokenID],
    force_token_index: usize,
    candidates: &Canidates,
) -> Option<TokenID> {
    force_tokens
        .get(force_token_index)
        .copied()
        .filter(|token_id| candidates.get_by_id(*token_id).is_some())
}

fn remaining_force_tokens(
    target: &str,
    prefix_text: &str,
    tokenizer: &LlamaTokenizerEnv,
) -> Vec<TokenID> {
    tokenizer.tokenize(&remaining_force_text(target, prefix_text))
}

fn remaining_force_text(target: &str, prefix_text: &str) -> String {
    let force_text = format!(" {target}");
    let consumed = force_text
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(force_text.len()))
        .filter(|index| prefix_text.ends_with(&force_text[..*index]))
        .max()
        .unwrap_or(0);
    force_text[consumed..].to_string()
}

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
) -> (
    HashMap<TokenID, f32>,
    Vec<(String, String)>,
    Vec<Vec<TokenID>>,
) {
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

    for (token, _) in model.tokens(true) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::Canidate;

    #[test]
    fn force_selects_next_allowed_category_token() {
        let candidates = Canidates::new(vec![
            Canidate {
                token_id: 10,
                probability: 0.8,
                logit: 4.0,
                embedding_logit: 0.0,
            },
            Canidate {
                token_id: 20,
                probability: 0.2,
                logit: 1.0,
                embedding_logit: 0.0,
            },
        ]);
        assert_eq!(forced_token_id(&[20, 30], 0, &candidates), Some(20));
        assert_eq!(forced_token_id(&[20, 30], 1, &candidates), None);
    }

    #[test]
    fn distribution_metrics_report_margin_and_normalized_entropy() {
        let names = ["a".to_string(), "b".to_string()];
        let metrics = distribution_metrics([(&names[0], 2.0), (&names[1], 1.0)].into_iter());
        assert_eq!(metrics.winner_category.as_deref(), Some("a"));
        assert!((metrics.top_two_margin - 1.0).abs() < f32::EPSILON);
        assert!(metrics.normalized_entropy > 0.0);
        assert!(metrics.normalized_entropy < 1.0);
    }

    #[test]
    fn force_text_accounts_for_grammar_fast_forward_prefix() {
        assert_eq!(remaining_force_text("Dosing", "Category: "), "Dosing");
        assert_eq!(remaining_force_text("Dosing", "Category: Dos"), "ing");
        assert_eq!(remaining_force_text("Dosing", "Category:"), " Dosing");
    }

    /// Smoke-test hot SeqState restore across two generations with the same system prompt.
    /// Requires MODEL_PATH (and enough Metal/CPU memory). Run with:
    /// `MODEL_PATH=... cargo test -p inference hot_system_kv_reuse -- --ignored --nocapture`
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires a local GGUF via MODEL_PATH"]
    async fn hot_system_kv_reuse_two_generations() {
        use crate::grammar::VCmessage;
        use std::path::PathBuf;

        let model_path = std::env::var("MODEL_PATH").expect("MODEL_PATH");
        let cache_dir = PathBuf::from(
            std::env::var("CONTEXT_CACHE_DIR").unwrap_or_else(|_| "context_cache".into()),
        );
        let _ = std::fs::create_dir_all(&cache_dir);

        let engine = InferenceEngine::new(InferenceConfig {
            model_path: PathBuf::from(model_path),
            context_cache_dir: cache_dir,
            max_tokens: 32,
            top_candidate_count: 5,
        })
        .expect("load engine");

        let messages = vec![
            VCmessage {
                category: "Safety".into(),
                kind: String::new(),
                description: "Safety information".into(),
                mlr_message: "Here is safety info.".into(),
                message: "Here is safety info.".into(),
            },
            VCmessage {
                category: "Dosing".into(),
                kind: String::new(),
                description: "Dosing information".into(),
                mlr_message: "Here is dosing info.".into(),
                message: "Here is dosing info.".into(),
            },
        ];
        let grammar = GrammarFlow::new("TestBrand", &messages).expect("grammar");

        for (i, prompt) in ["tell me about safety", "how do I dose this"].iter().enumerate() {
            let mut rx = engine
                .generate((*prompt).to_string(), grammar.clone(), vec![])
                .await;
            let mut done = false;
            while let Some(event) = rx.recv().await {
                match event {
                    InferenceEvent::Done { full_text, .. } => {
                        println!("gen {i} ok: {}", full_text.chars().take(80).collect::<String>());
                        done = true;
                        break;
                    }
                    InferenceEvent::Error { message } => {
                        panic!("gen {i} failed: {message}");
                    }
                    InferenceEvent::Token(_) => {}
                }
            }
            assert!(done, "gen {i} produced no Done event");
        }
    }
}
