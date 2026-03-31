// manages all the context

use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::LlamaModel,
    token::LlamaToken,
};
use std::{num::NonZero, path::Path, sync::Arc};

use crate::token::Canidate;

use super::token::{Canidates, TokenID};

pub trait Llm {
    fn get_canidates(&mut self) -> Canidates;
    fn feed_tokens(&mut self, tokens: &[TokenID]);
}

fn to_llama_tokens(tokens: &[TokenID]) -> Vec<LlamaToken> {
    tokens.iter().map(|t| LlamaToken(*t as i32)).collect()
}

fn get_file_hash(tokens: &[TokenID]) -> String {
    let mut hash: u64 = 5381;

    for &val in tokens {
        hash = hash.wrapping_mul(33).wrapping_add(val as u64);
        hash ^= (hash << 5).wrapping_add(val as u64);
    }

    format!("{:x}", hash)
}

pub struct LlamaLlm {
    model: &'static Arc<LlamaModel>,
    seq_id: i32,
    current_token_position: i32,
    ctx: LlamaContext<'static>,
    batch: LlamaBatch<'static>,
    batch_size: usize,
}

impl LlamaLlm {
    /// If the initial tokens have been saved, load from the cache to skip
    /// the expensive re-encoding of the system prompt.
    pub fn new(
        backend: &LlamaBackend,
        model: Arc<LlamaModel>,
        initial_tokens: &[TokenID],
        context_cache_dir: &Path,
    ) -> Self {
        let batch_size = 2048_u32; // handles large system prompt (~1400-1600 tokens)
        let context_size = 8192; // supports extended conversations (5+ exchanges)
        let seq_id = 0;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZero::new(context_size).unwrap()))
            .with_n_batch(batch_size);

        // Leak the Arc to get a 'static reference — the model lives for the whole process
        let model_ref: &'static Arc<LlamaModel> = Box::leak(Box::new(model));
        let mut ctx = model_ref.new_context(backend, ctx_params).unwrap();

        // Load from the KV cache if available to avoid recomputing the prompt
        let mut cache_file = context_cache_dir.to_path_buf();
        cache_file.push(get_file_hash(initial_tokens));
        let load_from_cache = cache_file.exists();
        let llama_tokens = to_llama_tokens(initial_tokens);
        let mut current_token_position = 0;

        if load_from_cache {
            println!("Getting model ctx cache: {:?}", cache_file);
            let cached_tokens = ctx
                .load_session_file(&cache_file, context_size as usize)
                .unwrap();
            assert_eq!(
                cached_tokens, llama_tokens,
                "llama ctx was not saved correctly"
            );
            current_token_position = initial_tokens.len() as i32;
        }

        let batch = LlamaBatch::new(batch_size as usize, 1);
        let mut llm = Self {
            model: model_ref,
            current_token_position,
            seq_id,
            batch,
            batch_size: batch_size as usize,
            ctx,
        };

        if !load_from_cache {
            llm.feed_tokens(initial_tokens);
            llm.ctx
                .save_session_file(&cache_file, &llama_tokens)
                .unwrap();
        }

        llm
    }
}

impl Llm for LlamaLlm {
    fn feed_tokens(&mut self, tokens: &[TokenID]) {
        if tokens.is_empty() {
            return;
        }
        let llama_tokens = to_llama_tokens(tokens);

        // Feed in chunks so the batch never exceeds its allocated capacity.
        // Only the very last token of the very last chunk needs logits=true
        // (subsequent `get_candidates` reads from that position).
        let chunks: Vec<_> = llama_tokens.chunks(self.batch_size).collect();
        let last_chunk_idx = chunks.len() - 1;

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            self.batch.clear();
            let is_last_chunk = chunk_idx == last_chunk_idx;
            let last_token_idx = chunk.len() - 1;

            for (i, token) in chunk.iter().enumerate() {
                let logits = is_last_chunk && i == last_token_idx;
                self.batch
                    .add(*token, self.current_token_position, &[self.seq_id], logits)
                    .unwrap();
                self.current_token_position += 1;
            }

            self.ctx.decode(&mut self.batch).unwrap();
        }
    }

    fn get_canidates(&mut self) -> Canidates {
        let canidates: Vec<_> = self
            .ctx
            .candidates()
            .map(|c| Canidate {
                token_id: c.id().0 as TokenID,
                probability: c.p(),
                logit: c.logit(),
                embedding_logit: 0.0,
            })
            .collect();
        Canidates::new(canidates)
    }
}
