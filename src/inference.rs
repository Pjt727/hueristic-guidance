// manages all the context

use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::LlamaModel,
    token::{data::LlamaTokenData, LlamaToken},
};
use std::{
    num::NonZero,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::token::Canidate;

use super::token::{Canidates, TokenID};

const CONTEXT_CACHE_DIR: &str = "context_cache";

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
    batch_size: u32,
    seq_id: i32,
    current_token_position: i32,
    ctx: LlamaContext<'static>,
}

impl LlamaLlm {
    /// if the initial tokens has been saved it will try to load from there
    pub fn new(backend: &LlamaBackend, model: Arc<LlamaModel>, initial_tokens: &[TokenID]) -> Self {
        let batch_size = 512_u32;
        let context_size = 2048;
        let seq_id = 0;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZero::new(context_size).unwrap())) // Context size
            .with_n_batch(batch_size);

        // Leak the Arc to get a 'static reference - this avoids lifetime annotation issues
        // The model will live for the entire program duration
        let model_ref: &'static Arc<LlamaModel> = Box::leak(Box::new(model));
        let mut ctx = model_ref.new_context(backend, ctx_params).unwrap();

        // get from the initial cache if there is one avaiable so we
        // do not need to recompute the initial expensive prompt
        let mut possible_context_cache = PathBuf::from(CONTEXT_CACHE_DIR);
        possible_context_cache.push(get_file_hash(initial_tokens));
        let load_from_cache = possible_context_cache.exists();
        let llama_tokens = to_llama_tokens(initial_tokens);
        // loading from the cache only sets the KV cache, which means we still need to feed the
        // tokens
        let mut current_token_position = 0;
        if load_from_cache {
            let cached_tokens = ctx
                .load_session_file(&possible_context_cache, context_size as usize)
                .unwrap();
            assert_eq!(
                cached_tokens, llama_tokens,
                "llama ctx was not saved correctly"
            );
            // llama cpp expects this in sequence
            current_token_position = initial_tokens.len() as i32;
        }
        let mut llm = Self {
            model: model_ref,
            batch_size,
            current_token_position,
            seq_id,
            ctx,
        };

        if !load_from_cache {
            llm.feed_tokens(initial_tokens);
            llm.ctx
                .save_session_file(&possible_context_cache, &llama_tokens)
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
        let mut batch = LlamaBatch::new(self.batch_size as usize, 1);

        for prefix_token in &llama_tokens[..llama_tokens.len() - 1] {
            batch
                .add(
                    *prefix_token,
                    self.current_token_position,
                    &[self.seq_id],
                    false,
                )
                .unwrap();
            self.current_token_position += 1;
        }
        // add the last token with fix one
        let last_token = llama_tokens.last().unwrap();
        batch
            .add(
                *last_token,
                self.current_token_position,
                &[self.seq_id],
                true,
            )
            .unwrap();
        self.current_token_position += 1;

        self.ctx.decode(&mut batch).unwrap();
    }

    fn get_canidates(&mut self) -> Canidates {
        let canidates: Vec<_> = self
            .ctx
            .candidates()
            .map(|c| Canidate {
                token_id: c.id().0 as TokenID,
                probability: c.p(),
                logit: c.logit(),
            })
            .collect();
        Canidates::new(canidates)
    }
}
