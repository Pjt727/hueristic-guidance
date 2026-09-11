// manages all the context

use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::LlamaModel,
    token::LlamaToken,
    LlamaStateSeqFlags, SeqState,
};
use std::{num::NonZero, path::Path, sync::Arc};

use crate::token::Canidate;

use super::token::{Canidates, TokenID};

pub trait Llm {
    fn get_canidates(&mut self) -> Canidates;
    fn feed_tokens(&mut self, tokens: &[TokenID]) -> anyhow::Result<()>;
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
    #[allow(dead_code)]
    model: &'static Arc<LlamaModel>,
    seq_id: i32,
    current_token_position: i32,
    /// Token count for the cached system prompt prefix. Generation appends after this.
    system_token_len: i32,
    /// Full sequence state captured at the system-prompt boundary. Used to rewind
    /// hybrid/SSM models (e.g. Qwen3.5) where `clear_kv_cache_seq` cannot roll back.
    system_seq_state: SeqState,
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
    ) -> anyhow::Result<Self> {
        let batch_size = 2048_u32; // handles large system prompt (~1400-1600 tokens)
        let context_size = 8192; // supports extended conversations (5+ exchanges)
        let seq_id = 0;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZero::new(context_size).unwrap()))
            .with_n_batch(batch_size);
        let system_token_len = initial_tokens.len() as i32;

        // Leak the Arc to get a 'static reference — the model lives for the whole process
        let model_ref: &'static Arc<LlamaModel> = Box::leak(Box::new(model));
        let mut ctx = model_ref.new_context(backend, ctx_params)?;
        let mut batch = LlamaBatch::new(batch_size as usize, 1);
        let mut current_token_position = 0i32;

        // Load from the KV cache if available to avoid recomputing the prompt
        let mut cache_file = context_cache_dir.to_path_buf();
        cache_file.push(get_file_hash(initial_tokens));
        let load_from_cache = cache_file.exists();
        let llama_tokens = to_llama_tokens(initial_tokens);

        if load_from_cache {
            println!("Getting model ctx cache: {:?}", cache_file);
            let cached_tokens = ctx.load_session_file(&cache_file, context_size as usize)?;
            anyhow::ensure!(
                cached_tokens == llama_tokens,
                "llama ctx was not saved correctly"
            );
            current_token_position = system_token_len;
        } else {
            feed_tokens_raw(
                &mut ctx,
                &mut batch,
                seq_id,
                &mut current_token_position,
                batch_size as usize,
                initial_tokens,
            )?;
            ctx.save_session_file(&cache_file, &llama_tokens)
                .map_err(|e| anyhow::anyhow!("failed to save session file: {e}"))?;
        }

        let system_seq_state = ctx
            .state_seq_get(seq_id, LlamaStateSeqFlags::empty())
            .map_err(|e| anyhow::anyhow!("failed to snapshot system prompt state: {e}"))?;

        Ok(Self {
            model: model_ref,
            current_token_position,
            system_token_len,
            system_seq_state,
            seq_id,
            batch,
            batch_size: batch_size as usize,
            ctx,
        })
    }

    pub fn system_token_hash(tokens: &[TokenID]) -> String {
        get_file_hash(tokens)
    }

    pub fn system_token_len(&self) -> i32 {
        self.system_token_len
    }

    /// Restore the full sequence state captured at the system-prompt boundary.
    /// Required for Qwen3.5 / hybrid-recurrent models where `clear_kv_cache_seq`
    /// cannot roll back positions.
    pub fn restore_system_prompt_state(&mut self) -> anyhow::Result<()> {
        self.ctx
            .state_seq_set(&self.system_seq_state, self.seq_id)
            .map_err(|e| anyhow::anyhow!("failed to restore system prompt state: {e}"))?;
        self.current_token_position = self.system_token_len;
        Ok(())
    }
}

fn feed_tokens_raw(
    ctx: &mut LlamaContext<'static>,
    batch: &mut LlamaBatch<'static>,
    seq_id: i32,
    current_token_position: &mut i32,
    batch_size: usize,
    tokens: &[TokenID],
) -> anyhow::Result<()> {
    if tokens.is_empty() {
        return Ok(());
    }
    let llama_tokens = to_llama_tokens(tokens);
    let chunks: Vec<_> = llama_tokens.chunks(batch_size).collect();
    let last_chunk_idx = chunks.len() - 1;

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        batch.clear();
        let is_last_chunk = chunk_idx == last_chunk_idx;
        let last_token_idx = chunk.len() - 1;

        for (i, token) in chunk.iter().enumerate() {
            let logits = is_last_chunk && i == last_token_idx;
            batch
                .add(*token, *current_token_position, &[seq_id], logits)
                .map_err(|e| anyhow::anyhow!("failed to add token to batch: {e}"))?;
            *current_token_position += 1;
        }

        ctx.decode(batch)
            .map_err(|e| anyhow::anyhow!("llama_decode failed: {e}"))?;
    }
    Ok(())
}

impl Llm for LlamaLlm {
    fn feed_tokens(&mut self, tokens: &[TokenID]) -> anyhow::Result<()> {
        feed_tokens_raw(
            &mut self.ctx,
            &mut self.batch,
            self.seq_id,
            &mut self.current_token_position,
            self.batch_size,
            tokens,
        )
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
