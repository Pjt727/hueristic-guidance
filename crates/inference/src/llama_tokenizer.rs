use llama_cpp_2::{
    model::{AddBos, LlamaModel},
    token::LlamaToken,
    TokenToStringError,
};
use llguidance::toktrie::{TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use std::sync::Arc;

use crate::token::TokenID;

/// Chat wrapper tokens for the currently loaded GGUF.
/// Llama-3 and ChatML (Qwen) use different special-token vocabularies.
#[derive(Debug, Clone)]
pub struct ChatFormat {
    pub start_header: &'static str,
    pub end_header: &'static str,
    pub end_turn: &'static str,
    /// Tokens that must exist in the vocab and be marked special in the toktrie.
    pub special_tokens: &'static [&'static str],
}

impl ChatFormat {
    pub const LLAMA3: Self = Self {
        start_header: "<|start_header_id|>",
        end_header: "<|end_header_id|>",
        end_turn: "<|eot_id|>",
        special_tokens: &["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
    };

    pub const CHATML: Self = Self {
        start_header: "<|im_start|>",
        end_header: "\n",
        end_turn: "<|im_end|>\n",
        special_tokens: &["<|im_start|>", "<|im_end|>"],
    };

    pub fn wrap_system(&self, content: &str) -> String {
        format!(
            "{}system{}{}{}",
            self.start_header, self.end_header, content, self.end_turn
        )
    }

    pub fn wrap_user_turn(&self, prompt: &str, assistant_prefix: &str) -> String {
        format!(
            "{}user{}{}{}{}assistant{}{}",
            self.start_header,
            self.end_header,
            prompt,
            self.end_turn,
            self.start_header,
            self.end_header,
            assistant_prefix
        )
    }
}

fn tokenizes_as_single_token(model: &LlamaModel, text: &str) -> bool {
    matches!(
        model.str_to_token(text, AddBos::Never),
        Ok(tokens) if tokens.len() == 1
    )
}

fn detect_chat_format(model: &LlamaModel) -> anyhow::Result<ChatFormat> {
    if ChatFormat::LLAMA3
        .special_tokens
        .iter()
        .all(|token| tokenizes_as_single_token(model, token))
    {
        tracing::info!("using Llama-3 chat special tokens");
        return Ok(ChatFormat::LLAMA3);
    }
    if ChatFormat::CHATML
        .special_tokens
        .iter()
        .all(|token| tokenizes_as_single_token(model, token))
    {
        tracing::info!("using ChatML (Qwen) chat special tokens");
        return Ok(ChatFormat::CHATML);
    }

    let arch = model
        .meta_val_str("general.architecture")
        .unwrap_or_else(|_| "unknown".to_string());
    anyhow::bail!(
        "model architecture '{arch}' does not expose Llama-3 or ChatML special tokens; \
         cannot wrap prompts for constrained decoding"
    )
}

fn token_to_bytes(model: &LlamaModel, token: LlamaToken) -> Vec<u8> {
    match model.token_to_piece_bytes(token, 8, true, None) {
        Ok(bytes) => bytes,
        Err(TokenToStringError::InsufficientBufferSpace(i)) => model
            .token_to_piece_bytes(
                token,
                (-i).try_into().expect("buffer size is positive"),
                true,
                None,
            )
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

pub struct LlamaTokenizerEnv {
    model: Arc<LlamaModel>,
    tok_trie: TokTrie,
    pub chat_format: ChatFormat,
}

impl LlamaTokenizerEnv {
    pub fn new(model: Arc<LlamaModel>) -> anyhow::Result<Self> {
        let chat_format = detect_chat_format(&model)?;

        let mut must_have_special_token_ids: Vec<_> = chat_format
            .special_tokens
            .iter()
            .map(|t_str| {
                model
                    .str_to_token(t_str, AddBos::Never)
                    .expect("special token presence already verified")[0]
            })
            .collect();

        // `true` = decode special/control tokens rather than treating them as plaintext.
        let all_words: Vec<Vec<u8>> = model
            .tokens(true)
            .map(|(t, _t_str)| {
                let mut bytes = token_to_bytes(&model, t);
                // https://github.com/guidance-ai/llguidance/blob/main/docs/special_tokens.md
                // need to add 0xff prefix to words in the tree
                if must_have_special_token_ids.contains(&t) {
                    tracing::debug!(token = %t, "marking chat special token in toktrie");
                    must_have_special_token_ids = must_have_special_token_ids
                        .iter()
                        .cloned()
                        .filter(|t_1| t_1 != &t)
                        .collect();
                    bytes.insert(0, TokTrie::SPECIAL_TOKEN_MARKER);
                }
                bytes
            })
            .collect();

        assert!(
            must_have_special_token_ids.is_empty(),
            "Expected special tokens missing {}",
            must_have_special_token_ids
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let token_info = TokRxInfo {
            vocab_size: model.n_vocab() as u32,
            tok_eos: model.token_eos().0 as u32,
            tok_bos: Some(model.token_bos().0 as u32),
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        let tok_trie = TokTrie::from(&token_info, &all_words);
        Ok(Self {
            model,
            tok_trie,
            chat_format,
        })
    }

    /// Decode tokens to text by concatenating piece bytes first.
    ///
    /// Qwen (and other byte-fallback tokenizers) emit individual bytes as tokens.
    /// Decoding each token with strict UTF-8 fails; joining bytes then lossily
    /// decoding recovers the intended string across token boundaries.
    pub fn tokens_to_string(&self, tokens: &[TokenID]) -> String {
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for &tid in tokens {
            bytes.extend(token_to_bytes(
                &self.model,
                LlamaToken::new(tid as i32),
            ));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl TokenizerEnv for LlamaTokenizerEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.tok_trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        let text = String::from_utf8_lossy(s);
        self.model
            .str_to_token(&text, AddBos::Never)
            .unwrap()
            .iter()
            .map(|&t| t.0 as TokenId)
            .collect()
    }
}
