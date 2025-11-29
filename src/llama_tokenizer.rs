use llama_cpp_2::{
    model::{AddBos, LlamaModel, Special},
    token::LlamaToken,
};
use llguidance::toktrie::{TokTrie, TokenId, TokenizerEnv};
use std::sync::Arc;

pub struct LlamaTokenizerEnv {
    model: Arc<LlamaModel>,
    tok_trie: TokTrie,
}

impl LlamaTokenizerEnv {
    pub fn new(
        tok_trie: TokTrie,
        model: Arc<LlamaModel>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Build the token trie from the model's vocabulary
        Ok(Self { model, tok_trie })
    }
}

impl TokenizerEnv for LlamaTokenizerEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.tok_trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        // Convert bytes to string (lossy conversion for invalid UTF-8)
        let text = String::from_utf8_lossy(s);

        // Use llama.cpp tokenization without adding BOS
        // AddBos::Never ensures we don't add a beginning-of-sequence token
        self.model
            .str_to_token(&text, AddBos::Never)
            .unwrap()
            .iter()
            .map(|&t| t.0 as TokenId)
            .collect()
        // Ok(tokens) => tokens.iter().map(|&t| t as TokenId).collect(),
    }
}
