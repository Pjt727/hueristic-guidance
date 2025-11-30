use llama_cpp_2::{
    model::{AddBos, LlamaModel, Special},
    token::LlamaToken,
};
use llguidance::{
    api::{GrammarInit, TopLevelGrammar},
    earley::SlicedBiasComputer,
    toktrie::{InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv},
    Constraint, ParserFactory,
};
use std::sync::Arc;

use crate::{
    get_default_grammar_flow,
    grammar::{END_TOKEN, SPECIAL_TOKENS},
    token::TokenID,
};

pub struct LlamaTokenizerEnv {
    model: Arc<LlamaModel>,
    tok_trie: TokTrie,
}

impl LlamaTokenizerEnv {
    pub fn new(model: Arc<LlamaModel>) -> Self {
        let mut tok_end_of_turn = None;

        let all_words: Vec<Vec<u8>> = model
            .tokens(Special::Tokenize)
            .map(|(t, t_str)| {
                let mut bytes = model.token_to_bytes(t, Special::Tokenize).unwrap();

                // https://github.com/guidance-ai/llguidance/blob/main/docs/special_tokens.md
                if let Ok(string_rep) = t_str
                    && SPECIAL_TOKENS.contains(&string_rep.as_str())
                {
                    if string_rep == END_TOKEN {
                        tok_end_of_turn = Some(t.0 as TokenID)
                    }
                    bytes.insert(0, TokTrie::SPECIAL_TOKEN_MARKER)
                }
                bytes
            })
            .collect();

        let token_info = TokRxInfo {
            vocab_size: model.tokens(Special::Tokenize).count() as u32,
            tok_eos: model.token_eos().0 as u32,
            tok_bos: Some(model.token_bos().0 as u32),
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None, // maybe I need find the token
        };
        let tok_trie = TokTrie::from(&token_info, &all_words);
        Self { model, tok_trie }
    }

    pub fn tokens_to_string(&self, tokens: &[TokenID]) -> String {
        self.model
            .tokens_to_str(
                &tokens
                    .iter()
                    .map(|t| LlamaToken::new(*t as i32))
                    .collect::<Vec<_>>(),
                Special::Tokenize,
            )
            .unwrap()
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
