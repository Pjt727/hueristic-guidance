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
use std::{collections::HashMap, sync::Arc};

use crate::{
    get_default_grammar_flow,
    grammar::{END_TOKEN, SPECIAL_TOKENS},
    token::TokenID,
};

pub struct LlamaTokenizerEnv {
    model: Arc<LlamaModel>,
    tok_trie: TokTrie,
    special_tokens_to_string: HashMap<TokenID, String>,
}

impl LlamaTokenizerEnv {
    pub fn new(model: Arc<LlamaModel>) -> Self {
        let mut tok_end_of_turn = None;
        let mut special_tokens_to_string = HashMap::new();

        let all_words: Vec<Vec<u8>> = model
            .tokens(Special::Tokenize)
            .map(|(t, t_str)| {
                let mut bytes = model.token_to_bytes(t, Special::Tokenize).unwrap();

                // https://github.com/guidance-ai/llguidance/blob/main/docs/special_tokens.md
                // need to add 0xff prefix to words in the tree
                if let Ok(string_rep) = t_str {
                    let maybe_token = SPECIAL_TOKENS
                        .iter()
                        .find(|t| t.to_string() == string_rep.to_string());
                    if let Some(special_token) = maybe_token {
                        if string_rep == END_TOKEN {
                            tok_end_of_turn = Some(t.0 as TokenID)
                        }
                        special_tokens_to_string.insert(t.0 as u32, special_token.to_string());
                        bytes.insert(0, TokTrie::SPECIAL_TOKEN_MARKER)
                    }
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
            tok_end_of_turn: None,
        };
        let tok_trie = TokTrie::from(&token_info, &all_words);
        Self {
            model,
            tok_trie,
            special_tokens_to_string,
        }
    }

    pub fn tokens_to_string(&self, tokens: &[TokenID]) -> String {
        let res = self.model.tokens_to_str(
            &tokens
                .iter()
                .map(|t| LlamaToken::new(*t as i32))
                .collect::<Vec<_>>(),
            Special::Tokenize,
        );
        match res {
            Ok(s) => s,
            Err(_) => "Invalid utf-8".to_string(),
        }
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
