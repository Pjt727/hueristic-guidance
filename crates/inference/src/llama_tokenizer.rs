use llama_cpp_2::{
    model::{AddBos, LlamaModel, Special},
    token::LlamaToken,
};
use llguidance::toktrie::{TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use std::sync::Arc;

use crate::token::TokenID;

// lama 3
pub const ID_START_TOKEN: &str = "<|start_header_id|>";
pub const ID_END_TOKEN: &str = "<|end_header_id|>";
pub const END_TURN_TOKEN: &str = "<|eot_id|>";

pub const SPECIAL_TOKENS: [&str; 3] = [ID_START_TOKEN, ID_END_TOKEN, END_TURN_TOKEN];

pub struct LlamaTokenizerEnv {
    model: Arc<LlamaModel>,
    tok_trie: TokTrie,
}

impl LlamaTokenizerEnv {
    pub fn new(model: Arc<LlamaModel>) -> Self {
        let _tok_end_of_turn =
            Some(model.str_to_token(END_TURN_TOKEN, AddBos::Never).unwrap()[0].0 as u32);

        let mut must_have_special_token_ids: Vec<_> = SPECIAL_TOKENS
            .iter()
            .map(|t_str| model.str_to_token(t_str, AddBos::Never).unwrap()[0])
            .collect();

        let all_words: Vec<Vec<u8>> = model
            .tokens(Special::Tokenize)
            .map(|(t, _t_str)| {
                let mut bytes = model.token_to_bytes(t, Special::Tokenize).unwrap();
                // https://github.com/guidance-ai/llguidance/blob/main/docs/special_tokens.md
                // need to add 0xff prefix to words in the tree
                if must_have_special_token_ids.contains(&t) {
                    println!("Special Token Found! {t}");
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
            vocab_size: model.tokens(Special::Tokenize).count() as u32,
            tok_eos: model.token_eos().0 as u32,
            tok_bos: Some(model.token_bos().0 as u32),
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        let tok_trie = TokTrie::from(&token_info, &all_words);
        Self { model, tok_trie }
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
        let text = String::from_utf8_lossy(s);
        self.model
            .str_to_token(&text, AddBos::Never)
            .unwrap()
            .iter()
            .map(|&t| t.0 as TokenId)
            .collect()
    }
}
