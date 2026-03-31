use std::io;
use std::sync::Arc;

use llguidance::Constraint;
use llguidance::toktrie::TokenizerEnv;

use crate::grammar::GrammarFlow;

use crate::inference::Llm;
use crate::llama_tokenizer::{END_TURN_TOKEN, ID_END_TOKEN, ID_START_TOKEN, LlamaTokenizerEnv};
use crate::token::Canidate;

pub struct ConversationInfo {
    pub tokenizer: Arc<LlamaTokenizerEnv>,
    pub llm: Box<dyn Llm>,
    pub grammar_flow: GrammarFlow,
    pub constraint: Constraint,
}

pub struct ConversationData {
    pub tokenizer: Arc<LlamaTokenizerEnv>,
    pub llm: Box<dyn Llm>,
    pub grammar_flow: GrammarFlow,
    pub constraint: Constraint,
    running_input: String,
}

impl ConversationData {
    pub fn new(info: ConversationInfo) -> Self {
        ConversationData {
            tokenizer: info.tokenizer,
            llm: info.llm,
            grammar_flow: info.grammar_flow,
            constraint: info.constraint,
            running_input: "".to_string(),
        }
    }

    pub fn simple_hitl_generation(&mut self, max_tokens: usize, top_candidate_count: usize) {
        let initial_tokens = self.constraint.process_prompt(vec![]);
        self.llm.feed_tokens(&initial_tokens);

        self.running_input = self.tokenizer.tokens_to_string(&initial_tokens);

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");
        input = input.strip_suffix("\n").unwrap().to_string();

        let initial_prompt = format!(
            "{ID_START_TOKEN}user{ID_END_TOKEN}{input}{END_TURN_TOKEN}{ID_START_TOKEN}assistant{ID_END_TOKEN}{}",
            self.running_input
        );
        let initial_tokens = self.tokenizer.tokenize(&initial_prompt);
        self.llm.feed_tokens(&initial_tokens);

        println!("Conversation: ```\n{initial_prompt}\n```");
        self.running_input = initial_prompt;

        for _ in 0..max_tokens {
            match self.commit_inference(top_candidate_count) {
                true => (),
                false => break,
            };
        }
        println!("Conversation complete: ```\n{}\n```", self.running_input);
    }

    /// Commit one token (plus any fast-forward tokens). Returns `true` to continue.
    pub fn commit_inference(&mut self, top_candidate_count: usize) -> bool {
        let mut canidates = self.llm.get_canidates();
        let premask_top = canidates.top_n(top_candidate_count);
        let mask = self.constraint.compute_mask().unwrap();
        let sample_mask = match &mask.sample_mask {
            Some(m) => m,
            None => return false,
        };
        canidates.constrain(sample_mask);
        let mask_top = canidates.top_n(top_candidate_count);
        println!("\n\nTop {top_candidate_count} unmasked tokens:");
        self.print_canidates(&premask_top);
        println!("\n\nTop {top_candidate_count} masked tokens:");
        self.print_canidates(&mask_top);

        let last_token = mask_top.first().unwrap().token_id;

        let last_commit_result = self.constraint.commit_token(Some(last_token)).unwrap();
        let ff_tokens = last_commit_result.ff_tokens;

        let mut last_commit_result = None;
        for token in ff_tokens.iter().skip(1) {
            last_commit_result = Some(self.constraint.commit_token(Some(*token)).unwrap())
        }

        if ff_tokens.is_empty() {
            self.llm.feed_tokens(&[last_token]);
            self.running_input += &self.tokenizer.tokens_to_string(&[last_token]);
        } else {
            self.llm.feed_tokens(&ff_tokens);
            self.running_input += &self.tokenizer.tokens_to_string(&ff_tokens);
        }

        println!("Conversation: ```\n{}\n```", self.running_input);

        if let Some(commit_result) = last_commit_result {
            return !commit_result.stop;
        }

        true
    }

    pub fn grammar_test(&mut self) {}

    fn print_canidates(&self, canidates: &[Canidate]) {
        for (i, c) in canidates.iter().enumerate() {
            println!(
                "{}. {} - {:.4}",
                i + 1,
                self.tokenizer.tokens_to_string(&[c.token_id]),
                c.logit
            )
        }
    }
}

struct CategoryBias {
    category: String,
    bias: f32,
}

struct ScaledCanidate {
    canidate: Canidate,
    total_bias: usize,
    bias_count: usize,
}

fn get_margins() -> Vec<(CategoryBias, f32)> {
    vec![]
}

fn get_biased_canidates(
    margins: Vec<(CategoryBias, f32)>,
    candidates: Vec<Canidate>,
) -> Vec<Canidate> {
    let _scaled_canidates = candidates
        .iter()
        .cloned()
        .map(|c| ScaledCanidate {
            canidate: c,
            total_bias: 0,
            bias_count: 0,
        })
        .collect::<Vec<_>>();

    for (_bias, _scale) in margins {}
    vec![]
}
