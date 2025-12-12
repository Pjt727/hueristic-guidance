use std::io;
use std::sync::Arc;

use llguidance::toktrie::TokenizerEnv;
use llguidance::Constraint;

use crate::grammar::GrammarFlow;

use crate::inference::Llm;
use crate::llama_tokenizer::{LlamaTokenizerEnv, END_TURN_TOKEN, ID_END_TOKEN, ID_START_TOKEN};
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

        // dbg!(self.constraint.parser.dump_state());
        // self.constraint
        //     .commit_token(Some(self.tokenizer.tokenize(END_TURN_TOKEN_TOKEN)[0]))
        //     .unwrap();
        // dbg!(self.constraint.parser.dump_state());

        // get an initial user prompt
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

    /// find the next token to commit and commits it, also forces any of the next tokens
    /// returns whether the grammar is done
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

        // choose the top masked
        let last_token = mask_top.first().unwrap().token_id;

        let last_commit_result = self.constraint.commit_token(Some(last_token)).unwrap();
        let ff_tokens = last_commit_result.ff_tokens;

        // commit the ff_tokens to the constraint and the llm
        let mut last_commit_result = None;
        for token in ff_tokens.iter().skip(1) {
            last_commit_result = Some(self.constraint.commit_token(Some(*token)).unwrap())
        }

        // the ff tokens would include the current token
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

        return true;
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
