use std::io;
use std::sync::Arc;

use llguidance::toktrie::TokenizerEnv;
use llguidance::Constraint;

use crate::grammar::GrammarFlow;

use crate::inference::Llm;
use crate::llama_tokenizer::{LlamaTokenizerEnv, END_TURN_TOKEN, ID_END_TOKEN, ID_START_TOKEN};
use crate::token::Canidate;

pub struct ConversationData {
    pub tokenizer: Arc<LlamaTokenizerEnv>,
    pub llm: Box<dyn Llm>,
    pub grammar_flow: GrammarFlow,
    pub constraint: Constraint,
}

impl ConversationData {
    pub fn simple_hitl_generation(&mut self, max_tokens: usize, top_candidate_count: usize) {
        let initial_tokens = self.constraint.process_prompt(vec![]);
        self.llm.feed_tokens(&initial_tokens);

        let mut running_input = self.tokenizer.tokens_to_string(&initial_tokens);

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
        let initial_prompt = format!("{ID_START_TOKEN}user{ID_END_TOKEN}{input}{END_TURN_TOKEN}{ID_START_TOKEN}assistant{ID_END_TOKEN}{running_input}");
        let initial_tokens = self.tokenizer.tokenize(&initial_prompt);
        self.llm.feed_tokens(&initial_tokens);

        println!("Conversation: ```{initial_prompt}```");

        for _ in 0..max_tokens {
            // the conversation state should be
            // "...<|start_header_id|>assistant<|end_header_id|>...";
            // getting the top candiates from llm inferance
            let mut canidates = self.llm.get_canidates();
            let premask_top = canidates.top_n(top_candidate_count);
            let mask = self.constraint.compute_mask().unwrap();
            let sample_mask = mask.sample_mask.clone().unwrap();
            canidates.constrain(&sample_mask);
            let mask_top = canidates.top_n(top_candidate_count);
            println!("\n\nTop {top_candidate_count} unmasked tokens:");
            self.print_canidates(&premask_top);
            println!("\n\nTop {top_candidate_count} masked tokens:");
            self.print_canidates(&mask_top);

            // choose the top masked
            let last_token = mask_top.first().unwrap().token_id;

            // dbg!(self.constraint.parser.dump_state());
            let last_commit_result = self.constraint.commit_token(Some(last_token)).unwrap();
            let ff_tokens = last_commit_result.ff_tokens;
            // dbg!(self.tokenizer.tokens_to_string(&ff_tokens));
            // commit the ff_tokens to the constraint and the llm
            let mut last_commit_result = None;
            for token in ff_tokens.iter().skip(1) {
                last_commit_result = Some(self.constraint.commit_token(Some(*token)).unwrap())
            }

            // the ff tokens would include the current token
            if ff_tokens.is_empty() {
                self.llm.feed_tokens(&[last_token]);
                running_input += &self.tokenizer.tokens_to_string(&[last_token]);
            } else {
                self.llm.feed_tokens(&ff_tokens);
                running_input += &self.tokenizer.tokens_to_string(&ff_tokens);
            }

            println!("Conversation: ```{running_input}```");

            if let Some(r) = last_commit_result
                && r.stop
            {
                println!("The grammar is over stopping");
                break;
            }
        }
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
