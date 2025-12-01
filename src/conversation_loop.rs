use std::io;
use std::sync::Arc;

use llguidance::toktrie::TokenizerEnv;
use llguidance::Constraint;

use crate::grammar::{GrammarFlow, END_TURN_TOKEN_TOKEN};

use crate::inference::Llm;
use crate::llama_tokenizer::LlamaTokenizerEnv;
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

        let mut is_human_turn = true;

        // dbg!(self.constraint.parser.dump_state());
        // self.constraint
        //     .commit_token(Some(self.tokenizer.tokenize(END_TURN_TOKEN_TOKEN)[0]))
        //     .unwrap();
        // dbg!(self.constraint.parser.dump_state());

        println!("Conversation: ```{running_input}```");

        for _ in 0..max_tokens {
            let last_token;
            if is_human_turn {
                // the conversation state should be "...<|start_header_id|>user<|end_header_id|>";
                is_human_turn = false;
                let mut input = String::new();
                io::stdin()
                    .read_line(&mut input)
                    .expect("Failed to read line");
                input = input.strip_suffix("\n").unwrap().to_string();

                // add the the human tokens to the constraint and the llm
                let human_input_tokens = self.tokenizer.tokenize(&input);
                running_input += &input;
                self.constraint.force_tokens(&human_input_tokens).unwrap();
                self.llm.feed_tokens(&human_input_tokens);

                // we have to add the end token
                let _mask = self.constraint.compute_mask(); // we have to computer the mask or it
                                                            // wont' add the token
                last_token = self.tokenizer.tokenize(END_TURN_TOKEN_TOKEN)[0];
            } else {
                is_human_turn = true;
                // the conversation state should be "...<|start_header_id|>assistant<|end_header_id|>";
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
                last_token = mask_top.first().unwrap().token_id;
            }
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
