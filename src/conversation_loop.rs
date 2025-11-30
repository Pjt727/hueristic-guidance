use std::sync::Arc;

use llama_cpp_2::model::LlamaModel;
use llguidance::{toktrie::TokEnv, Constraint};

use crate::grammar::GrammarFlow;

use crate::inference::Llm;
use crate::llama_tokenizer::LlamaTokenizerEnv;
use crate::token::{Canidate, Canidates};

pub struct ConversationData {
    pub tokenizer: Arc<LlamaTokenizerEnv>,
    pub llm: Box<dyn Llm>,
    pub grammar_flow: GrammarFlow,
    pub constraint: Constraint,
}

impl ConversationData {
    pub fn simple_generation(&mut self, max_tokens: usize, top_candidate_count: usize) {
        // see if the loop needs to generate any grammar first
        let initial_tokens = self.constraint.process_prompt(vec![]);
        // self.constraint.force_tokens(&initial_tokens).unwrap();
        // dbg!(initial_token);
        // // let commit_result = self.constraint.commit_token(Some("")).unwrap();
        // let commit_result = self.constraint.commit_token(None).unwrap();
        // dbg!(&commit_result);
        // for token in &initial_token {
        //     self.constraint.force_tokens(Some(*token)).unwrap();
        // }
        // self.llm.feed_tokens(&commit_result.ff_tokens);
        self.llm.feed_tokens(&initial_tokens);

        let mut running_input = "".to_string();

        for _ in 0..max_tokens {
            // getting and showing top canidates
            let mut canidates = self.llm.get_canidates();
            let premask_top = canidates.top_n_by_logits(top_candidate_count);
            let mask = self.constraint.compute_mask().unwrap();
            let sample_mask = mask.sample_mask.clone().unwrap();
            canidates.constrain(&sample_mask);
            let mask_top = canidates.top_n_by_logits(top_candidate_count);
            println!("\n\nTop {top_candidate_count} unmasked tokens:");
            self.print_canidates(&premask_top);
            println!("\n\nTop {top_candidate_count} masked tokens:");
            self.print_canidates(&mask_top);

            // choose the top masked
            let chosen_token = mask_top.first().unwrap();
            let commit_result = self
                .constraint
                .commit_token(Some(chosen_token.token_id))
                .unwrap();
            let ff_tokens = commit_result.ff_tokens;
            // commit the ff_tokens to the constraint and the llm
            let mut last_commit_result = None;
            for token in ff_tokens.iter().skip(1) {
                last_commit_result = Some(self.constraint.commit_token(Some(*token)).unwrap())
            }
            self.llm.feed_tokens(&ff_tokens);
            running_input += &self.tokenizer.tokens_to_string(&ff_tokens);

            println!("Conversation: ```{running_input}```");

            if let Some(r) = last_commit_result
                && r.stop
            {
                println!("The grammar is over stopping");
                break;
            }
        }
    }

    fn print_canidates(&self, canidates: &Vec<Canidate>) {
        for (i, c) in canidates.iter().enumerate() {
            println!(
                "{}. {} - {:.4}%",
                i,
                self.tokenizer.tokens_to_string(&[c.token_id]),
                c.probability * 100.0
            )
        }
    }
}
