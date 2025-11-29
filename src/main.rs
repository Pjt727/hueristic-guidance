mod conversation_grammar;
mod llama_tokenizer;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaModel, Special};
use llama_cpp_2::token::LlamaToken;
use llama_tokenizer::LlamaTokenizerEnv;
use llguidance::api::{GrammarInit, TopLevelGrammar};
use llguidance::earley::SlicedBiasComputer;
use llguidance::ffi::{
    llg_new_constraint_lark, llg_new_tokenizer, LlgConstraintInit, LlgTokenizer,
};
use llguidance::toktrie::{
    ApproximateTokEnv, InferenceCapabilities, SimpleVob, TokEnv, TokRxInfo, TokTrie, TokenizerEnv,
};
use llguidance::{Constraint, ParserFactory};
use std::io;
use std::num::NonZero;
use std::path::PathBuf;
use std::sync::Arc;

use crate::conversation_grammar::{Conversation, VCmessage};

const SIMPLE_LARK_GRAMMAR: &str = r#"start: "Roses are red,\nViolets are " COLOR "\nThe honey's " ADJECTIVE ", and so are you"
ADJECTIVE: "sweet" | "bitter"
COLOR: "blue" | "red"
"#;

fn to_llama_token(usize_token: u32) -> LlamaToken {
    LlamaToken(usize_token as i32)
}

fn tokens_to_string(model: Arc<LlamaModel>, tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(|t| {
            model
                .token_to_str(to_llama_token(*t), Special::Tokenize)
                .unwrap()
        })
        .fold("".to_string(), |s1, s2| s1 + &s2)
}
// HCP:
// “Can you send me sample info?”
// VC:
// """
// Category: samples
// Samples for {brandname} are available at the closest store.
// """
//
// HCP:
// “What’s the dosage?”
// VC:
// """
// Category: dosing
// Dosage information for {brandname} is available on the back of the bottle.
// """

fn get_default_conversation() -> Conversation {
    Conversation::new(
        "XARELTO".to_string(),
        vec![
            VCmessage {
                category: "samples".to_string(),
                message: "Samples for XARELTO are available at the closest store.".to_string(),
            },
            VCmessage {
                category: "dosing".to_string(),
                message: "Dosage information for XARELTO is available on the back of the bottle."
                    .to_string(),
            },
        ],
    )
}

fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    // Find the maximum score for numerical stability
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Shift scores and calculate exponentials
    let exp_scores: Vec<f32> = scores
        .iter()
        .map(|&score| (score - max_score).exp())
        .collect();

    // Calculate the sum of exponentials
    let sum_exp_scores: f32 = exp_scores.iter().sum();

    // Calculate softmax probabilities
    exp_scores
        .iter()
        .map(|&exp_score| exp_score / sum_exp_scores)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing llama.cpp backend...");
    let backend = LlamaBackend::init()?;
    let model_path = PathBuf::from("models/Meta-Llama-3-8B.Q5_K_M.gguf");
    println!("Loading model from {:?}...", model_path);

    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;
    let model = Arc::new(model);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZero::new(2048).unwrap())) // Context size
        .with_n_batch(512); // Batch size

    // setup the tokenizer
    println!("Creating tokenizer environment...");
    let token_info = TokRxInfo {
        vocab_size: model.tokens(Special::Tokenize).count() as u32,
        tok_eos: model.token_eos().0 as u32,
        tok_bos: Some(model.token_bos().0 as u32),
        tok_pad: None,
        tok_unk: None,
        tok_end_of_turn: None,
    };
    let all_words: Vec<Vec<u8>> = model
        .tokens(Special::Tokenize)
        .map(|(t, _)| model.token_to_bytes(t, Special::Tokenize).unwrap())
        .collect();
    let tok_trie = TokTrie::from(&token_info, &all_words);
    let tokenizer = LlamaTokenizerEnv::new(tok_trie, Arc::clone(&model))?;
    let tok_env: Arc<dyn TokenizerEnv + Sync + 'static> = Arc::new(tokenizer);

    // set up the grammar specifics
    let parser_factory = ParserFactory::new(
        &tok_env,
        InferenceCapabilities {
            ff_tokens: true,
            conditional_ff_tokens: true,
            backtrack: true,
            fork: true,
        },
        &SlicedBiasComputer::general_slices(),
    )?;
    let grammar = TopLevelGrammar::from_lark(SIMPLE_LARK_GRAMMAR.to_string());
    let token_parser =
        parser_factory.create_parser_from_init_default(GrammarInit::Serialized(grammar))?;

    // get the constraint object which is used to compute the masks
    let mut constraint = Constraint::new(token_parser);

    // starting input which needs to go into both the constraint as well as the llm
    let starting_input = "Roses are red,\nViolets are ";
    let token_ids = tok_env.tokenize(starting_input);
    let token_ids_llama = model.str_to_token(starting_input, llama_cpp_2::model::AddBos::Never);
    dbg!(&token_ids);
    dbg!(&token_ids_llama);
    // dbg!(tok_env.tokenize_is_canonical());
    // dbg!(tokens_to_string(model.clone(), &token_ids));
    constraint.start_without_prompt();
    constraint.force_tokens(&token_ids)?;
    // why is the healed string "  roses are red violets are roses are red violets are"
    // dbg!(model.token_attr(LlamaToken(28705)));
    // dbg!(model.token_to_bytes(LlamaToken(28705), Special::Tokenize)?);
    // dbg!(model.token_eos());
    // dbg!(model.token_bos());
    // dbg!(healed_string);

    // Create context and process the prompt
    let mut ctx = model.new_context(&backend, ctx_params)?;

    // Create a batch and add the prompt tokens
    let mut batch = LlamaBatch::new(512, 1);
    let seq_id = 0;

    for (i, &token_id) in token_ids.iter().enumerate() {
        let is_last = i == token_ids.len() - 1;
        batch.add(to_llama_token(token_id), i as i32, &[seq_id], is_last)?;
    }

    // Encode the prompt
    ctx.decode(&mut batch)?;

    println!("\nPrompt processed. Starting inference loop...");
    println!("Input: {}", starting_input);
    print!("Output: ");

    // Inference loop
    let mut n_current = token_ids.len();
    let max_tokens = 20;
    let mut running_input = starting_input.to_string();

    for _ in 0..max_tokens {
        // Get logits for the last token
        let logits = ctx.candidates();

        // Get mask for valid token
        let mask = constraint.compute_mask()?;
        let sample_mask = mask.sample_mask.clone().unwrap();

        // get the logits from the next canidates
        let mut unmasked_logit_ids: Vec<_> = logits
            .enumerate()
            .map(|(id, data)| (id as u32, data.logit()))
            .collect();
        let logit_id_clone = unmasked_logit_ids.clone();
        let masked_logit_ids: Vec<_> = logit_id_clone
            .iter()
            .filter(|(id, _)| sample_mask.is_allowed(*id))
            .collect();
        let masked_logits: Vec<_> = masked_logit_ids.iter().map(|(_, p)| *p).collect();
        let masked_logits = softmax(&masked_logits);
        let mut masked_logit_ids: Vec<_> = masked_logit_ids
            .iter()
            .zip(masked_logits)
            .map(|((id, _), new_p)| (*id, new_p * 100.0))
            .collect();

        // Sort by logit (higher is better)
        masked_logit_ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        unmasked_logit_ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\n\nTop 5 unmasked tokens:");
        for (i, (token_id, logit)) in unmasked_logit_ids.iter().take(5).enumerate() {
            let token_str = tokens_to_string(model.clone(), &[*token_id]);
            println!("  {}. {:?} (logit: {:.4})", i + 1, token_str, logit);
        }

        println!("\n\nTop 5 masked tokens:");
        for (i, (token_id, logit)) in masked_logit_ids.iter().take(5).enumerate() {
            let token_str = tokens_to_string(model.clone(), &[*token_id]);
            println!("  {}. {:?} (logit: {:.4})", i + 1, token_str, logit);
        }

        // Select the token with highest probability (greedy sampling)
        let next_token = masked_logit_ids[0].0;

        // Check if we hit EOS
        if next_token == model.token_eos().0 as u32 {
            println!("\nReached EOS token");
            break;
        }

        // Add the token to the constraint and see if there are any FF tokens

        let commit_result = constraint.commit_token(Some(next_token))?;
        dbg!(&commit_result);
        dbg!(tokens_to_string(model.clone(), &commit_result.ff_tokens));
        for token in &commit_result.ff_tokens {
            constraint.commit_token(Some(*token))?;
        }

        // Add the token(s) to the batch for next iteration
        let all_llama_tokens: Vec<_> = commit_result
            .ff_tokens
            .iter()
            .cloned()
            .map(to_llama_token)
            .collect();
        batch.clear();
        for (i, &token_id) in all_llama_tokens.iter().enumerate() {
            let is_last = i == all_llama_tokens.len() - 1;
            batch.add(token_id, (n_current + i) as i32, &[seq_id], is_last)?;
        }
        n_current += all_llama_tokens.len();

        // Decode
        ctx.decode(&mut batch)?;

        // Add the token str representations to the running string
        running_input += &tokens_to_string(model.clone(), &commit_result.ff_tokens);
        // Print the full text
        print!("Text: {}", running_input);
        std::io::Write::flush(&mut std::io::stdout())?;

        // wait to generate next token(s)
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer)?; // Wait for user to press Enter
        if commit_result.stop {
            println!("\nReached the end of the grammar");
            break;
        }
    }

    println!("\n\nGeneration complete!");

    Ok(())
}
