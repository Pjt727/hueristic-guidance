mod constraints;
mod conversation_loop;
mod grammar;
mod inference;
mod llama_tokenizer;
mod token;

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

use crate::conversation_loop::ConversationData;
use crate::grammar::{GrammarFlow, VCmessage};

fn get_default_grammar_flow() -> GrammarFlow {
    GrammarFlow::new(
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing llama.cpp backend...");
    let backend = LlamaBackend::init()?;
    let model_path = PathBuf::from("models/Meta-Llama-3-8B.Q5_K_M.gguf");
    println!("Loading model from {:?}...", model_path);

    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;
    let model = Arc::new(model);

    let tokenizer = llama_tokenizer::LlamaTokenizerEnv::new(model.clone());
    let tokenizer = Arc::new(tokenizer);
    let grammar_flow = get_default_grammar_flow();
    let system_message = grammar_flow.get_system_prompt();
    let initial_tokens = tokenizer.tokenize(&system_message);
    let llm = inference::LlamaLlm::new(&backend, model.clone(), &initial_tokens);
    let tok_env = llama_tokenizer::LlamaTokenizerEnv::new(model.clone());
    let tok_env: Arc<dyn TokenizerEnv + Sync + 'static> = Arc::new(tok_env);

    let constraint = constraints::new_default_constraint(&grammar_flow, &tok_env);

    let mut conversation_data = ConversationData {
        tokenizer,
        llm: Box::new(llm),
        grammar_flow,
        constraint,
    };

    conversation_data.simple_generation(200, 5);

    Ok(())
}
