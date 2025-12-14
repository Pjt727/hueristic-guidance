mod constraints;
mod conversation_loop;
mod csv_loader;
mod grammar;
mod inference;
mod llama_tokenizer;
mod token;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llguidance::toktrie::TokenizerEnv;
use std::path::PathBuf;
use std::sync::Arc;

use crate::conversation_loop::{ConversationData, ConversationInfo};
use crate::grammar::GrammarFlow;

fn get_pemazyre_grammar_flow() -> Result<GrammarFlow, Box<dyn std::error::Error>> {
    let csv_path = "data/pemazyre.csv";
    let vc_messages = csv_loader::load_pemazyre_responses(csv_path)?;

    println!("Loaded {} Pemazyre response messages from CSV", vc_messages.len());

    Ok(GrammarFlow::new("PEMAZYRE".to_string(), vc_messages))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing llama.cpp backend...");
    let backend = LlamaBackend::init()?;
    // let model_path = PathBuf::from("models/Meta-Llama-3-8B.Q5_K_M.gguf");
    let model_path = PathBuf::from("models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf");
    println!("Loading model from {:?}...", model_path);

    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;
    let model = Arc::new(model);

    let tokenizer = llama_tokenizer::LlamaTokenizerEnv::new(model.clone());
    let tokenizer = Arc::new(tokenizer);
    let grammar_flow = get_pemazyre_grammar_flow()?;
    let system_message = grammar_flow.get_system_prompt();
    let initial_tokens = tokenizer.tokenize(&system_message);
    let llm = inference::LlamaLlm::new(&backend, model.clone(), &initial_tokens);
    let tok_env = llama_tokenizer::LlamaTokenizerEnv::new(model.clone());
    let tok_env: Arc<dyn TokenizerEnv + Sync + 'static> = Arc::new(tok_env);

    let constraint = constraints::new_default_constraint(&grammar_flow, &tok_env);

    let mut conversation_data = ConversationData::new(ConversationInfo {
        llm: Box::new(llm),
        tokenizer,
        grammar_flow,
        constraint,
    });

    conversation_data.simple_hitl_generation(200, 10);

    Ok(())
}
