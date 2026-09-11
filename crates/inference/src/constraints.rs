use llguidance::{
    api::{GrammarInit, TopLevelGrammar},
    earley::SlicedBiasComputer,
    toktrie::{InferenceCapabilities, TokEnv},
    Constraint, ParserFactory,
};

use crate::grammar::GrammarFlow;

pub fn new_parser_factory(tok_env: &TokEnv) -> ParserFactory {
    ParserFactory::new(
        tok_env,
        InferenceCapabilities {
            ff_tokens: true,
            conditional_ff_tokens: true,
            backtrack: true,
            fork: true,
        },
        &SlicedBiasComputer::general_slices(),
    )
    .expect("failed to build llguidance ParserFactory")
}

pub fn new_constraint(parser_factory: &ParserFactory, grammar_flow: &GrammarFlow) -> Constraint {
    let grammar = TopLevelGrammar::from_lark(grammar_flow.lark_grammar.to_string());
    let token_parser = parser_factory
        .create_parser_from_init_default(GrammarInit::Serialized(grammar))
        .expect("failed to create grammar parser");
    Constraint::new(token_parser)
}

pub fn new_default_constraint(grammar_flow: &GrammarFlow, tok_env: &TokEnv) -> Constraint {
    new_constraint(&new_parser_factory(tok_env), grammar_flow)
}
