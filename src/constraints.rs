use llguidance::{
    api::{GrammarInit, TopLevelGrammar},
    earley::SlicedBiasComputer,
    toktrie::{InferenceCapabilities, TokEnv},
    Constraint, ParserFactory,
};

use crate::grammar::GrammarFlow;

pub fn new_default_constraint(conversation: &GrammarFlow, tok_env: &TokEnv) -> Constraint {
    let parser_factory = ParserFactory::new(
        &tok_env,
        InferenceCapabilities {
            ff_tokens: true,
            conditional_ff_tokens: true,
            backtrack: true,
            fork: true,
        },
        &SlicedBiasComputer::general_slices(),
    )
    .unwrap();
    let grammar = TopLevelGrammar::from_lark(conversation.lark_grammar.to_string());
    let token_parser = parser_factory
        .create_parser_from_init_default(GrammarInit::Serialized(grammar))
        .unwrap();

    // get the constraint object which is used to compute the masks
    Constraint::new(token_parser)
}
