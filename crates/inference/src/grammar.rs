use askama::Template;
use serde::{Deserialize, Serialize};

use crate::llama_tokenizer::{END_TURN_TOKEN, ID_END_TOKEN, ID_START_TOKEN};

// ---------------------------------------------------------------------------
// Shared wire type
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VCmessage {
    pub category: String,
    pub kind: String,
    pub description: String,
    pub mlr_message: String, // Message with placeholder URLs like [https://docupdate.io/x1]
    pub message: String,     // Message with actual URLs
}

// ---------------------------------------------------------------------------
// Askama templates
// ---------------------------------------------------------------------------

#[derive(Template)]
#[template(path = "system_prompt.txt", escape = "none")]
struct SystemPromptTemplate<'a> {
    brand_name: &'a str,
    messages: &'a [VCmessage],
}

/// Each entry is a fully-escaped lark string literal for one complete response,
/// e.g. `"Category: Safety\n\nApproved message text..."`.
/// Lark interprets `\n` inside a string literal as a newline character.
#[derive(Template)]
#[template(path = "grammar.lark", escape = "none")]
struct GrammarTemplate {
    response_literals: Vec<String>,
}

// ---------------------------------------------------------------------------
// Helper — escape a plain string for use inside a lark double-quoted literal
// ---------------------------------------------------------------------------

fn lark_str_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// GrammarFlow — rendered once per agent selection
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct GrammarFlow {
    pub system_prompt: String,
    pub lark_grammar: String,
}

impl GrammarFlow {
    pub fn new(brand_name: &str, vc_messages: &[VCmessage]) -> anyhow::Result<Self> {
        let system_prompt = SystemPromptTemplate {
            brand_name,
            messages: vc_messages,
        }
        .render()
        .map_err(|e| anyhow::anyhow!("failed to render system_prompt template: {e}"))?;

        // Build one lark string literal per approved response.
        // Each literal encodes the complete output the model should produce:
        //   "Category: {name}\n\n{message text}"
        // After the grammar disambiguates which option was chosen, all remaining
        // characters are emitted as fast-forward tokens.
        let response_literals = vc_messages
            .iter()
            .map(|m| {
                let esc_cat = lark_str_escape(&m.category);
                let esc_msg = lark_str_escape(&m.message);
                format!("\"Category: {esc_cat}\\n\\n{esc_msg}\"")
            })
            .collect::<Vec<_>>();

        let lark_grammar = GrammarTemplate { response_literals }
            .render()
            .map_err(|e| anyhow::anyhow!("failed to render grammar template: {e}"))?;

        Ok(Self {
            system_prompt,
            lark_grammar,
        })
    }

    pub fn get_system_prompt(&self) -> String {
        format!(
            "{ID_START_TOKEN}system{ID_END_TOKEN}{}{END_TURN_TOKEN}",
            self.system_prompt
        )
    }
}
