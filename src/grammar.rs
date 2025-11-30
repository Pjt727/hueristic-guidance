use serde::{Deserialize, Serialize};

pub const ID_START_TOKEN: &str = "<|start_header_id|>";
pub const ID_END_TOKEN: &str = "<|end_header_id|>";
pub const END_TOKEN: &str = "<|eot_id|>";
pub const SPECIAL_TOKENS: [&str; 3] = [ID_START_TOKEN, ID_END_TOKEN, END_TOKEN];

const STARTING_CONVO_TEMPLATE: &str = r#"
This is a simulated conversation between:

* **HCP** — a healthcare provider who speaks freely.
* **VC** — the Virtual Coordinator, a chatbot that responds **only with predefined, approved messages**.
  The VC never improvises, never invents new content, and never produces text outside the approved list.

---

### **VC Behavior Rules:**

1. The HCP asks questions in free natural language.
2. The VC must **choose the most appropriate message** from the approved list below.
3. The VC must output the **exact wording** of the chosen message.
4. The VC must never invent new categories, new text, or alternative phrasings.
5. The VC should answer in **natural language**.

---

### **Approved VC Responses:**

Category: samples
Samples for {brandname} are available at the closest store.

Category: dosing
*Dosage information for {brandname} is available on the back of the bottle.*

---

### **Output Format:**

When responding, **only output the VC’s chosen sentence**.
Do not output the category name, do not explain the choice, and do not refer to the rules.

---

# Example of Expected Behavior

HCP:
“Can you send me sample info?”
VC:
"""
Category: samples
Samples for {brandname} are available at the closest store.
"""

HCP:
“What’s the dosage?”
VC:
"""
Category: dosing
Dosage information for {brandname} is available on the back of the bottle.
"""
"#;

const STARTING_CONVO_GRAMMAR_TEMPLATE: &str = r#"start: hcp_response
hcp_response: <|start_header_id|> "HCP" <|end_header_id|> hcp_content <|eot_id|> vc_response
hcp_content: /(.|\n){hcplimit}/
vc_response: <|start_header_id|> "VC" <|end_header_id|> "Category: " category_responses <|eot_id|> (hcp_response)?

// brand specific category information
"#;

#[derive(Debug, Serialize, Deserialize)]
pub struct VCmessage {
    pub category: String,
    pub message: String,
}

pub struct GrammarFlow {
    pub possible_vc_messages: Vec<VCmessage>,
    pub brand_name: String,
    pub system_prompt: String,
    pub lark_grammar: String,
}

impl GrammarFlow {
    pub fn new(brand_name: String, vc_messages: Vec<VCmessage>) -> Self {
        let system_prompt = STARTING_CONVO_TEMPLATE.replace("{brandname}", &brand_name);
        let mut lark_grammar = STARTING_CONVO_GRAMMAR_TEMPLATE.replace("{hcplimit}", "{0,50}");
        let category_response = vc_messages
            .iter()
            .map(|vc_m| {
                format!(
                    "\"{category}\\n\" response_{category}",
                    category = vc_m.category
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        lark_grammar += &format!("category_responses: {category_response}\n");
        let category_messages = vc_messages
            .iter()
            .map(|vc_m| format!("response_{}: \"{}\"", vc_m.category, vc_m.message))
            .collect::<Vec<_>>()
            .join("\n");
        lark_grammar += &category_messages;
        // dbg!(&lark_grammar);
        // println!("{}", &lark_grammar);

        // let category_responses = format!("category_responses: {}", );
        Self {
            possible_vc_messages: vc_messages,
            brand_name,
            system_prompt,
            lark_grammar,
        }
    }

    pub fn get_system_prompt(&self) -> String {
        format!(
            "<|start_header_id|><|end_header_id|>{}<|eot_id|>",
            self.system_prompt
        )
    }

    pub fn get_initial_message(&self) -> String {
        "Hello! I am reaching out to you about XARELTO please feel free to ask any questions about drug samples, or dosing.".to_string()
    }
}
