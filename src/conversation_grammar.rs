use serde::{Deserialize, Serialize};

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

const STARTING_CONVO_GRAMMAR: &str = r#"start: initial_message hcp_response
initial_message: <|start_header_id|> "VC" <|end_header_id|> /(.|\n)*/ <|eot_id|>
hcp_response: <|start_header_id|> "VC" <|end_header_id|> hcp_content <|eot_id|>
hcp_content: /(.|\n){hcplimit}/
vc_response: <|start_header_id|> "VC" <|end_header_id|> "Category: " category_responses <|eot_id|> (hcp_response)?

// categories get added
"#;

#[derive(Debug, Serialize, Deserialize)]
pub struct VCmessage {
    pub category: String,
    pub message: String,
}

pub struct Conversation {
    possible_vc_messages: Vec<VCmessage>,
    brand_name: String,
    system_prompt: String,
    lark_grammar: String,
}

impl Conversation {
    pub fn new(brand_name: String, vc_messages: Vec<VCmessage>) -> Self {
        let system_prompt = STARTING_CONVO_TEMPLATE.replace("{brandname}", &brand_name);
        let mut lark_grammar = STARTING_CONVO_GRAMMAR.to_string();
        let category_response = vc_messages
            .iter()
            .map(|vc_m| {
                format!(
                    "\"{category}\n\" response_{category}",
                    category = vc_m.category
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        lark_grammar += &format!("category_responses: {category_response}");
        let category_messages = vc_messages
            .iter()
            .map(|vc_m| format!("response_{}: \"{}\"", vc_m.category, vc_m.message))
            .collect::<Vec<_>>()
            .join("\n");
        lark_grammar += &category_messages;

        // let category_responses = format!("category_responses: {}", );
        Self {
            possible_vc_messages: vc_messages,
            brand_name,
            system_prompt,
            lark_grammar,
        }
    }
}
