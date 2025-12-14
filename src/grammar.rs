use serde::{Deserialize, Serialize};

use crate::llama_tokenizer::{END_TURN_TOKEN, ID_END_TOKEN, ID_START_TOKEN};

// lama 4

const STARTING_CONVO_TEMPLATE: &str = r#"
You are a virtual Coordinator chatbot providing information about {brandname}.
You interact with healthcare providers who ask questions about the brand.
You must always respond with text directly from the approved list.

---

## **Response rules**

1. Respond with the most appropriate response from the **Approved VC Responses**
2. You must choose the response from the approved list.
3. You must only respond with the chosen response, do not include any other text.
4. Match the HCP's question to the appropriate category based on the description.

---

### **Approved Responses List**

{approved_responses}

---

### **Important Output Formatting**

Only respond in the following format:

Category: NAME OF CATEGORY HERE

VC MESSAGE HERE
"#;

const STARTING_CONVO_GRAMMAR_TEMPLATE: &str = r#"
start: "Category: " reponse_option "\n"
// brand specific category information
"#;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VCmessage {
    pub category: String,
    pub kind: String,
    pub description: String,
    pub mlr_message: String, // Message with placeholder URLs like [https://docupdate.io/x1]
    pub message: String,     // Message with actual URLs
}

pub struct GrammarFlow {
    pub possible_vc_messages: Vec<VCmessage>,
    pub brand_name: String,
    pub system_prompt: String,
    pub lark_grammar: String,
}

impl GrammarFlow {
    pub fn new(brand_name: String, vc_messages: Vec<VCmessage>) -> Self {
        // Generate approved responses list with descriptions
        let approved_responses = vc_messages
            .iter()
            .map(|vc_m| {
                format!(
                    "Category: {}\nDescription: {}\nResponse: {}",
                    vc_m.category, vc_m.description, vc_m.message
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        // Replace placeholders in template
        let system_prompt = STARTING_CONVO_TEMPLATE
            .replace("{brandname}", &brand_name)
            .replace("{approved_responses}", &approved_responses);

        // Build Lark grammar for response validation
        let mut lark_grammar = STARTING_CONVO_GRAMMAR_TEMPLATE.replace("{hcplimit}", "{0,50}");
        let response_options = vc_messages
            .iter()
            .map(|vc_m| "\"".to_owned() + &vc_m.category + "\"")
            .collect::<Vec<_>>()
            .join(" | ");
        lark_grammar += &format!("reponse_option[capture]: {response_options}");

        Self {
            possible_vc_messages: vc_messages,
            brand_name,
            system_prompt,
            lark_grammar,
        }
    }

    fn generate_sample_question(description: &str, index: usize) -> String {
        // Simplify description to make a natural question
        let simplified = description
            .replace("HCP is asking for information about ", "")
            .replace("HCP requests information about ", "")
            .replace("HCP is asking about ", "")
            .replace("HCP requests ", "")
            .replace("HCPs requesting information about ", "")
            .replace("HCP asks about ", "")
            .to_lowercase();

        // Generate varied sample questions based on the description
        match index % 4 {
            0 => format!("Can you help me with {}?", simplified),
            1 => format!("I need information about {}", simplified),
            2 => format!("Tell me about {}", simplified),
            _ => format!("What do you have on {}?", simplified),
        }
    }

    pub fn get_system_prompt(&self) -> String {
        let prompt = format!(
            "{ID_START_TOKEN}system{ID_END_TOKEN}{}{END_TURN_TOKEN}",
            self.system_prompt
        );
        dbg!(&prompt);
        prompt
    }

    pub fn get_initial_message(&self) -> String {
        todo!();
        "Hello! I am reaching out to you about XARELTO please feel free to ask any questions about drug samples, or dosing.".to_string()
    }
}
