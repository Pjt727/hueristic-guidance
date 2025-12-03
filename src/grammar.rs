use serde::{Deserialize, Serialize};

use crate::llama_tokenizer::{END_TURN_TOKEN, ID_END_TOKEN, ID_START_TOKEN};

// lama 4

const STARTING_CONVO_TEMPLATE: &str = r#"
You are a virtual Coordinator chatbot providing information about {brandname}.
You interact with healthcare providers who ask questions about the brand.
You must always respond with text directly from the approved list.

---

## **Response rules**

1. Respond with the most appropiate response from the **Approved VC Responses**
2. You must choose the response from the approved list.
3. You must only respond with the chosen response, do not include any other text.

---

### **Approved Responses List**

Category: samples
Samples for {brandname} are available at the closest store.

Category: dosage
Dosage information for {brandname} is available on the back of the bottle.

---

### **Example Conversation**

The healhcare providers says:

Can you send me sample info?

The virtual coordinator responds:

Category: samples
Samples for {brandname} are available at the closest store.


The healhcare providers says:

dosage please

The virtual coordinator responds:

Category: dosage
Dosage information for {brandname} is available on the back of the bottle.
"#;

const STARTING_CONVO_GRAMMAR_TEMPLATE: &str = r#"
start: "Category " reponse_option "\n"
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
        // let category_response = vc_messages
        //     .iter()
        //     .map(|vc_m| {
        //         format!(
        //             "\"{category}\\n\" response_{category}",
        //             category = vc_m.category,
        //         )
        //     })
        //     .collect::<Vec<_>>()
        //     .join(" | ");
        let response_options = vc_messages
            .iter()
            .map(|vc_m| "\"".to_owned() + &vc_m.category + "\"")
            .collect::<Vec<_>>()
            .join(" | ");
        lark_grammar += &format!("reponse_option[capture]: {response_options}");
        // lark_grammar += &format!("category_responses: {category_response}\n");
        // let category_messages = vc_messages
        //     .iter()
        //     .map(|vc_m| format!("response_{}: \"{}\"", vc_m.category, vc_m.message))
        //     .collect::<Vec<_>>()
        //     .join("\n");
        // lark_grammar += &category_messages;
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
