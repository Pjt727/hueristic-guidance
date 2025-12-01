use serde::{Deserialize, Serialize};

// lama 3
// pub const ID_START_TOKEN: &str = "<|start_header_id|>";
// pub const ID_END_TOKEN: &str = "<|end_header_id|>";
// pub const END_TURN_TOKEN_TOKEN: &str = "<|eot_id|>";

// lama 4
pub const ID_START_TOKEN: &str = "<|header_start|>";
pub const ID_END_TOKEN: &str = "<|header_end|>";
pub const END_TURN_TOKEN_TOKEN: &str = "<|eot|>";
pub const SPECIAL_TOKENS: [&str; 3] = [ID_START_TOKEN, ID_END_TOKEN, END_TURN_TOKEN_TOKEN];

const STARTING_CONVO_TEMPLATE: &str = r#"
You are a virtual Coordinator chatbot providing information about {brandname}.
You interact with healthcare providers (HCP's) who ask questions about the brand.
You must always respond with text directly from the approved list

---

## **Response rules**

1. Respond with the most appropiate response from the **Approved VC Responses**
2. You must **choose the correct category** from the approved list.
3. Before the message text, you must output:

   Category: <chosen_category>

---

### **Approved VC Responses:**

Category: samples  
Samples for {brandname} are available at the closest store.

Category: dosing  
*Dosage information for {brandname} is available on the back of the bottle.*

---

### **Example Conversation**

HCP:
```
Can you send me sample info?
```

VC:
```
Samples for {brandname} are available at the closest store.
```

HCP:
```
dosage please
```

VC:
```
Dosage information for {brandname} is available on the back of the bottle.
```
"#;

const STARTING_CONVO_GRAMMAR_TEMPLATE: &str = r#"start: hcp_response
hcp_response: {startheader} "user" {endheader} hcp_content{hcplimit} {endturn} vc_response
hcp_content: <[0-128002]>
vc_response: {startheader} "assistant" {endheader} "Category: " category_responses {endturn} hcp_response

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
        lark_grammar = lark_grammar.replace("{startheader}", ID_START_TOKEN);
        lark_grammar = lark_grammar.replace("{endheader}", ID_END_TOKEN);
        lark_grammar = lark_grammar.replace("{endturn}", END_TURN_TOKEN_TOKEN);
        let category_response = vc_messages
            .iter()
            .map(|vc_m| {
                format!(
                    "\"{category}\\n\" response_{category}",
                    category = vc_m.category,
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
        todo!();
        "Hello! I am reaching out to you about XARELTO please feel free to ask any questions about drug samples, or dosing.".to_string()
    }
}
