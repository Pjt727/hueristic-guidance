pub mod api;
pub mod components;

use inference_types::StepCandidates;
use leptos::prelude::*;

use components::{AgentSelector, CandidatePanel, PromptInput, TokenStreamView};

#[component]
pub fn App() -> impl IntoView {
    let (prompt, set_prompt) = signal(String::new());
    let (agent_id, set_agent_id) = signal::<Option<i32>>(None);
    let (status, set_status) = signal("Ready".to_string());
    let (streaming, set_streaming) = signal(false);
    let (steps, set_steps) = signal::<Vec<StepCandidates>>(vec![]);
    let (selected_idx, set_selected_idx) = signal::<Option<usize>>(None);

    let on_generate = move || {
        let p = prompt.get_untracked();
        let Some(aid) = agent_id.get_untracked() else {
            set_status.set("Select a VC agent first".to_string());
            return;
        };
        if p.trim().is_empty() {
            return;
        }
        set_steps.set(vec![]);
        set_selected_idx.set(None);
        set_status.set("Starting…".to_string());
        set_streaming.set(true);

        leptos::task::spawn_local(async move {
            match api::start_inference(p, aid).await {
                Ok(session_id) => {
                    set_status.set(format!("Streaming {session_id}"));
                    api::open_sse_stream(session_id, set_steps, set_status, set_streaming);
                }
                Err(e) => {
                    set_status.set(format!("Error: {e}"));
                    set_streaming.set(false);
                }
            }
        });
    };

    let on_next_token = move || {
        set_status.set("Step-through not yet implemented".to_string());
    };

    view! {
        <div>
            <h1>"Heuristic Guidance"</h1>

            <AgentSelector
                selected_agent_id=agent_id
                set_selected_agent_id=set_agent_id
            />

            <PromptInput
                prompt=prompt
                set_prompt=set_prompt
                streaming=streaming
                on_generate=on_generate
                on_next_token=on_next_token
            />

            <p id="status">{status}</p>

            <TokenStreamView steps=steps set_selected_idx=set_selected_idx />

            <Show when=move || selected_idx.get().is_some()>
                {move || {
                    let idx = selected_idx.get()?;
                    let step = steps.get().into_iter().nth(idx)?;
                    Some(view! { <CandidatePanel step=step /> })
                }}
            </Show>
        </div>
    }
}
