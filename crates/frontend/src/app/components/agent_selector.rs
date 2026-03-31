use leptos::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlSelectElement;

use crate::app::api;

/// Themed agent dropdown + "System Prompt" button with modal preview.
/// Fetches available agent IDs from GET /agents on mount.
#[component]
pub fn AgentSelector(
    selected_agent_id: ReadSignal<Option<i32>>,
    set_selected_agent_id: WriteSignal<Option<i32>>,
) -> impl IntoView {
    let (agent_ids, set_agent_ids) = signal::<Vec<i32>>(vec![]);
    let (load_error, set_load_error) = signal::<Option<String>>(None);
    let (show_modal, set_show_modal) = signal(false);
    let (system_prompt_text, set_system_prompt_text) = signal::<Option<String>>(None);
    let (prompt_loading, set_prompt_loading) = signal(false);

    // Fetch agent list once on mount
    leptos::task::spawn_local(async move {
        match api::fetch_agents().await {
            Ok(ids) => {
                if let Some(&first) = ids.first() {
                    set_selected_agent_id.set(Some(first));
                }
                set_agent_ids.set(ids);
            }
            Err(e) => set_load_error.set(Some(e)),
        }
    });

    let open_prompt_modal = move |_| {
        let Some(aid) = selected_agent_id.get_untracked() else {
            return;
        };
        set_show_modal.set(true);
        set_system_prompt_text.set(None);
        set_prompt_loading.set(true);
        leptos::task::spawn_local(async move {
            match api::fetch_system_prompt(aid).await {
                Ok(text) => set_system_prompt_text.set(Some(text)),
                Err(e) => set_system_prompt_text.set(Some(format!("Error: {e}"))),
            }
            set_prompt_loading.set(false);
        });
    };

    let close_modal = move |_| set_show_modal.set(false);

    view! {
        <div class="agent-selector">
            <label for="agent-select">"VC Agent"</label>

            <Show when=move || !agent_ids.get().is_empty()>
                <select
                    id="agent-select"
                    on:change=move |ev| {
                        let val = ev
                            .target()
                            .and_then(|t| t.dyn_into::<HtmlSelectElement>().ok())
                            .and_then(|s| s.value().parse::<i32>().ok());
                        set_selected_agent_id.set(val);
                        set_system_prompt_text.set(None);
                    }
                >
                    {move || {
                        agent_ids
                            .get()
                            .into_iter()
                            .map(|id| view! { <option value=id.to_string()>{format!("Agent {id}")}</option> })
                            .collect_view()
                    }}
                </select>
            </Show>

            <Show when=move || agent_ids.get().is_empty() && load_error.get().is_none()>
                <span class="muted">"Loading…"</span>
            </Show>

            <Show when=move || load_error.get().is_some()>
                <span class="error">
                    "Failed to load agents: "
                    {move || load_error.get().unwrap_or_default()}
                </span>
            </Show>

            <Show when=move || selected_agent_id.get().is_some()>
                <button class="btn-ghost" on:click=open_prompt_modal>
                    "System Prompt"
                </button>
            </Show>
        </div>

        // Modal
        <Show when=move || show_modal.get()>
            <div
                class="modal-overlay"
                on:click=move |ev| {
                    // Close when clicking the backdrop (not the modal content)
                    if ev
                        .target()
                        .and_then(|t| t.dyn_into::<web_sys::Element>().ok())
                        .map(|el| el.class_name() == "modal-overlay")
                        .unwrap_or(false)
                    {
                        set_show_modal.set(false);
                    }
                }
            >
                <div class="modal">
                    <div class="modal-header">
                        <h2>
                            "System Prompt — Agent "
                            {move || selected_agent_id.get().unwrap_or(0)}
                        </h2>
                        <button class="modal-close" on:click=close_modal>
                            "✕ Close"
                        </button>
                    </div>

                    <Show when=move || prompt_loading.get()>
                        <p class="muted">"Loading…"</p>
                    </Show>

                    <Show when=move || !prompt_loading.get()>
                        <pre>{move || system_prompt_text.get().unwrap_or_default()}</pre>
                    </Show>
                </div>
            </div>
        </Show>
    }
}
