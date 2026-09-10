use inference_types::AgentInfo;
use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::app::api;


/// Live-search agent selector.
/// Fetches all agents on mount, then filters by the search input.
/// Shows up to 10 results in a scrollable list.
#[component]
pub fn AgentSelector(
    selected_agent_id: ReadSignal<Option<i32>>,
    set_selected_agent_id: WriteSignal<Option<i32>>,
) -> impl IntoView {
    let (all_agents, set_all_agents) = signal::<Vec<AgentInfo>>(vec![]);
    let (load_error, set_load_error) = signal::<Option<String>>(None);
    let (agents_loaded, set_agents_loaded) = signal(false);
    let (query, set_query) = signal(String::new());
    let (show_list, set_show_list) = signal(false);
    let (show_modal, set_show_modal) = signal(false);
    let (system_prompt_text, set_system_prompt_text) = signal::<Option<String>>(None);
    let (prompt_loading, set_prompt_loading) = signal(false);

    // Fetch full agent list once on mount
    leptos::task::spawn_local(async move {
        match api::fetch_agents().await {
            Ok(agents) => {
                if let Some(first) = agents.first() {
                    set_selected_agent_id.set(Some(first.id));
                    set_query.set(first.name.clone());
                }
                set_all_agents.set(agents);
            }
            Err(e) => set_load_error.set(Some(e)),
        }
        set_agents_loaded.set(true);
    });

    // Filtered list — all matching agents; CSS scroll handles overflow
    let filtered = move || {
        let q = query.get().to_lowercase();
        all_agents
            .get()
            .into_iter()
            .filter(|a| a.name.to_lowercase().contains(&q) || a.id.to_string().contains(&q))
            .collect::<Vec<_>>()
    };

    // Name of currently selected agent (for modal title)
    let selected_name = move || {
        let id = selected_agent_id.get()?;
        all_agents.get().into_iter().find(|a| a.id == id).map(|a| a.name)
    };

    let open_prompt_modal = move |_| {
        let Some(aid) = selected_agent_id.get_untracked() else { return };
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

    view! {
        <div class="agent-selector">
            <label>"VC Agent"</label>

            // Search input
            <div style="position:relative; display:inline-block;">
                <Show when=move || !agents_loaded.get()>
                    <input
                        type="text"
                        placeholder="Loading…"
                        disabled=true
                        style="width:220px;"
                    />
                </Show>

                <Show when=move || agents_loaded.get() && load_error.get().is_none()>
                    <input
                        type="text"
                        placeholder="Search agents…"
                        style="width:220px;"
                        prop:value=move || query.get()
                        on:input=move |ev| {
                            let v = event_target_value(&ev);
                            set_query.set(v);
                            set_show_list.set(true);
                        }
                        on:focus=move |_| set_show_list.set(true)
                        on:blur=move |_| {
                            // Small delay so click on list item fires first
                            leptos::task::spawn_local(async move {
                                crate::app::components::agent_selector::sleep_ms(150).await;
                                set_show_list.set(false);
                            });
                        }
                    />

                    // Dropdown list
                    <Show when=move || show_list.get() && !filtered().is_empty()>
                        <ul style="
                            position:absolute; z-index:100; left:0; top:100%;
                            margin:0; padding:0; list-style:none;
                            background:#1e1e2e; border:1px solid #444;
                            width:220px; max-height:260px; overflow-y:auto;
                        ">
                            {move || filtered().into_iter().map(|agent| {
                                let id = agent.id;
                                let name = agent.name.clone();
                                let is_selected = move || selected_agent_id.get() == Some(id);
                                view! {
                                    <li
                                        style=move || format!(
                                            "padding:6px 10px; cursor:pointer; font-size:0.9em; {}",
                                            if is_selected() { "background:#313244; font-weight:bold;" } else { "" }
                                        )
                                        on:mousedown=move |_| {
                                            set_selected_agent_id.set(Some(id));
                                            set_query.set(name.clone());
                                            set_show_list.set(false);
                                        }
                                    >
                                        {agent.name.clone()}
                                        <span style="color:#888; font-size:0.8em; margin-left:6px;">
                                            {format!("#{id}")}
                                        </span>
                                    </li>
                                }
                            }).collect_view()}
                        </ul>
                    </Show>
                </Show>

                <Show when=move || agents_loaded.get() && load_error.get().is_none() && all_agents.get().is_empty()>
                    <span class="error">"No agents found"</span>
                </Show>
            </div>

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

        // System prompt modal
        <Show when=move || show_modal.get()>
            <div
                class="modal-overlay"
                on:click=move |ev| {
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
                            "System Prompt — "
                            {move || selected_name().unwrap_or_default()}
                        </h2>
                        <button class="modal-close" on:click=move |_| set_show_modal.set(false)>
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

/// Async sleep using a JS Promise — needed in WASM (no tokio::time).
pub async fn sleep_ms(ms: u32) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms as i32)
            .unwrap();
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}
