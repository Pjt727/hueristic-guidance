use leptos::prelude::*;

#[component]
pub fn PromptInput(
    prompt: ReadSignal<String>,
    set_prompt: WriteSignal<String>,
    streaming: ReadSignal<bool>,
    on_generate: impl Fn() + 'static,
    on_next_token: impl Fn() + 'static,
) -> impl IntoView {
    view! {
        <div>
            <textarea
                rows="4"
                placeholder="Type your question here…"
                prop:value=move || prompt.get()
                on:input=move |ev| set_prompt.set(event_target_value(&ev))
            />
            <div>
                <button
                    disabled=move || streaming.get()
                    on:click=move |_| on_generate()
                >
                    "Generate"
                </button>
                <button
                    disabled=move || !streaming.get()
                    on:click=move |_| on_next_token()
                >
                    "Next Token"
                </button>
            </div>
        </div>
    }
}
