use inference_types::StepCandidates;
use leptos::prelude::*;

/// Color the token chip based on its probability (green = high, red = low).
fn prob_color(prob: f32) -> String {
    let r = ((1.0 - prob) * 255.0) as u8;
    let g = (prob * 255.0) as u8;
    format!("rgb({r},{g},60)")
}

#[component]
pub fn TokenStreamView(
    steps: ReadSignal<Vec<StepCandidates>>,
    set_selected_idx: WriteSignal<Option<usize>>,
) -> impl IntoView {
    view! {
        <div id="token-stream">
            {move || {
                steps.get()
                    .into_iter()
                    .enumerate()
                    .map(|(i, step)| {
                        let is_ff = step.top_alternatives.is_empty();
                        let color = if is_ff {
                            "#555".to_string()
                        } else {
                            prob_color(step.chosen.probability)
                        };
                        let text = step.chosen.text.clone();
                        let prob = step.chosen.probability;
                        let logit = step.chosen.logit;
                        let title = if is_ff {
                            "grammar forced".to_string()
                        } else {
                            format!("p={prob:.4} logit={logit:.4}")
                        };
                        view! {
                            <span
                                class="token-chip"
                                title=title
                                style=format!("border-color:{color}")
                                on:click=move |_| set_selected_idx.set(Some(i))
                            >
                                {text}
                            </span>
                        }
                    })
                    .collect_view()
            }}
        </div>
    }
}
