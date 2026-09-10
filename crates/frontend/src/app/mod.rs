pub mod api;
pub mod components;

use inference_types::{ClassifierMethodResult, StepCandidates};
use leptos::prelude::*;

use components::{AgentSelector, BulkTestPage, CandidatePanel, PromptInput, TokenStreamView};

#[derive(Clone, Copy, PartialEq)]
enum Page {
    Inference,
    BulkTest,
}

#[component]
pub fn App() -> impl IntoView {
    let (page, set_page) = signal(Page::Inference);

    view! {
        <div>
            // Page navigation
            <nav style="display:flex; gap:0.5rem; margin-bottom:1rem;">
                <button
                    class=move || if page.get() == Page::Inference { "btn-active" } else { "" }
                    on:click=move |_| set_page.set(Page::Inference)
                >
                    "Inference"
                </button>
                <button
                    class=move || if page.get() == Page::BulkTest { "btn-active" } else { "" }
                    on:click=move |_| set_page.set(Page::BulkTest)
                >
                    "Bulk Test"
                </button>
            </nav>

            <Show when=move || page.get() == Page::Inference>
                <InferencePage />
            </Show>
            <Show when=move || page.get() == Page::BulkTest>
                <BulkTestPage />
            </Show>
        </div>
    }
}

#[component]
fn InferencePage() -> impl IntoView {
    let (prompt, set_prompt) = signal(String::new());
    let (agent_id, set_agent_id) = signal::<Option<i32>>(None);
    let (status, set_status) = signal("Ready".to_string());
    let (streaming, set_streaming) = signal(false);
    let (steps, set_steps) = signal::<Vec<StepCandidates>>(vec![]);
    let (selected_idx, set_selected_idx) = signal::<Option<usize>>(None);

    // Classifier state
    let (classifier_methods, set_classifier_methods) = signal::<Vec<String>>(vec![]);
    let (selected_methods, set_selected_methods) = signal::<Vec<String>>(vec![]);
    let (classifier_results, set_classifier_results) =
        signal::<Vec<ClassifierMethodResult>>(vec![]);

    // Fetch available classifier methods on mount
    leptos::task::spawn_local(async move {
        if let Ok(methods) = api::fetch_classifier_methods().await {
            set_selected_methods.set(methods.clone());
            set_classifier_methods.set(methods);
        }
    });

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
        set_classifier_results.set(vec![]);
        set_status.set("Starting…".to_string());
        set_streaming.set(true);

        let p_clone = p.clone();
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

        // Also run classifiers if any are selected
        let sel = selected_methods.get_untracked();
        if !sel.is_empty() {
            leptos::task::spawn_local(async move {
                match api::classify_single(&p_clone, aid).await {
                    Ok(results) => {
                        // Filter to only the selected methods
                        let filtered: Vec<ClassifierMethodResult> = results
                            .into_iter()
                            .filter(|r| sel.contains(&r.method_name))
                            .collect();
                        set_classifier_results.set(filtered);
                    }
                    Err(_e) => {
                        // Classification failed; results stay empty
                    }
                }
            });
        }
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

            // Classifier method checkboxes
            <Show when=move || !classifier_methods.get().is_empty()>
                <div style="margin:0.5rem 0; display:flex; gap:1rem; align-items:center; flex-wrap:wrap;">
                    <span style="font-size:0.85rem; color:#aaa;">"Classifiers:"</span>
                    {move || {
                        classifier_methods.get().into_iter().map(|method| {
                            let method_clone = method.clone();
                            let method_label = method.clone();
                            let is_checked = move || selected_methods.get().contains(&method_clone);
                            let method_for_toggle = method.clone();
                            view! {
                                <label style="font-size:0.85rem; display:flex; align-items:center; gap:0.3rem; cursor:pointer;">
                                    <input
                                        type="checkbox"
                                        prop:checked=is_checked
                                        on:change=move |_| {
                                            set_selected_methods.update(|sel| {
                                                if sel.contains(&method_for_toggle) {
                                                    sel.retain(|m| m != &method_for_toggle);
                                                } else {
                                                    sel.push(method_for_toggle.clone());
                                                }
                                            });
                                        }
                                    />
                                    {method_label}
                                </label>
                            }
                        }).collect_view()
                    }}
                </div>
            </Show>

            <PromptInput
                prompt=prompt
                set_prompt=set_prompt
                streaming=streaming
                on_generate=on_generate
                on_next_token=on_next_token
            />

            <p id="status">{status}</p>

            <TokenStreamView steps=steps set_selected_idx=set_selected_idx />

            // Classifier results panel — visible when classifiers ran OR when inference has a decision step
            <Show when=move || {
                !classifier_results.get().is_empty()
                    || steps.get().iter().any(|s| !s.category_top_tokens.is_empty())
            }>
                {move || {
                    let mut results = classifier_results.get();

                    // Build LLM row from the first decision step (non-empty category_top_tokens)
                    let llm_row: Option<ClassifierMethodResult> = steps.get()
                        .iter()
                        .find(|s| !s.category_top_tokens.is_empty())
                        .and_then(|step| {
                            let mut cat_scores: Vec<(String, f32)> = step.category_top_tokens.iter()
                                .map(|ct| {
                                    let combined = ct.best_token.logit + ct.best_token.embedding_logit;
                                    (ct.category_name.clone(), combined)
                                })
                                .collect();
                            cat_scores.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            if cat_scores.is_empty() {
                                return None;
                            }
                            // Margin sigmoid: same formula as the classifiers crate.
                            // k=20 → margin 0.05 ≈ 73%, 0.10 ≈ 88%.
                            let confidence = if cat_scores.len() < 2 {
                                1.0_f32
                            } else {
                                let margin = cat_scores[0].1 - cat_scores[1].1;
                                1.0 / (1.0 + (-20.0_f32 * margin).exp())
                            };
                            let chosen = cat_scores[0].0.clone();
                            // Store softmax probabilities in scores for reference.
                            let max_s = cat_scores[0].1;
                            let exps: Vec<f32> = cat_scores.iter().map(|(_, s)| (s - max_s).exp()).collect();
                            let sum: f32 = exps.iter().sum();
                            let scores = cat_scores.into_iter().enumerate()
                                .map(|(i, (name, _))| inference_types::ClassifierScore {
                                    category_name: name,
                                    score: if sum == 0.0 { 0.0 } else { (exps[i] / sum) as f64 },
                                })
                                .collect();
                            Some(ClassifierMethodResult {
                                method_name: "llm".to_string(),
                                chosen_category: chosen,
                                confidence: confidence as f64,
                                latency_ms: 0,
                                scores,
                            })
                        });

                    if let Some(llm) = llm_row {
                        results.insert(0, llm);
                    }

                    if results.is_empty() {
                        return None;
                    }

                    // Detect disagreements across all rows
                    let chosen_cats: Vec<String> = results.iter().map(|r| r.chosen_category.clone()).collect();
                    let first_cat = chosen_cats.first().cloned().unwrap_or_default();
                    let has_disagreement = chosen_cats.iter().any(|c| c != &first_cat);

                    let header_style = "background:#1e1e1e; text-align:left; padding:4px 10px; font-size:0.82rem; color:#aaa;";
                    let cell_style_base = "padding:4px 10px; font-size:0.82rem; font-family:monospace;";

                    Some(view! {
                        <div style="margin-top:1rem;">
                            <h3 style="font-size:0.9rem; margin-bottom:0.4rem; color:#aaa;">"Classifier Results"</h3>
                            <table style="border-collapse:collapse; width:auto; min-width:400px;">
                                <thead>
                                    <tr>
                                        <th style=header_style>"Method"</th>
                                        <th style=header_style>"Chosen Category"</th>
                                        <th style=header_style>"Confidence"</th>
                                        <th style=header_style>"Latency (ms)"</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {results.into_iter().map(|r| {
                                        let disagrees = has_disagreement && r.chosen_category != first_cat;
                                        let cat_style = if disagrees {
                                            format!("{cell_style_base} color:#ff9800; font-weight:bold;")
                                        } else {
                                            cell_style_base.to_string()
                                        };
                                        let conf_pct = format!("{:.1}%", r.confidence * 100.0);
                                        let latency = if r.method_name == "llm" {
                                            "—".to_string()
                                        } else {
                                            r.latency_ms.to_string()
                                        };
                                        view! {
                                            <tr style="border-bottom:1px solid #2a2a2a;">
                                                <td style=cell_style_base>{r.method_name}</td>
                                                <td style=cat_style>{r.chosen_category}</td>
                                                <td style=cell_style_base>{conf_pct}</td>
                                                <td style=cell_style_base>{latency}</td>
                                            </tr>
                                        }
                                    }).collect_view()}
                                </tbody>
                            </table>
                            {if has_disagreement {
                                Some(view! {
                                    <p style="font-size:0.78rem; color:#ff9800; margin-top:0.3rem;">
                                        "Methods disagree on category."
                                    </p>
                                })
                            } else {
                                None
                            }}
                        </div>
                    })
                }}
            </Show>

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
