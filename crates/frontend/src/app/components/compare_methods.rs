use std::collections::HashMap;

use leptos::prelude::*;

use crate::app::api;
use crate::app::components::AgentSelector;

/// Sleep for `ms` milliseconds using browser setTimeout.
async fn sleep_ms(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms)
            .unwrap();
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// A single classification result from one method.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct MethodResult {
    pub method_name: String,
    pub chosen_category: String,
    pub confidence: f64,
    pub latency_ms: u64,
    pub scores: Vec<ScoreEntry>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ScoreEntry {
    pub category_name: String,
    pub score: f64,
}

/// One row from the comparison results endpoint.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct CompareResultRow {
    pub example_id: i64,
    pub example_text: String,
    pub correct_categories: Vec<String>,
    pub results: HashMap<String, MethodResult>,
}

/// Summary of a comparison run.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct CompareRunSummary {
    pub id: i64,
    pub agent_id: i64,
    pub methods: String,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub total: Option<i64>,
    pub summary: Option<String>,
}

/// Single-query classify response.
#[derive(Clone, Debug, serde::Deserialize)]
struct ClassifyResponse {
    results: Vec<MethodResult>,
}

#[component]
pub fn CompareMethodsPage() -> impl IntoView {
    let (agent_id, set_agent_id) = signal::<Option<i32>>(None);
    let (status, set_status) = signal("Select an agent and run a comparison".to_string());
    let (running, set_running) = signal(false);
    let (results, set_results) = signal::<Vec<CompareResultRow>>(vec![]);
    let (current_run_id, set_current_run_id) = signal::<Option<i64>>(None);
    let (selected_idx, set_selected_idx) = signal::<Option<usize>>(None);

    // Single query test
    let (test_prompt, set_test_prompt) = signal(String::new());
    let (test_results, set_test_results) = signal::<Vec<MethodResult>>(vec![]);
    let (test_running, set_test_running) = signal(false);

    // Past runs
    let (past_runs, set_past_runs) = signal::<Vec<CompareRunSummary>>(vec![]);
    leptos::task::spawn_local(async move {
        match api::fetch_compare_runs().await {
            Ok(runs) => set_past_runs.set(runs),
            Err(_) => {}
        }
    });

    // Run comparison
    let on_run = move || {
        let Some(aid) = agent_id.get_untracked() else {
            set_status.set("Select a VC agent first".to_string());
            return;
        };
        set_results.set(vec![]);
        set_selected_idx.set(None);
        set_status.set("Starting comparison...".to_string());
        set_running.set(true);

        leptos::task::spawn_local(async move {
            match api::start_comparison(aid).await {
                Ok(run_id) => {
                    set_current_run_id.set(Some(run_id));
                    set_status.set(format!("Running comparison (run {run_id})..."));
                    // Poll for results (comparison runs are fast, no SSE needed)
                    loop {
                        sleep_ms(2000).await;
                        match api::fetch_compare_results(run_id).await {
                            Ok(r) if !r.is_empty() => {
                                let n = r.len();
                                set_results.set(r);
                                set_status.set(format!("Comparison complete — {n} examples"));
                                set_running.set(false);
                                // Refresh past runs list
                                if let Ok(runs) = api::fetch_compare_runs().await {
                                    set_past_runs.set(runs);
                                }
                                break;
                            }
                            Ok(_) => {} // not done yet
                            Err(e) => {
                                set_status.set(format!("Error: {e}"));
                                set_running.set(false);
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    set_status.set(format!("Error: {e}"));
                    set_running.set(false);
                }
            }
        });
    };

    // Single query classify
    let on_test = move || {
        let Some(aid) = agent_id.get_untracked() else { return; };
        let prompt = test_prompt.get_untracked();
        if prompt.trim().is_empty() { return; }
        set_test_running.set(true);
        set_test_results.set(vec![]);
        leptos::task::spawn_local(async move {
            match api::classify_single(&prompt, aid).await {
                Ok(r) => set_test_results.set(r),
                Err(e) => set_status.set(format!("Classify error: {e}")),
            }
            set_test_running.set(false);
        });
    };

    view! {
        <div style="max-width:1200px;">
            <h1>"Compare Classification Methods"</h1>

            <AgentSelector
                selected_agent_id=agent_id
                set_selected_agent_id=set_agent_id
            />

            // ── Single query test ──────────────────────────────────────
            <div style="margin-top:1rem; padding:0.75rem; background:#1a1a2e; border-radius:4px;">
                <h3 style="margin:0 0 0.5rem; font-size:0.95rem; color:#aaa;">"Quick Classify"</h3>
                <div style="display:flex; gap:0.5rem; align-items:flex-start;">
                    <textarea
                        style="flex:1; min-height:50px; resize:vertical;"
                        placeholder="Type a query to classify with all methods..."
                        prop:value=move || test_prompt.get()
                        on:input=move |e| set_test_prompt.set(event_target_value(&e))
                    />
                    <button
                        disabled=move || test_running.get() || agent_id.get().is_none()
                        on:click=move |_| on_test()
                    >
                        {move || if test_running.get() { "Classifying..." } else { "Classify" }}
                    </button>
                </div>

                <Show when=move || !test_results.get().is_empty()>
                    <table style="width:100%; border-collapse:collapse; margin-top:0.5rem; font-size:0.85rem;">
                        <thead>
                            <tr style="background:#1e1e1e;">
                                <th style="text-align:left; padding:3px 8px">"Method"</th>
                                <th style="text-align:left; padding:3px 8px">"Chosen Category"</th>
                                <th style="text-align:right; padding:3px 8px">"Confidence"</th>
                                <th style="text-align:right; padding:3px 8px">"Latency"</th>
                            </tr>
                        </thead>
                        <tbody>
                            {move || test_results.get().into_iter().map(|r| {
                                let conf_pct = format!("{:.1}%", r.confidence * 100.0);
                                let lat = format!("{}ms", r.latency_ms);
                                view! {
                                    <tr style="border-bottom:1px solid #2a2a2a;">
                                        <td style="padding:3px 8px; font-family:monospace; color:#7eb8da;">{r.method_name}</td>
                                        <td style="padding:3px 8px; font-family:monospace;">{r.chosen_category}</td>
                                        <td style="padding:3px 8px; text-align:right; font-family:monospace;">{conf_pct}</td>
                                        <td style="padding:3px 8px; text-align:right; font-family:monospace; color:#888;">{lat}</td>
                                    </tr>
                                }
                            }).collect_view()}
                        </tbody>
                    </table>
                </Show>
            </div>

            // ── Bulk comparison ────────────────────────────────────────
            <div style="margin-top:1rem; display:flex; gap:0.5rem; align-items:center;">
                <button
                    disabled=move || running.get() || agent_id.get().is_none()
                    on:click=move |_| on_run()
                >
                    {move || if running.get() { "Running..." } else { "Run Bulk Comparison" }}
                </button>
            </div>

            <p id="status">{status}</p>

            // ── Previous runs ──────────────────────────────────────────
            <Show when=move || !past_runs.get().is_empty()>
                <details style="margin-top:0.5rem;">
                    <summary style="cursor:pointer; font-size:0.9rem; color:#aaa; user-select:none;">
                        "Previous comparison runs"
                    </summary>
                    <table style="width:100%; border-collapse:collapse; margin-top:0.4rem; font-size:0.85rem;">
                        <thead>
                            <tr style="background:#1e1e1e;">
                                <th style="text-align:left; padding:3px 6px">"Agent"</th>
                                <th style="text-align:left; padding:3px 6px">"Started"</th>
                                <th style="text-align:left; padding:3px 6px">"Methods"</th>
                                <th style="text-align:right; padding:3px 6px">"Examples"</th>
                                <th style="padding:3px 6px"></th>
                            </tr>
                        </thead>
                        <tbody>
                            {move || past_runs.get().into_iter().map(|run| {
                                let started = run.started_at.get(..16).unwrap_or(&run.started_at).to_string();
                                let total_label = run.total.map(|t| t.to_string()).unwrap_or("—".to_string());
                                let run_id = run.id;
                                // Parse methods JSON for display
                                let methods_display: String = serde_json::from_str::<Vec<String>>(&run.methods)
                                    .map(|m| m.join(", "))
                                    .unwrap_or(run.methods.clone());
                                view! {
                                    <tr style="border-bottom:1px solid #2a2a2a;">
                                        <td style="padding:3px 6px; font-family:monospace;">{run.agent_id}</td>
                                        <td style="padding:3px 6px; color:#aaa;">{started}</td>
                                        <td style="padding:3px 6px; font-family:monospace; font-size:0.8rem; color:#7eb8da;">{methods_display}</td>
                                        <td style="padding:3px 6px; text-align:right; font-family:monospace;">{total_label}</td>
                                        <td style="padding:3px 6px;">
                                            <button
                                                style="font-size:0.8rem; padding:1px 8px;"
                                                on:click=move |_| {
                                                    set_results.set(vec![]);
                                                    set_selected_idx.set(None);
                                                    set_current_run_id.set(Some(run_id));
                                                    set_status.set(format!("Loading run {run_id}..."));
                                                    leptos::task::spawn_local(async move {
                                                        match api::fetch_compare_results(run_id).await {
                                                            Ok(r) => {
                                                                let n = r.len();
                                                                set_results.set(r);
                                                                set_status.set(format!("Loaded run {run_id} — {n} examples"));
                                                            }
                                                            Err(e) => set_status.set(format!("Error: {e}")),
                                                        }
                                                    });
                                                }
                                            >
                                                "Load"
                                            </button>
                                        </td>
                                    </tr>
                                }
                            }).collect_view()}
                        </tbody>
                    </table>
                </details>
            </Show>

            // ── Summary table ──────────────────────────────────────────
            <Show when=move || !results.get().is_empty()>
                {move || {
                    let rs = results.get();
                    let total = rs.len();

                    // Collect all method names from results
                    let mut method_names: Vec<String> = rs.iter()
                        .flat_map(|r| r.results.keys().cloned())
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    method_names.sort();

                    // Compute per-method stats
                    let stats: Vec<(String, usize, f64, f64)> = method_names.iter().map(|method| {
                        let mut correct = 0usize;
                        let mut total_conf = 0.0f64;
                        let mut total_lat = 0.0f64;
                        let mut count = 0usize;
                        for r in &rs {
                            if let Some(mr) = r.results.get(method) {
                                count += 1;
                                total_conf += mr.confidence;
                                total_lat += mr.latency_ms as f64;
                                if r.correct_categories.contains(&mr.chosen_category) {
                                    correct += 1;
                                }
                            }
                        }
                        let avg_conf = if count == 0 { 0.0 } else { total_conf / count as f64 };
                        let avg_lat = if count == 0 { 0.0 } else { total_lat / count as f64 };
                        (method.clone(), correct, avg_conf, avg_lat)
                    }).collect();

                    // Find best accuracy
                    let best_acc = stats.iter().map(|(_, c, _, _)| *c).max().unwrap_or(0);

                    view! {
                        <div style="margin-top:1rem;">
                            <h3 style="margin:0 0 0.5rem; font-size:1rem;">"Method Accuracy Summary"</h3>
                            <table style="border-collapse:collapse; font-size:0.85rem; margin-bottom:1rem;">
                                <thead>
                                    <tr style="background:#1e1e1e;">
                                        <th style="text-align:left; padding:4px 12px">"Method"</th>
                                        <th style="text-align:right; padding:4px 12px">"Accuracy"</th>
                                        <th style="text-align:right; padding:4px 12px">"Avg Confidence"</th>
                                        <th style="text-align:right; padding:4px 12px">"Avg Latency"</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {stats.into_iter().map(|(method, correct, avg_conf, avg_lat)| {
                                        let acc_pct = if total == 0 { 0.0 } else { correct as f64 / total as f64 * 100.0 };
                                        let is_best = correct == best_acc;
                                        let color = if is_best { "#4caf50" } else { "#ccc" };
                                        view! {
                                            <tr style="border-bottom:1px solid #2a2a2a;">
                                                <td style="padding:4px 12px; font-family:monospace; color:#7eb8da;">{method}</td>
                                                <td style=format!("padding:4px 12px; text-align:right; font-family:monospace; font-weight:bold; color:{color};")>
                                                    {format!("{correct}/{total} ({acc_pct:.0}%)")}
                                                </td>
                                                <td style="padding:4px 12px; text-align:right; font-family:monospace; color:#aaa;">
                                                    {format!("{:.1}%", avg_conf * 100.0)}
                                                </td>
                                                <td style="padding:4px 12px; text-align:right; font-family:monospace; color:#888;">
                                                    {format!("{avg_lat:.0}ms")}
                                                </td>
                                            </tr>
                                        }
                                    }).collect_view()}
                                </tbody>
                            </table>
                        </div>
                    }
                }}
            </Show>

            // ── Per-example results table ──────────────────────────────
            <Show when=move || !results.get().is_empty()>
                {move || {
                    let rs = results.get();
                    // Collect method names
                    let mut method_names: Vec<String> = rs.iter()
                        .flat_map(|r| r.results.keys().cloned())
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    method_names.sort();

                    view! {
                        <table style="width:100%; border-collapse:collapse; margin-top:0.5rem; font-size:0.85rem;">
                            <thead>
                                <tr style="background:#2a2a2a;">
                                    <th style="text-align:left; padding:4px 6px">"#"</th>
                                    <th style="text-align:left; padding:4px 6px; max-width:250px;">"Example"</th>
                                    {method_names.iter().map(|m| {
                                        let name = m.clone();
                                        view! {
                                            <th style="text-align:left; padding:4px 6px; color:#7eb8da;">{name}</th>
                                        }
                                    }).collect_view()}
                                    <th style="text-align:left; padding:4px 6px; color:#aaa;">"Correct"</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rs.into_iter().enumerate().map(|(i, r)| {
                                    let is_selected = selected_idx.get() == Some(i);
                                    let preview: String = r.example_text.chars().take(60).collect();
                                    let preview = if r.example_text.len() > 60 { format!("{preview}...") } else { preview };

                                    // Check if all methods agree
                                    let choices: Vec<String> = r.results.values().map(|mr| mr.chosen_category.clone()).collect();
                                    let all_agree = choices.windows(2).all(|w| w[0] == w[1]);
                                    let any_correct = r.results.values().any(|mr| r.correct_categories.contains(&mr.chosen_category));
                                    let all_correct = r.results.values().all(|mr| r.correct_categories.contains(&mr.chosen_category));

                                    let row_bg = if is_selected {
                                        "#2d3a2d"
                                    } else if all_correct && all_agree {
                                        "transparent"
                                    } else if !any_correct {
                                        "rgba(244,67,54,0.08)"
                                    } else {
                                        "rgba(255,235,59,0.06)"
                                    };

                                    let expected = r.correct_categories.join(", ");

                                    let method_cells = method_names.iter().map(|m| {
                                        let mr = r.results.get(m);
                                        let (cat, conf, color) = match mr {
                                            Some(mr) => {
                                                let is_correct = r.correct_categories.contains(&mr.chosen_category);
                                                let color = if is_correct { "#4caf50" } else { "#f44336" };
                                                (mr.chosen_category.clone(), format!("{:.0}%", mr.confidence * 100.0), color)
                                            }
                                            None => ("—".to_string(), "".to_string(), "#666"),
                                        };
                                        view! {
                                            <td style=format!("padding:4px 6px; font-family:monospace; font-size:0.8rem; color:{color};")>
                                                {cat}
                                                <span style="color:#666; font-size:0.7rem; margin-left:4px;">{conf}</span>
                                            </td>
                                        }
                                    }).collect_view();

                                    let r_clone = r.clone();
                                    view! {
                                        <tr
                                            style=format!("background:{row_bg}; cursor:pointer; border-bottom:1px solid #333;")
                                            on:click=move |_| {
                                                if is_selected {
                                                    set_selected_idx.set(None);
                                                } else {
                                                    set_selected_idx.set(Some(i));
                                                }
                                            }
                                        >
                                            <td style="padding:4px 6px; font-family:monospace; color:#666;">{r_clone.example_id}</td>
                                            <td style="padding:4px 6px; font-family:monospace; font-size:0.8rem; max-width:250px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                                                {preview}
                                            </td>
                                            {method_cells}
                                            <td style="padding:4px 6px; font-family:monospace; font-size:0.8rem; color:#aaa;">{expected}</td>
                                        </tr>
                                    }
                                }).collect_view()}
                            </tbody>
                        </table>
                    }
                }}
            </Show>

            // ── Expanded detail for selected example ──────────────────
            <Show when=move || selected_idx.get().is_some()>
                {move || {
                    let idx = selected_idx.get()?;
                    let r = results.get().into_iter().nth(idx)?;

                    Some(view! {
                        <div style="margin-top:1rem; padding:1rem; background:#1a1a2e; border-radius:4px;">
                            <h3 style="margin:0 0 0.5rem; font-size:0.95rem;">
                                "Example " {r.example_id} " — Detail"
                            </h3>
                            <p style="font-family:monospace; font-size:0.82rem; color:#ccc; white-space:pre-wrap; word-break:break-word; margin:0 0 0.75rem;">
                                {r.example_text.clone()}
                            </p>
                            <p style="font-size:0.8rem; color:#888; margin:0 0 0.5rem;">
                                "Correct: " {r.correct_categories.join(", ")}
                            </p>

                            // Per-method detail with full score ranking
                            {r.results.iter().map(|(method, mr)| {
                                let is_correct = r.correct_categories.contains(&mr.chosen_category);
                                let status_color = if is_correct { "#4caf50" } else { "#f44336" };
                                let status_label = if is_correct { "Correct" } else { "Incorrect" };

                                // Show top-5 scores
                                let top_scores: Vec<_> = mr.scores.iter().take(5).cloned().collect();

                                view! {
                                    <div style="margin-top:0.75rem; padding:0.5rem; background:#111; border-radius:3px;">
                                        <div style="display:flex; align-items:baseline; gap:0.5rem; margin-bottom:0.3rem;">
                                            <span style="font-family:monospace; color:#7eb8da; font-weight:bold;">{method.clone()}</span>
                                            <span style="font-family:monospace; font-size:0.85rem;">
                                                {mr.chosen_category.clone()}
                                            </span>
                                            <span style=format!("font-size:0.8rem; color:{status_color}; font-weight:bold;")>
                                                {status_label}
                                            </span>
                                            <span style="font-size:0.8rem; color:#888;">
                                                {format!("({:.1}% conf, {}ms)", mr.confidence * 100.0, mr.latency_ms)}
                                            </span>
                                        </div>
                                        <table style="border-collapse:collapse; font-size:0.8rem; width:100%;">
                                            <thead>
                                                <tr>
                                                    <th style="text-align:left; padding:2px 8px; color:#666;">"Rank"</th>
                                                    <th style="text-align:left; padding:2px 8px; color:#666;">"Category"</th>
                                                    <th style="text-align:right; padding:2px 8px; color:#666;">"Score"</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {top_scores.into_iter().enumerate().map(|(rank, s)| {
                                                    let is_chosen = s.category_name == mr.chosen_category;
                                                    let font_weight = if is_chosen { "bold" } else { "normal" };
                                                    view! {
                                                        <tr>
                                                            <td style=format!("padding:2px 8px; color:#666; font-weight:{font_weight};")>
                                                                {rank + 1}
                                                            </td>
                                                            <td style=format!("padding:2px 8px; font-family:monospace; font-weight:{font_weight};")>
                                                                {s.category_name}
                                                            </td>
                                                            <td style=format!("padding:2px 8px; text-align:right; font-family:monospace; font-weight:{font_weight};")>
                                                                {format!("{:.4}", s.score)}
                                                            </td>
                                                        </tr>
                                                    }
                                                }).collect_view()}
                                            </tbody>
                                        </table>
                                    </div>
                                }
                            }).collect_view()}
                        </div>
                    })
                }}
            </Show>
        </div>
    }
}
