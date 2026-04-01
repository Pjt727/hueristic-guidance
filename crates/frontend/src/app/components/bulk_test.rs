use std::collections::HashMap;

use inference_types::StepCandidates;
use leptos::ev;
use leptos::prelude::*;

use crate::app::api::{self, BulkTestRunSummary, OptimizeResponse, TestResult};
use crate::app::components::{AgentSelector, CandidatePanel, TokenStreamView};

/// Estimate how many examples would be classified correctly if each category's
/// kappa were replaced with the values in `kappas`.
///
/// For categories absent from `kappas`, `fallback_kappa` is used (pass `10.0`
/// to simulate the original default, or `0.0` for logit-only).
///
/// Returns `(correct, total)`.  Examples with no `category_top_tokens` in any
/// step are skipped and not counted in `total`.
fn simulate_accuracy(
    results: &[TestResult],
    kappas: &HashMap<String, f64>,
    fallback_kappa: f64,
) -> (usize, usize) {
    let mut correct = 0usize;
    let mut total = 0usize;

    for result in results {
        // Find the first step that has category_top_tokens populated.
        let Some(step) = result.steps.iter().find(|s| !s.category_top_tokens.is_empty()) else {
            continue;
        };

        total += 1;

        // Pick the category with the highest simulated total score.
        let best = step.category_top_tokens.iter().max_by(|a, b| {
            let kappa_a = kappas.get(&a.category_name).copied().unwrap_or(fallback_kappa);
            let kappa_b = kappas.get(&b.category_name).copied().unwrap_or(fallback_kappa);
            let score_a = a.best_token.logit as f64 + kappa_a * a.sim_score as f64;
            let score_b = b.best_token.logit as f64 + kappa_b * b.sim_score as f64;
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(token) = best {
            if result.correct_categories.contains(&token.category_name) {
                correct += 1;
            }
        }
    }

    (correct, total)
}

#[component]
pub fn BulkTestPage() -> impl IntoView {
    let (agent_id, set_agent_id) = signal::<Option<i32>>(None);
    let (running, set_running) = signal(false);
    let (status, set_status) = signal("Select an agent and click Run Bulk Test".to_string());
    let (results, set_results) = signal::<Vec<TestResult>>(vec![]);
    let (total, set_total) = signal::<usize>(0);

    // Which result row is expanded
    let (selected_result_idx, set_selected_result_idx) = signal::<Option<usize>>(None);
    // Steps of the currently-expanded result
    let (expanded_steps, set_expanded_steps) = signal::<Vec<StepCandidates>>(vec![]);
    // Which step (token) inside the expanded result is highlighted in CandidatePanel
    let (selected_step_idx, set_selected_step_idx) = signal::<Option<usize>>(None);

    // The run_id of the currently-displayed results (set when loading a past run)
    let (current_run_id, set_current_run_id) = signal::<Option<i64>>(None);
    // Optimisation results
    let (optimize_result, set_optimize_result) = signal::<Option<OptimizeResponse>>(None);
    let (optimize_running, set_optimize_running) = signal(false);
    let (optimize_error, set_optimize_error) = signal::<Option<String>>(None);
    let (apply_running, set_apply_running) = signal(false);
    let (apply_status, set_apply_status) = signal::<Option<String>>(None);

    // Previous runs — loaded once on mount
    let (past_runs, set_past_runs) = signal::<Vec<BulkTestRunSummary>>(vec![]);
    let (past_runs_error, set_past_runs_error) = signal::<Option<String>>(None);
    leptos::task::spawn_local(async move {
        match api::fetch_bulk_test_runs().await {
            Ok(runs) => set_past_runs.set(runs),
            Err(e) => set_past_runs_error.set(Some(e)),
        }
    });

    // Right panel width (px) — draggable from the left edge
    let (panel_width, set_panel_width) = signal(460_f64);
    let is_dragging = RwSignal::new(false);

    // Document-level mousemove / mouseup to complete drag gestures that leave the handle
    let _mm = window_event_listener(ev::mousemove, move |e: web_sys::MouseEvent| {
        if is_dragging.get_untracked() {
            let vw = web_sys::window()
                .and_then(|w| w.inner_width().ok())
                .and_then(|v| v.as_f64())
                .unwrap_or(1024.0);
            let new_w = (vw - e.client_x() as f64).max(260.0).min(vw * 0.85);
            set_panel_width.set(new_w);
        }
    });
    let _mu = window_event_listener(ev::mouseup, move |_: web_sys::MouseEvent| {
        is_dragging.set(false);
    });

    let on_run = move || {
        let Some(aid) = agent_id.get_untracked() else {
            set_status.set("Select a VC agent first".to_string());
            return;
        };
        set_results.set(vec![]);
        set_selected_result_idx.set(None);
        set_expanded_steps.set(vec![]);
        set_selected_step_idx.set(None);
        set_total.set(0);
        set_status.set("Starting bulk test…".to_string());
        set_running.set(true);

        leptos::task::spawn_local(async move {
            match api::start_bulk_test(aid).await {
                Ok((bulk_test_id, run_id)) => {
                    set_current_run_id.set(Some(run_id));
                    set_optimize_result.set(None);
                    set_optimize_error.set(None);
                    set_status.set(format!("Running — {bulk_test_id}"));
                    api::open_bulk_test_stream(
                        bulk_test_id,
                        set_results,
                        set_status,
                        set_running,
                        set_total,
                    );
                }
                Err(e) => {
                    set_status.set(format!("Error: {e}"));
                    set_running.set(false);
                }
            }
        });
    };

    view! {
        // ── Right panel: fixed to right edge, draggable left border ─────────
        <Show when=move || selected_result_idx.get().is_some()>
            <div style=move || format!(
                "position:fixed; right:0; top:0; width:{w}px; height:100vh; \
                 background:#111; border-left:1px solid #2a2a2a; z-index:10; \
                 display:flex; box-sizing:border-box;",
                w = panel_width.get()
            )>
                // Drag handle — 6 px strip at the left edge
                <div
                    style="width:6px; flex-shrink:0; cursor:col-resize; \
                           background:rgba(255,255,255,0.04); \
                           border-right:1px solid #333; \
                           transition:background 0.15s;"
                    on:mousedown=move |e| {
                        e.prevent_default(); // prevent text selection while dragging
                        is_dragging.set(true);
                    }
                />

                // Scrollable content area (hidden scrollbar)
                <div style="flex:1; overflow-y:scroll; scrollbar-width:none; \
                            padding:1rem; box-sizing:border-box; min-width:0;">
                    // Example header + full text
                    {move || {
                        let idx = selected_result_idx.get()?;
                        let result = results.get().into_iter().nth(idx)?;
                        let status_color = if result.success { "#4caf50" } else { "#f44336" };
                        let status_label = if result.success { "✓ Correct" } else { "✗ Incorrect" };
                        Some(view! {
                            <div style="margin-bottom:0.75rem;">
                                <div style="display:flex; align-items:baseline; gap:0.5rem; margin-bottom:0.4rem;">
                                    <span style="font-weight:bold; font-size:0.95rem;">
                                        "Example " {result.example_id}
                                    </span>
                                    <span style=format!("color:{status_color}; font-weight:bold; font-size:0.9rem;")>
                                        {status_label}
                                    </span>
                                </div>
                                <p style="font-family:monospace; font-size:0.82rem; color:#ccc; \
                                          margin:0 0 0.75rem; white-space:pre-wrap; word-break:break-word;">
                                    {result.example_text.clone()}
                                </p>
                                <p style="font-size:0.78rem; color:#666; margin:0 0 0.4rem;">
                                    "Click a token to inspect candidates."
                                </p>
                                <TokenStreamView
                                    steps=expanded_steps
                                    set_selected_idx=set_selected_step_idx
                                />
                            </div>
                        })
                    }}

                    // Per-token candidate detail
                    <Show when=move || selected_step_idx.get().is_some()>
                        {move || {
                            let idx = selected_step_idx.get()?;
                            let step = expanded_steps.get().into_iter().nth(idx)?;
                            Some(view! { <CandidatePanel step=step /> })
                        }}
                    </Show>
                </div>
            </div>
        </Show>

        // ── Main content: right padding tracks the open panel ────────────────
        <div style=move || {
            if selected_result_idx.get().is_some() {
                format!("padding-right:{}px;", panel_width.get() + 16.0)
            } else {
                String::new()
            }
        }>
            <div style="max-width:900px;">
                <h1>"Bulk Test"</h1>

                <AgentSelector
                    selected_agent_id=agent_id
                    set_selected_agent_id=set_agent_id
                />

                <div style="margin-top:0.75rem;">
                    <button
                        disabled=move || running.get() || agent_id.get().is_none()
                        on:click=move |_| on_run()
                    >
                        "Run Bulk Test"
                    </button>
                </div>

                <p id="status">{status}</p>

                // ── Previous runs ────────────────────────────────────────────
                <Show when=move || !past_runs.get().is_empty()>
                    <details style="margin-top:0.5rem;">
                        <summary style="cursor:pointer; font-size:0.9rem; color:#aaa; user-select:none;">
                            "Previous bulk tests"
                        </summary>
                        <Show when=move || past_runs_error.get().is_some()>
                            <p style="color:#f44336; font-size:0.85rem;">
                                {move || past_runs_error.get().unwrap_or_default()}
                            </p>
                        </Show>
                        <table style="width:100%; border-collapse:collapse; margin-top:0.4rem; font-size:0.85rem;">
                            <thead>
                                <tr style="background:#1e1e1e;">
                                    <th style="text-align:left; padding:3px 6px">"Agent"</th>
                                    <th style="text-align:left; padding:3px 6px">"Started"</th>
                                    <th style="text-align:right; padding:3px 6px">"Pass rate"</th>
                                    <th style="padding:3px 6px"></th>
                                </tr>
                            </thead>
                            <tbody>
                                {move || past_runs.get().into_iter().map(|run| {
                                    let pass_label = match (run.success_count, run.total) {
                                        (Some(s), Some(t)) if t > 0 =>
                                            format!("{s}/{t} ({:.0}%)", s as f64 / t as f64 * 100.0),
                                        (Some(s), Some(t)) => format!("{s}/{t}"),
                                        _ => "—".to_string(),
                                    };
                                    let started = run.started_at.get(..16).unwrap_or(&run.started_at).to_string();
                                    let run_id = run.id;
                                    view! {
                                        <tr style="border-bottom:1px solid #2a2a2a;">
                                            <td style="padding:3px 6px; font-family:monospace;">{run.agent_id}</td>
                                            <td style="padding:3px 6px; color:#aaa;">{started}</td>
                                            <td style="padding:3px 6px; text-align:right; font-family:monospace;">{pass_label}</td>
                                            <td style="padding:3px 6px;">
                                                <button
                                                    style="font-size:0.8rem; padding:1px 8px;"
                                                    on:click=move |_| {
                                                        set_results.set(vec![]);
                                                        set_selected_result_idx.set(None);
                                                        set_expanded_steps.set(vec![]);
                                                        set_selected_step_idx.set(None);
                                                        set_optimize_result.set(None);
                                                        set_optimize_error.set(None);
                                                        set_current_run_id.set(Some(run_id));
                                                        set_status.set(format!("Loading run {run_id}…"));
                                                        leptos::task::spawn_local(async move {
                                                            match api::fetch_bulk_test_run(run_id).await {
                                                                Ok(r) => {
                                                                    let n = r.len();
                                                                    let s = r.iter().filter(|x| x.success).count();
                                                                    set_results.set(r);
                                                                    set_total.set(n);
                                                                    set_status.set(format!(
                                                                        "Loaded run {run_id} — {s}/{n} passed"
                                                                    ));
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

                // Progress summary
                <Show when=move || !results.get().is_empty()>
                    {move || {
                        let rs = results.get();
                        let done = rs.len();
                        let success = rs.iter().filter(|r| r.success).count();
                        let tot = total.get();
                        let pct = if done == 0 { 0.0 } else { success as f64 / done as f64 * 100.0 };
                        let label = if tot > 0 {
                            format!("{done}/{tot} complete — {success} passed ({pct:.0}%)")
                        } else {
                            format!("{done} complete — {success} passed ({pct:.0}%)")
                        };
                        view! { <p style="font-weight:bold;">{label}</p> }
                    }}
                </Show>

                // ── Optimise weights ─────────────────────────────────────────
                <Show when=move || current_run_id.get().is_some() && !results.get().is_empty()>
                    <div style="margin-top:1rem;">
                        <button
                            disabled=move || optimize_running.get()
                            on:click=move |_| {
                                let Some(rid) = current_run_id.get_untracked() else { return; };
                                set_optimize_result.set(None);
                                set_optimize_error.set(None);
                                set_apply_status.set(None);
                                set_optimize_running.set(true);
                                leptos::task::spawn_local(async move {
                                    match api::optimize_weights(rid).await {
                                        Ok(resp) => {
                                            set_optimize_result.set(Some(resp));
                                            set_optimize_error.set(None);
                                        }
                                        Err(e) => set_optimize_error.set(Some(e)),
                                    }
                                    set_optimize_running.set(false);
                                });
                            }
                        >
                            {move || if optimize_running.get() { "Optimising…" } else { "Optimise Category Weights" }}
                        </button>

                        <Show when=move || optimize_error.get().is_some()>
                            <p style="color:#f44336; font-size:0.85rem; margin-top:0.4rem;">
                                {move || optimize_error.get().unwrap_or_default()}
                            </p>
                        </Show>

                        <Show when=move || optimize_result.get().is_some()>
                            {move || {
                                let resp = optimize_result.get()?;
                                let weights_for_apply = resp.weights.clone();

                                // Simulated accuracy with optimised kappas.
                                let rs = results.get();
                                let (sim_correct, sim_total) =
                                    simulate_accuracy(&rs, &resp.weights, 10.0);
                                // Baseline: logit-only (kappa = 0 for all).
                                let empty: HashMap<String, f64> = HashMap::new();
                                let (logit_correct, logit_total) =
                                    simulate_accuracy(&rs, &empty, 0.0);
                                // Baseline: original kappa = 10.
                                let (k10_correct, k10_total) =
                                    simulate_accuracy(&rs, &empty, 10.0);

                                let fmt_acc = |c: usize, t: usize| -> String {
                                    if t == 0 { "—".into() }
                                    else { format!("{c}/{t} ({:.0}%)", c as f64 / t as f64 * 100.0) }
                                };

                                let mut entries: Vec<(String, f64)> = resp.weights.clone().into_iter().collect();
                                entries.sort_by(|(a, _), (b, _)| a.cmp(b));
                                Some(view! {
                                    <div style="margin-top:0.75rem;">
                                        <p style="font-size:0.82rem; color:#aaa; margin:0 0 0.5rem;">
                                            {format!("Optimised on {} examples ({} skipped). \
                                                      These are the new kappa values (default baseline = 10.0). \
                                                      Higher = more embedding influence; lower = less.",
                                                resp.examples_used, resp.examples_skipped)}
                                        </p>

                                        // Accuracy comparison table
                                        <table style="border-collapse:collapse; font-size:0.82rem; margin-bottom:0.75rem;">
                                            <thead>
                                                <tr style="background:#1e1e1e;">
                                                    <th style="text-align:left; padding:3px 12px">"Scenario"</th>
                                                    <th style="text-align:right; padding:3px 12px">"Accuracy"</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr style="border-bottom:1px solid #2a2a2a;">
                                                    <td style="padding:3px 12px; color:#aaa;">"Logit only (kappa = 0)"</td>
                                                    <td style="padding:3px 12px; text-align:right; font-family:monospace;">
                                                        {fmt_acc(logit_correct, logit_total)}
                                                    </td>
                                                </tr>
                                                <tr style="border-bottom:1px solid #2a2a2a;">
                                                    <td style="padding:3px 12px; color:#aaa;">"Default (kappa = 10)"</td>
                                                    <td style="padding:3px 12px; text-align:right; font-family:monospace;">
                                                        {fmt_acc(k10_correct, k10_total)}
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td style="padding:3px 12px; font-weight:bold;">"Optimised kappas (estimated)"</td>
                                                    <td style=format!(
                                                        "padding:3px 12px; text-align:right; font-family:monospace; \
                                                         font-weight:bold; color:{};",
                                                        if sim_correct >= k10_correct { "#4caf50" } else { "#f44336" }
                                                    )>
                                                        {fmt_acc(sim_correct, sim_total)}
                                                    </td>
                                                </tr>
                                            </tbody>
                                        </table>
                                        <table style="border-collapse:collapse; font-size:0.85rem; min-width:340px;">
                                            <thead>
                                                <tr style="background:#1e1e1e;">
                                                    <th style="text-align:left; padding:3px 10px">"Category"</th>
                                                    <th style="text-align:right; padding:3px 10px">"Optimal kappa"</th>
                                                    <th style="text-align:right; padding:3px 10px">"Δ from 10.0"</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {entries.into_iter().map(|(cat, w)| {
                                                    let delta = w - 10.0;
                                                    let delta_color = if delta > 0.5 { "#4caf50" }
                                                                      else if delta < -0.5 { "#f44336" }
                                                                      else { "#aaa" };
                                                    let delta_str = format!("{:+.3}", delta);
                                                    view! {
                                                        <tr style="border-bottom:1px solid #2a2a2a;">
                                                            <td style="padding:3px 10px; font-family:monospace;">{cat}</td>
                                                            <td style="padding:3px 10px; text-align:right; font-family:monospace;">
                                                                {format!("{:.4}", w)}
                                                            </td>
                                                            <td style=format!("padding:3px 10px; text-align:right; \
                                                                              font-family:monospace; color:{delta_color};")>
                                                                {delta_str}
                                                            </td>
                                                        </tr>
                                                    }
                                                }).collect_view()}
                                            </tbody>
                                        </table>
                                        <div style="margin-top:0.75rem; display:flex; align-items:center; gap:0.75rem;">
                                            <button
                                                disabled=move || apply_running.get()
                                                on:click={
                                                    let weights = weights_for_apply.clone();
                                                    move |_| {
                                                        let Some(rid) = current_run_id.get_untracked() else { return; };
                                                        let weights = weights.clone();
                                                        set_apply_running.set(true);
                                                        set_apply_status.set(None);
                                                        leptos::task::spawn_local(async move {
                                                            match api::apply_weights(rid, &weights).await {
                                                                Ok(r) => {
                                                                    let msg = if r.unmatched_categories.is_empty() {
                                                                        format!("Saved — {} message(s) updated.", r.updated)
                                                                    } else {
                                                                        format!(
                                                                            "Saved — {} message(s) updated. Unmatched: {}",
                                                                            r.updated,
                                                                            r.unmatched_categories.join(", ")
                                                                        )
                                                                    };
                                                                    set_apply_status.set(Some(msg));
                                                                }
                                                                Err(e) => set_apply_status.set(Some(format!("Error: {e}"))),
                                                            }
                                                            set_apply_running.set(false);
                                                        });
                                                    }
                                                }
                                            >
                                                {move || if apply_running.get() { "Saving…" } else { "Apply These Kappas" }}
                                            </button>
                                            <Show when=move || apply_status.get().is_some()>
                                                <span style="font-size:0.82rem; color:#aaa;">
                                                    {move || apply_status.get().unwrap_or_default()}
                                                </span>
                                            </Show>
                                        </div>
                                    </div>
                                })
                            }}
                        </Show>
                    </div>
                </Show>

                // Results table
                <Show when=move || !results.get().is_empty()>
                    <table style="width:100%; border-collapse:collapse; margin-top:0.5rem; font-size:0.9rem;">
                        <thead>
                            <tr style="background:#2a2a2a;">
                                <th style="text-align:left; padding:4px 8px">"#"</th>
                                <th style="text-align:left; padding:4px 8px">"Example"</th>
                                <th style="text-align:left; padding:4px 8px">"Chosen Category"</th>
                                <th style="text-align:left; padding:4px 8px">"Expected Categories"</th>
                                <th style="text-align:center; padding:4px 8px">"Result"</th>
                            </tr>
                        </thead>
                        <tbody>
                            {move || {
                                results.get().into_iter().enumerate().map(|(i, r)| {
                                    let is_selected = selected_result_idx.get() == Some(i);
                                    let bg = if is_selected { "#2d3a2d" } else { "transparent" };
                                    let badge_color = if r.success { "#4caf50" } else { "#f44336" };
                                    let badge = if r.success { "✓" } else { "✗" };
                                    let preview: String = r.example_text.chars().take(80).collect();
                                    let preview = if r.example_text.len() > 80 {
                                        format!("{preview}…")
                                    } else {
                                        preview
                                    };
                                    let cat = r.chosen_category.clone().unwrap_or_else(|| "—".to_string());
                                    let expected = r.correct_categories.join(", ");
                                    let steps = r.steps.clone();
                                    view! {
                                        <tr
                                            style=format!("background:{bg}; cursor:pointer; border-bottom:1px solid #333;")
                                            on:click=move |_| {
                                                if is_selected {
                                                    set_selected_result_idx.set(None);
                                                    set_expanded_steps.set(vec![]);
                                                    set_selected_step_idx.set(None);
                                                } else {
                                                    set_selected_result_idx.set(Some(i));
                                                    set_expanded_steps.set(steps.clone());
                                                    set_selected_step_idx.set(None);
                                                }
                                            }
                                        >
                                            <td style="padding:4px 8px; font-family:monospace;">{r.example_id}</td>
                                            <td style="padding:4px 8px; font-family:monospace;">{preview}</td>
                                            <td style="padding:4px 8px; font-family:monospace;">{cat}</td>
                                            <td style="padding:4px 8px; font-family:monospace; color:#aaa;">{expected}</td>
                                            <td style=format!("padding:4px 8px; text-align:center; color:{badge_color}; font-weight:bold;")>
                                                {badge}
                                            </td>
                                        </tr>
                                    }
                                }).collect_view()
                            }}
                        </tbody>
                    </table>
                </Show>
            </div>
        </div>
    }
}
