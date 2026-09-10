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

                                let fmt_acc = |c: usize, t: usize| -> String {
                                    if t == 0 { "—".into() }
                                    else { format!("{c}/{t} ({:.0}%)", c as f64 / t as f64 * 100.0) }
                                };
                                let fmt_pct = |p: f64| -> String { format!("{p:.1}%") };

                                // Use server-computed accuracy if available, fallback to client sim.
                                let (logit_correct, logit_total) = resp.no_embedding_accuracy
                                    .as_ref()
                                    .map(|a| (a.correct, a.total))
                                    .unwrap_or_else(|| {
                                        let rs = results.get();
                                        let empty: HashMap<String, f64> = HashMap::new();
                                        simulate_accuracy(&rs, &empty, 0.0)
                                    });
                                let (k10_correct, k10_total) = resp.baseline_accuracy
                                    .as_ref()
                                    .map(|a| (a.correct, a.total))
                                    .unwrap_or_else(|| {
                                        let rs = results.get();
                                        let empty: HashMap<String, f64> = HashMap::new();
                                        simulate_accuracy(&rs, &empty, 10.0)
                                    });
                                let (sim_correct, sim_total) = resp.optimized_accuracy
                                    .as_ref()
                                    .map(|a| (a.correct, a.total))
                                    .unwrap_or_else(|| {
                                        let rs = results.get();
                                        simulate_accuracy(&rs, &resp.weights, 10.0)
                                    });

                                let mut entries: Vec<(String, f64)> = resp.weights.clone().into_iter().collect();
                                entries.sort_by(|(a, _), (b, _)| a.cmp(b));

                                // Ensemble optimization data.
                                let ensemble_opt = resp.ensemble_optimization.clone();

                                Some(view! {
                                    <div style="margin-top:0.75rem;">
                                        <p style="font-size:0.82rem; color:#aaa; margin:0 0 0.5rem;">
                                            {format!("Optimised on {} examples ({} skipped). \
                                                      These are the new kappa values (default baseline = 10.0). \
                                                      Higher = more embedding influence; lower = less.",
                                                resp.examples_used, resp.examples_skipped)}
                                        </p>

                                        // ── Kappa accuracy comparison ──────────────────
                                        <h4 style="font-size:0.85rem; margin:0.5rem 0 0.3rem; color:#ccc;">"Kappa Optimisation (LLM + Embedding Bias)"</h4>
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
                                                    <td style="padding:3px 12px; font-weight:bold;">"Optimised kappas"</td>
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

                                        // ── Per-category kappa table ──────────────────
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

                                        // ── Apply button ──────────────────────────────
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

                                        // ── Ensemble & LLM Analysis ───────────────────
                                        {match ensemble_opt {
                                            None => None,
                                            Some(eo) => {
                                                let ow = eo.optimal_weights.clone();
                                                let ot = eo.optimal_temperature.clone();
                                                let llm_pct = eo.llm_accuracy_pct;
                                                let bias_pct = eo.bias_corrected_accuracy_pct;

                                                // Build weight search rows (select interesting points).
                                                let weight_rows: Vec<_> = eo.weight_search.iter()
                                                    .filter(|wp| {
                                                        let w = (wp.tfidf_weight * 20.0).round();
                                                        w == w.floor() // every 5% step
                                                    })
                                                    .cloned()
                                                    .collect();

                                                // Category bias offsets.
                                                let bias_offsets = eo.category_bias_offsets.clone();

                                                // LLM category stats.
                                                let mut llm_stats: Vec<_> = eo.llm_category_stats.into_iter().collect();
                                                llm_stats.sort_by(|(a, _), (b, _)| a.cmp(b));

                                                // Baseline accuracies.
                                                let baseline = eo.baseline_accuracy.clone();

                                                Some(view! {
                                                    <div style="margin-top:1.5rem; border-top:1px solid #333; padding-top:1rem;">
                                                        <h4 style="font-size:0.85rem; margin:0 0 0.5rem; color:#ccc;">
                                                            "Classifier Ensemble Analysis"
                                                        </h4>

                                                        // ── Standalone method accuracy ─────
                                                        <table style="border-collapse:collapse; font-size:0.82rem; margin-bottom:0.75rem;">
                                                            <thead>
                                                                <tr style="background:#1e1e1e;">
                                                                    <th style="text-align:left; padding:3px 12px">"Method"</th>
                                                                    <th style="text-align:right; padding:3px 12px">"Accuracy"</th>
                                                                </tr>
                                                            </thead>
                                                            <tbody>
                                                                {baseline.into_iter().map(|(method, acc)| {
                                                                    view! {
                                                                        <tr style="border-bottom:1px solid #2a2a2a;">
                                                                            <td style="padding:3px 12px; color:#aaa;">{method}</td>
                                                                            <td style="padding:3px 12px; text-align:right; font-family:monospace;">
                                                                                {fmt_pct(acc)}
                                                                            </td>
                                                                        </tr>
                                                                    }
                                                                }).collect_view()}
                                                                {llm_pct.map(|p| view! {
                                                                    <tr style="border-bottom:1px solid #2a2a2a;">
                                                                        <td style="padding:3px 12px; color:#aaa;">"LLM (1st step logits)"</td>
                                                                        <td style="padding:3px 12px; text-align:right; font-family:monospace;">
                                                                            {fmt_pct(p)}
                                                                        </td>
                                                                    </tr>
                                                                })}
                                                                <tr style="border-bottom:1px solid #2a2a2a;">
                                                                    <td style="padding:3px 12px; font-weight:bold;">
                                                                        {format!("Best ensemble (TF-IDF {:.0}% / Embed {:.0}%)",
                                                                            ow.tfidf_weight * 100.0, ow.embedding_weight * 100.0)}
                                                                    </td>
                                                                    <td style="padding:3px 12px; text-align:right; font-family:monospace; font-weight:bold; color:#4caf50;">
                                                                        {fmt_pct(ow.accuracy_pct)}
                                                                    </td>
                                                                </tr>
                                                                {bias_pct.map(|p| view! {
                                                                    <tr>
                                                                        <td style="padding:3px 12px; color:#aaa;">
                                                                            "Embed + LLM bias offsets"
                                                                        </td>
                                                                        <td style="padding:3px 12px; text-align:right; font-family:monospace;">
                                                                            {fmt_pct(p)}
                                                                        </td>
                                                                    </tr>
                                                                })}
                                                            </tbody>
                                                        </table>

                                                        // ── Weight search grid ─────────────
                                                        <details style="margin-bottom:0.75rem;">
                                                            <summary style="font-size:0.82rem; color:#aaa; cursor:pointer;">
                                                                "Weight search grid (TF-IDF vs Embedding)"
                                                            </summary>
                                                            <table style="border-collapse:collapse; font-size:0.78rem; margin-top:0.3rem;">
                                                                <thead>
                                                                    <tr style="background:#1e1e1e;">
                                                                        <th style="text-align:right; padding:2px 8px">"TF-IDF %"</th>
                                                                        <th style="text-align:right; padding:2px 8px">"Embed %"</th>
                                                                        <th style="text-align:right; padding:2px 8px">"Accuracy"</th>
                                                                    </tr>
                                                                </thead>
                                                                <tbody>
                                                                    {weight_rows.into_iter().map(|wp| {
                                                                        let is_best = (wp.tfidf_weight - ow.tfidf_weight).abs() < 0.01;
                                                                        let style = if is_best {
                                                                            "border-bottom:1px solid #2a2a2a; background:#1a2e1a;"
                                                                        } else {
                                                                            "border-bottom:1px solid #2a2a2a;"
                                                                        };
                                                                        view! {
                                                                            <tr style=style>
                                                                                <td style="padding:2px 8px; text-align:right; font-family:monospace;">
                                                                                    {format!("{:.0}", wp.tfidf_weight * 100.0)}
                                                                                </td>
                                                                                <td style="padding:2px 8px; text-align:right; font-family:monospace;">
                                                                                    {format!("{:.0}", wp.embedding_weight * 100.0)}
                                                                                </td>
                                                                                <td style="padding:2px 8px; text-align:right; font-family:monospace;">
                                                                                    {fmt_pct(wp.accuracy_pct)}
                                                                                </td>
                                                                            </tr>
                                                                        }
                                                                    }).collect_view()}
                                                                </tbody>
                                                            </table>
                                                        </details>

                                                        // ── Temperature search ─────────────
                                                        <details style="margin-bottom:0.75rem;">
                                                            <summary style="font-size:0.82rem; color:#aaa; cursor:pointer;">
                                                                {format!("Temperature scaling (best T={:.4}, acc={:.1}%)",
                                                                    ot.temperature, ot.accuracy_pct)}
                                                            </summary>
                                                            <p style="font-size:0.78rem; color:#666; margin:0.2rem 0;">
                                                                "Tests softmax(embed_scores / T) combined with TF-IDF in a 50/50 ensemble. \
                                                                 Lower T amplifies small embedding differences."
                                                            </p>
                                                        </details>

                                                        // ── LLM-derived bias offsets ───────
                                                        {if !bias_offsets.is_empty() {
                                                            Some(view! {
                                                                <details style="margin-bottom:0.75rem;">
                                                                    <summary style="font-size:0.82rem; color:#aaa; cursor:pointer;">
                                                                        "LLM-derived category bias offsets"
                                                                    </summary>
                                                                    <p style="font-size:0.78rem; color:#666; margin:0.2rem 0 0.3rem;">
                                                                        "Residual between avg LLM softmax probability and avg embedding score. \
                                                                         Positive = LLM favours this category more than embeddings do."
                                                                    </p>
                                                                    <table style="border-collapse:collapse; font-size:0.78rem;">
                                                                        <thead>
                                                                            <tr style="background:#1e1e1e;">
                                                                                <th style="text-align:left; padding:2px 8px">"Category"</th>
                                                                                <th style="text-align:right; padding:2px 8px">"Bias offset"</th>
                                                                                <th style="text-align:right; padding:2px 8px">"Samples"</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody>
                                                                            {bias_offsets.into_iter().map(|bo| {
                                                                                let color = if bo.bias_offset > 0.01 { "#4caf50" }
                                                                                    else if bo.bias_offset < -0.01 { "#f44336" }
                                                                                    else { "#aaa" };
                                                                                view! {
                                                                                    <tr style="border-bottom:1px solid #2a2a2a;">
                                                                                        <td style="padding:2px 8px; font-family:monospace;">
                                                                                            {bo.category_name}
                                                                                        </td>
                                                                                        <td style=format!("padding:2px 8px; text-align:right; \
                                                                                            font-family:monospace; color:{color};")>
                                                                                            {format!("{:+.4}", bo.bias_offset)}
                                                                                        </td>
                                                                                        <td style="padding:2px 8px; text-align:right; font-family:monospace;">
                                                                                            {bo.sample_count}
                                                                                        </td>
                                                                                    </tr>
                                                                                }
                                                                            }).collect_view()}
                                                                        </tbody>
                                                                    </table>
                                                                </details>
                                                            })
                                                        } else {
                                                            None
                                                        }}

                                                        // ── LLM category stats ────────────
                                                        {if !llm_stats.is_empty() {
                                                            Some(view! {
                                                                <details>
                                                                    <summary style="font-size:0.82rem; color:#aaa; cursor:pointer;">
                                                                        "LLM per-category logit statistics"
                                                                    </summary>
                                                                    <table style="border-collapse:collapse; font-size:0.78rem; margin-top:0.3rem;">
                                                                        <thead>
                                                                            <tr style="background:#1e1e1e;">
                                                                                <th style="text-align:left; padding:2px 8px">"Category"</th>
                                                                                <th style="text-align:right; padding:2px 8px">"Avg logit"</th>
                                                                                <th style="text-align:right; padding:2px 8px">"Avg prob"</th>
                                                                                <th style="text-align:right; padding:2px 8px">"Chosen"</th>
                                                                                <th style="text-align:right; padding:2px 8px">"Correct"</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody>
                                                                            {llm_stats.into_iter().map(|(name, st)| {
                                                                                view! {
                                                                                    <tr style="border-bottom:1px solid #2a2a2a;">
                                                                                        <td style="padding:2px 8px; font-family:monospace;">{name}</td>
                                                                                        <td style="padding:2px 8px; text-align:right; font-family:monospace;">
                                                                                            {format!("{:.2}", st.avg_logit)}
                                                                                        </td>
                                                                                        <td style="padding:2px 8px; text-align:right; font-family:monospace;">
                                                                                            {format!("{:.1}%", st.avg_probability * 100.0)}
                                                                                        </td>
                                                                                        <td style="padding:2px 8px; text-align:right; font-family:monospace;">
                                                                                            {st.times_chosen}
                                                                                        </td>
                                                                                        <td style="padding:2px 8px; text-align:right; font-family:monospace;">
                                                                                            {st.times_correct}
                                                                                        </td>
                                                                                    </tr>
                                                                                }
                                                                            }).collect_view()}
                                                                        </tbody>
                                                                    </table>
                                                                </details>
                                                            })
                                                        } else {
                                                            None
                                                        }}
                                                    </div>
                                                })
                                            }
                                        }}
                                    </div>
                                })
                            }}
                        </Show>
                    </div>
                </Show>

                // Results table
                <Show when=move || !results.get().is_empty()>
                    {move || {
                        let rs = results.get();
                        // Collect classifier method names from the first result that has any
                        let clf_methods: Vec<String> = rs
                            .iter()
                            .flat_map(|r| r.classifier_results.iter().map(|c| c.method_name.clone()))
                            .collect::<std::collections::HashSet<_>>()
                            .into_iter()
                            .collect::<Vec<_>>();
                        let mut clf_methods_sorted = clf_methods.clone();
                        clf_methods_sorted.sort();

                        let clf_methods_hdr = clf_methods_sorted.clone();
                        let clf_methods_rows = clf_methods_sorted.clone();

                        view! {
                            <table style="width:100%; border-collapse:collapse; margin-top:0.5rem; font-size:0.9rem;">
                                <thead>
                                    <tr style="background:#2a2a2a;">
                                        <th style="text-align:left; padding:4px 8px">"#"</th>
                                        <th style="text-align:left; padding:4px 8px">"Example"</th>
                                        <th style="text-align:left; padding:4px 8px">"LLM Category"</th>
                                        <th style="text-align:left; padding:4px 8px">"Expected"</th>
                                        {clf_methods_hdr.into_iter().map(|m| {
                                            view! {
                                                <th style="text-align:left; padding:4px 8px; font-size:0.8rem; color:#aaa;">
                                                    {m}
                                                </th>
                                            }
                                        }).collect_view()}
                                        <th style="text-align:center; padding:4px 8px">"Result"</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {rs.into_iter().enumerate().map(|(i, r)| {
                                        let is_selected = selected_result_idx.get() == Some(i);
                                        let badge_color = if r.success { "#4caf50" } else { "#f44336" };
                                        let badge = if r.success { "✓" } else { "✗" };
                                        let preview: String = r.example_text.chars().take(80).collect();
                                        let preview = if r.example_text.len() > 80 {
                                            format!("{preview}…")
                                        } else {
                                            preview
                                        };
                                        let llm_cat = r.chosen_category.clone().unwrap_or_else(|| "—".to_string());
                                        let expected = r.correct_categories.join(", ");
                                        let steps = r.steps.clone();

                                        // Build a map of method → chosen_category for this row
                                        let clf_map: std::collections::HashMap<String, String> = r
                                            .classifier_results
                                            .iter()
                                            .map(|c| (c.method_name.clone(), c.chosen_category.clone()))
                                            .collect();

                                        // Determine if classifiers disagree with each other or with LLM
                                        let all_cats: Vec<&str> = {
                                            let mut v: Vec<&str> = r.classifier_results
                                                .iter()
                                                .map(|c| c.chosen_category.as_str())
                                                .collect();
                                            if let Some(ref c) = r.chosen_category {
                                                v.push(c.as_str());
                                            }
                                            v
                                        };
                                        let first_cat = all_cats.first().copied().unwrap_or("");
                                        let row_has_disagreement = all_cats.iter().any(|c| *c != first_cat);

                                        let row_bg = if is_selected {
                                            "#2d3a2d".to_string()
                                        } else if row_has_disagreement {
                                            "#2a2000".to_string()
                                        } else {
                                            "transparent".to_string()
                                        };

                                        let clf_cols: Vec<String> = clf_methods_rows.clone();
                                        let correct_cats = r.correct_categories.clone();

                                        view! {
                                            <tr
                                                style=format!("background:{row_bg}; cursor:pointer; border-bottom:1px solid #333;")
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
                                                <td style="padding:4px 8px; font-family:monospace;">{llm_cat}</td>
                                                <td style="padding:4px 8px; font-family:monospace; color:#aaa;">{expected}</td>
                                                {clf_cols.into_iter().map(|method| {
                                                    let cat = clf_map.get(&method).cloned().unwrap_or_else(|| "—".to_string());
                                                    let is_correct = correct_cats.contains(&cat);
                                                    let cell_color = if cat == "—" {
                                                        "#666"
                                                    } else if is_correct {
                                                        "#4caf50"
                                                    } else {
                                                        "#f44336"
                                                    };
                                                    view! {
                                                        <td style=format!(
                                                            "padding:4px 8px; font-family:monospace; font-size:0.8rem; color:{cell_color};"
                                                        )>
                                                            {cat}
                                                        </td>
                                                    }
                                                }).collect_view()}
                                                <td style=format!("padding:4px 8px; text-align:center; color:{badge_color}; font-weight:bold;")>
                                                    {badge}
                                                </td>
                                            </tr>
                                        }
                                    }).collect_view()}
                                </tbody>
                            </table>
                        }
                    }}
                </Show>
            </div>
        </div>
    }
}
