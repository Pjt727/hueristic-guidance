use inference_types::{
    AgentVersionInfo, ConfusionCase, StepCandidates, TokenWithProb, ValidationAmendment,
};
use leptos::prelude::*;

use crate::app::api::{self, ProposalState};

#[component]
pub fn LlmValidationPage() -> impl IntoView {
    let (versions, set_versions) = signal::<Vec<AgentVersionInfo>>(vec![]);
    let (filter, set_filter) = signal(String::new());
    let (selected, set_selected) = signal::<Option<(i32, i32)>>(None);
    let (classifier_backend, set_classifier_backend) = signal("local".to_string());
    let (running, set_running) = signal(false);
    let (active_session, set_active_session) = signal::<Option<String>>(None);
    let (status, set_status) =
        signal("Select an agent version and start an LLM-only validation loop".to_string());
    let (rounds, set_rounds) = signal::<Vec<inference_types::ValidationRoundSummary>>(vec![]);
    let (proposal, set_proposal) = signal::<Option<ProposalState>>(None);
    let (pending_session, set_pending_session) = signal::<Option<String>>(None);
    let (expanded_round, set_expanded_round) = signal::<Option<u32>>(None);

    leptos::task::spawn_local(async move {
        match api::fetch_agent_versions().await {
            Ok(v) => set_versions.set(v),
            Err(e) => set_status.set(format!("Failed to load versions: {e}")),
        }
    });

    let on_start = move |_| {
        let Some((agent_id, version_id)) = selected.get_untracked() else {
            set_status.set("Pick an agent version first".into());
            return;
        };
        let backend = classifier_backend.get_untracked();
        set_running.set(true);
        set_rounds.set(vec![]);
        set_proposal.set(None);
        set_pending_session.set(None);
        set_expanded_round.set(Some(0));
        set_status.set(format!("Starting LLM validation ({backend})…"));
        leptos::task::spawn_local(async move {
            match api::start_llm_validation(agent_id, version_id, &backend).await {
                Ok(session_id) => {
                    set_active_session.set(Some(session_id.clone()));
                    set_pending_session.set(Some(session_id.clone()));
                    api::open_llm_validation_stream(
                        session_id.clone(),
                        set_status,
                        set_running,
                        set_rounds,
                        set_proposal,
                        set_pending_session,
                        session_id,
                    );
                }
                Err(e) => {
                    set_status.set(format!("Start failed: {e}"));
                    set_running.set(false);
                    set_active_session.set(None);
                }
            }
        });
    };

    let on_cancel = move |_| {
        let Some(session_id) = active_session
            .get_untracked()
            .or_else(|| pending_session.get_untracked())
        else {
            set_status.set("Nothing to cancel".into());
            return;
        };
        set_status.set("Cancelling…".into());
        leptos::task::spawn_local(async move {
            match api::cancel_llm_validation(&session_id).await {
                Ok(()) => set_status.set("Cancel requested — waiting for current example to finish…".into()),
                Err(e) => set_status.set(format!("Cancel failed: {e}")),
            }
        });
    };

    let on_decide = move |approve: bool| {
        let Some(session_id) = pending_session.get_untracked() else {
            set_status.set("No pending proposal".into());
            return;
        };
        set_running.set(true);
        set_status.set(if approve {
            "Applying amendments…".into()
        } else {
            "Denying — baseline retest…".into()
        });
        leptos::task::spawn_local(async move {
            match api::decide_llm_validation(&session_id, approve).await {
                Ok(stream_id) => {
                    set_active_session.set(Some(stream_id.clone()));
                    api::open_llm_validation_stream(
                        stream_id,
                        set_status,
                        set_running,
                        set_rounds,
                        set_proposal,
                        set_pending_session,
                        session_id,
                    );
                }
                Err(e) => {
                    set_status.set(format!("Decision failed: {e}"));
                    set_running.set(false);
                }
            }
        });
    };

    view! {
        <div>
            <h2 style="margin:0 0 0.5rem;">"LLM Validation"</h2>
            <p style="color:#aaa; font-size:0.9rem; margin-top:0;">
                "Classify single-link validation examples with the local LLM or OpenAI (no embeddings). "
                "OpenAI analyzes confusions and proposes minimal description / validation fixes. "
                "Local runs include decision-step logits for each confusion."
            </p>

            <div style="margin:0.75rem 0; display:flex; gap:1.25rem; align-items:center; flex-wrap:wrap;">
                <span style="color:#ccc; font-size:0.9rem;">"Classifier:"</span>
                <label style="display:flex; gap:0.35rem; align-items:center; font-size:0.9rem; cursor:pointer;">
                    <input
                        type="radio"
                        name="classifier_backend"
                        value="local"
                        prop:checked=move || classifier_backend.get() == "local"
                        prop:disabled=move || running.get()
                        on:change=move |_| set_classifier_backend.set("local".into())
                    />
                    "Local LLM"
                </label>
                <label style="display:flex; gap:0.35rem; align-items:center; font-size:0.9rem; cursor:pointer;">
                    <input
                        type="radio"
                        name="classifier_backend"
                        value="openai"
                        prop:checked=move || classifier_backend.get() == "openai"
                        prop:disabled=move || running.get()
                        on:change=move |_| set_classifier_backend.set("openai".into())
                    />
                    "OpenAI"
                </label>
            </div>

            <div style="margin:0.75rem 0;">
                <input
                    type="text"
                    placeholder="Filter by agent name…"
                    style="width:min(420px,100%); padding:6px 8px;"
                    prop:value=move || filter.get()
                    on:input=move |ev| set_filter.set(event_target_value(&ev))
                />
            </div>

            <div style="max-height:220px; overflow:auto; border:1px solid #333; margin-bottom:0.75rem;">
                <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
                    <thead>
                        <tr style="background:#1e1e1e;">
                            <th style="text-align:left; padding:4px 8px;">"Agent"</th>
                            <th style="text-align:left; padding:4px 8px;">"ID"</th>
                            <th style="text-align:left; padding:4px 8px;">"Version"</th>
                            <th style="text-align:left; padding:4px 8px;">"Status"</th>
                        </tr>
                    </thead>
                    <tbody>
                        {move || {
                            let q = filter.get().to_lowercase();
                            versions
                                .get()
                                .into_iter()
                                .filter(|v| q.is_empty() || v.agent_name.to_lowercase().contains(&q))
                                .map(|v| {
                                    let key = (v.agent_id, v.version_id);
                                    let selected_row = selected.get() == Some(key);
                                    let bg = if selected_row { "#2d3a2d" } else { "transparent" };
                                    let name = v.agent_name.clone();
                                    let status_label = v.deployment_status.clone().unwrap_or_else(|| "—".into());
                                    view! {
                                        <tr
                                            style=format!("cursor:pointer; background:{bg}; border-bottom:1px solid #333;")
                                            on:click=move |_| set_selected.set(Some(key))
                                        >
                                            <td style="padding:4px 8px; font-family:monospace;">{name}</td>
                                            <td style="padding:4px 8px; font-family:monospace;">{v.agent_id}</td>
                                            <td style="padding:4px 8px; font-family:monospace;">{v.version_id}</td>
                                            <td style="padding:4px 8px;">{status_label}</td>
                                        </tr>
                                    }
                                })
                                .collect_view()
                        }}
                    </tbody>
                </table>
            </div>

            <div style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">
                <button
                    prop:disabled=move || running.get() || selected.get().is_none()
                    on:click=on_start
                >
                    {move || if running.get() { "Running…" } else { "Start LLM Validation" }}
                </button>
                <button
                    prop:disabled=move || !running.get()
                    style="background:#5a1e1e; color:#fff; border:1px solid #833;"
                    on:click=on_cancel
                >
                    "Cancel"
                </button>
            </div>

            <p id="status" style="margin-top:0.75rem; font-family:monospace; font-size:0.85rem; color:#ddd;">{status}</p>

            <Show when=move || !rounds.get().is_empty()>
                <div style="margin-top:1.25rem;">
                    <h3 style="font-size:1rem; color:#eee; margin-bottom:0.5rem;">"Results by round"</h3>
                    <div style="display:flex; flex-direction:column; gap:0.75rem;">
                        {move || rounds.get().into_iter().map(|r| {
                            let round_id = r.round;
                            let confusions = r.confusions.clone();
                            let confusion_count = confusions.len();
                            let passing_categories = r.passing_categories.clone();
                            let passing_close = r.passing_close_logit_count;
                            let success_count = r.success_count;
                            let exec = r.executive_summary.clone();
                            let is_open = move || expanded_round.get() == Some(round_id);
                            let accuracy_color = if r.accuracy_pct >= 95.0 {
                                "#4caf50"
                            } else if r.accuracy_pct >= 80.0 {
                                "#c9a227"
                            } else {
                                "#e57373"
                            };
                            view! {
                                <div style="border:1px solid #444; border-radius:4px; overflow:hidden;">
                                    <button
                                        style="width:100%; text-align:left; padding:0.65rem 0.75rem; background:#1a1a1a; border:none; color:inherit; cursor:pointer; display:flex; gap:1rem; flex-wrap:wrap; align-items:baseline;"
                                        on:click=move |_| {
                                            set_expanded_round.update(|cur| {
                                                *cur = if *cur == Some(round_id) { None } else { Some(round_id) };
                                            });
                                        }
                                    >
                                        <span style="font-weight:600; font-family:monospace;">{format!("Round {}", r.round)}</span>
                                        <span style=format!("font-family:monospace; font-size:1.05rem; color:{accuracy_color}; font-weight:700;")>
                                            {format!("{:.1}%", r.accuracy_pct)}
                                        </span>
                                        <span style="font-family:monospace; color:#bbb;">
                                            {format!("{}/{} correct", r.success_count, r.total)}
                                        </span>
                                        <span style="font-family:monospace; color:#e57373;">
                                            {format!("{confusion_count} wrong")}
                                        </span>
                                        <span style="font-family:monospace; color:#ffd54f;">
                                            {format!("{passing_close}/{success_count} passes close-logit")}
                                        </span>
                                        <span style="color:#888; font-size:0.85rem;">
                                            {format!("{} amendment(s)", r.amendments_applied.len())}
                                        </span>
                                        <span style="margin-left:auto; color:#888; font-size:0.8rem;">
                                            {move || if is_open() { "Hide details ▴" } else { "Show details ▾" }}
                                        </span>
                                    </button>

                                    <Show when=is_open>
                                        <div style="padding:0.75rem; background:#121212; border-top:1px solid #333; display:flex; flex-direction:column; gap:1rem;">
                                            {exec.as_ref().map(|s| executive_summary_view(s))}
                                            <div>
                                                <h4 style="margin:0 0 0.4rem; font-size:0.85rem; color:#ccc;">"Passing categories"</h4>
                                                {
                                                    if passing_categories.is_empty() {
                                                        view! {
                                                            <p style="color:#777; margin:0; font-size:0.85rem;">"No passing examples this round."</p>
                                                        }.into_any()
                                                    } else {
                                                        view! {
                                                            <table style="border-collapse:collapse; font-family:monospace; font-size:0.82rem; width:min(560px,100%);">
                                                                <thead>
                                                                    <tr style="color:#888;">
                                                                        <th style="text-align:left; padding:3px 8px;">"Category"</th>
                                                                        <th style="text-align:right; padding:3px 8px;">"Passed"</th>
                                                                        <th style="text-align:right; padding:3px 8px;">"Close logits"</th>
                                                                    </tr>
                                                                </thead>
                                                                <tbody>
                                                                    {passing_categories.iter().map(|p| {
                                                                        let close_style = if p.close_logit_count > 0 {
                                                                            "color:#ffd54f;"
                                                                        } else {
                                                                            "color:#bbb;"
                                                                        };
                                                                        view! {
                                                                            <tr style="border-bottom:1px solid #2a2a2a;">
                                                                                <td style="padding:3px 8px;">{p.category_name.clone()}</td>
                                                                                <td style="text-align:right; padding:3px 8px;">{p.pass_count}</td>
                                                                                <td style=format!("text-align:right; padding:3px 8px; {close_style}")>
                                                                                    {format!("{}/{}", p.close_logit_count, p.pass_count)}
                                                                                </td>
                                                                            </tr>
                                                                        }
                                                                    }).collect_view()}
                                                                </tbody>
                                                            </table>
                                                        }.into_any()
                                                    }
                                                }
                                            </div>
                                            <div>
                                                <h4 style="margin:0 0 0.4rem; font-size:0.85rem; color:#e57373;">"Confusions"</h4>
                                                {
                                                    if confusion_count == 0 {
                                                        view! {
                                                            <p style="color:#4caf50; margin:0;">"No confusions this round."</p>
                                                        }.into_any()
                                                    } else {
                                                        view! {
                                                            <div style="display:flex; flex-direction:column; gap:0.85rem;">
                                                                {confusions.iter().map(|c| confusion_card(c)).collect_view()}
                                                            </div>
                                                        }.into_any()
                                                    }
                                                }
                                            </div>
                                        </div>
                                    </Show>
                                </div>
                            }
                        }).collect_view()}
                    </div>
                </div>
            </Show>

            <Show when=move || proposal.get().is_some()>
                {move || proposal.get().map(|p| {
                    let amendments = p.amendments.clone();
                    let amendments_empty = amendments.is_empty();
                    let amendments_list = amendments.clone();
                    let exec = p.executive_summary.clone();
                    view! {
                    <div style="margin-top:1.25rem; border:1px solid #444; padding:0.75rem;">
                        <h3 style="margin-top:0; font-size:0.95rem;">
                            {format!("Proposed amendments (best round {} @ {:.1}%)", p.best_round, p.best_accuracy_pct)}
                        </h3>
                        {exec.as_ref().map(|s| executive_summary_view(s))}
                        <Show when=move || amendments_empty>
                            <p style="color:#aaa;">
                                "No amendments to apply — the best accuracy was the baseline (or later edits did not improve it). Deny to retest baseline, or Approve as a no-op."
                            </p>
                        </Show>
                        <ul style="padding-left:1.1rem;">
                            {amendments_list.iter().map(|a| amendment_view(a)).collect_view()}
                        </ul>
                        <div style="display:flex; gap:0.5rem; margin-top:0.75rem;">
                            <button
                                prop:disabled=move || running.get()
                                on:click=move |_| on_decide(true)
                            >
                                "Approve & write to DB"
                            </button>
                            <button
                                prop:disabled=move || running.get()
                                on:click=move |_| on_decide(false)
                            >
                                "Deny & retest baseline once"
                            </button>
                        </div>
                    </div>
                }})}
            </Show>
        </div>
    }
}

fn executive_summary_view(s: &inference_types::AnalysisExecutiveSummary) -> AnyView {
    let priorities = s.category_priority_assumptions.clone();
    let diffs = s.differentiation_decisions.clone();
    let labels = s.label_challenges.clone();
    let empty = priorities.is_empty() && diffs.is_empty() && labels.is_empty();
    if empty {
        return view! { <div></div> }.into_any();
    }
    view! {
        <div style="margin-bottom:0.75rem; border:1px solid #3a3a55; border-radius:4px; padding:0.6rem 0.75rem; background:#14141f;">
            <h4 style="margin:0 0 0.4rem; font-size:0.85rem; color:#9fa8da;">"Analysis executive summary"</h4>
            {(!priorities.is_empty()).then(|| view! {
                <div style="margin-bottom:0.4rem;">
                    <strong style="font-size:0.78rem; color:#bbb;">"Category priority"</strong>
                    <ul style="margin:0.2rem 0 0; padding-left:1.1rem; font-size:0.82rem; color:#ccc;">
                        {priorities.into_iter().map(|t| view! { <li>{t}</li> }).collect_view()}
                    </ul>
                </div>
            })}
            {(!diffs.is_empty()).then(|| view! {
                <div style="margin-bottom:0.4rem;">
                    <strong style="font-size:0.78rem; color:#bbb;">"Differentiation decisions"</strong>
                    <ul style="margin:0.2rem 0 0; padding-left:1.1rem; font-size:0.82rem; color:#ccc;">
                        {diffs.into_iter().map(|t| view! { <li>{t}</li> }).collect_view()}
                    </ul>
                </div>
            })}
            {(!labels.is_empty()).then(|| view! {
                <div>
                    <strong style="font-size:0.78rem; color:#bbb;">"Label challenges"</strong>
                    <ul style="margin:0.2rem 0 0; padding-left:1.1rem; font-size:0.82rem; color:#ccc;">
                        {labels.into_iter().map(|t| view! { <li>{t}</li> }).collect_view()}
                    </ul>
                </div>
            })}
        </div>
    }
    .into_any()
}

fn confusion_card(c: &ConfusionCase) -> AnyView {
    let predicted = c
        .predicted_category
        .clone()
        .unwrap_or_else(|| "(none)".into());
    let margin = c
        .top_logit_margin
        .map(|m| format!("{m:.2}"))
        .unwrap_or_else(|| "—".into());
    let steps = c.decision_steps.clone();
    let has_steps = !steps.is_empty();
    let analysis = c.analysis.clone();
    let close = c.close_top_logits;
    let non_grammar = c.non_grammar_top_logits;
    let example_text = c.example_text.clone();
    let expected = c.expected_category.clone();
    let example_id = c.example_id;
    let is_repeat = c.is_repeat_offender;
    let recurrence = c.recurrence_count;
    let prior_fails = c.prior_amendment_failures.clone();
    let has_prior_fails = !prior_fails.is_empty();
    let close_badge = close.then(|| format!("Close logits (Δ {margin})"));

    view! {
        <div style="border:1px solid #553333; border-radius:4px; padding:0.65rem; background:#1a1212;">
            <div style="display:flex; flex-wrap:wrap; gap:0.5rem 1rem; align-items:center; margin-bottom:0.5rem;">
                <span style="font-family:monospace; font-weight:700; color:#e57373;">{format!("#{example_id}")}</span>
                {is_repeat.then(|| view! {
                    <span style="background:#6a1b9a; color:#e1bee7; font-size:0.75rem; padding:2px 6px; border-radius:3px; font-weight:700;">
                        {format!("REPEAT OFFENDER ×{recurrence}")}
                    </span>
                })}
                {close_badge.map(|label| view! {
                    <span style="background:#5c4a12; color:#ffd54f; font-size:0.75rem; padding:2px 6px; border-radius:3px;">
                        {label}
                    </span>
                })}
                {non_grammar.then(|| view! {
                    <span style="background:#5c1a1a; color:#ff8a80; font-size:0.75rem; padding:2px 6px; border-radius:3px;">
                        "Non-grammar in top logits"
                    </span>
                })}
            </div>
            <div style="display:grid; grid-template-columns:auto 1fr; gap:0.25rem 0.75rem; margin-bottom:0.65rem; align-items:baseline;">
                <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.04em; color:#81c784; font-weight:700;">"Expected"</span>
                <span style="font-family:monospace; font-size:1.05rem; font-weight:700; color:#a5d6a7;">{expected}</span>
                <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.04em; color:#e57373; font-weight:700;">"Got"</span>
                <span style="font-family:monospace; font-size:1.05rem; font-weight:700; color:#ef9a9a;">{predicted}</span>
            </div>
            <div style="font-family:monospace; font-size:0.82rem; color:#ccc; white-space:pre-wrap; margin-bottom:0.5rem;">
                {example_text}
            </div>
            {has_prior_fails.then(|| view! {
                <div style="font-size:0.82rem; color:#ce93d8; margin-bottom:0.5rem; border-left:3px solid #6a1b9a; padding-left:0.5rem;">
                    <strong style="display:block; margin-bottom:0.35rem;">"Why ALL prior amendments failed:"</strong>
                    <ol style="margin:0; padding-left:1.1rem;">
                        {prior_fails.iter().enumerate().map(|(i, a)| {
                            let summary = a.amendment_summary.clone();
                            let why = a.why_it_failed.clone();
                            view! {
                                <li style="margin-bottom:0.4rem;">
                                    <div style="color:#b39ddb; font-family:monospace; font-size:0.78rem;">{format!("{}. {}", i + 1, summary)}</div>
                                    <div>{why}</div>
                                </li>
                            }
                        }).collect_view()}
                    </ol>
                </div>
            })}
            {analysis.map(|a| view! {
                <div style="font-size:0.82rem; color:#9fa8da; margin-bottom:0.5rem;">
                    <strong>"Analysis: "</strong>{a}
                </div>
            })}
            {
                if has_steps {
                    view! {
                        <div style="display:flex; flex-direction:column; gap:0.5rem;">
                            {steps.into_iter().enumerate().map(|(i, step)| {
                                view! { <DecisionStepView index=i step=step /> }
                            }).collect_view()}
                        </div>
                    }.into_any()
                } else {
                    view! {
                        <p style="color:#777; font-size:0.8rem; margin:0;">
                            "No local logit steps (OpenAI classify has no token logits)."
                        </p>
                    }.into_any()
                }
            }
        </div>
    }
    .into_any()
}

#[component]
fn DecisionStepView(index: usize, step: StepCandidates) -> impl IntoView {
    let constrained_ids: std::collections::HashSet<u32> =
        step.top_constrained.iter().map(|t| t.token_id).collect();
    let alternatives = step.top_alternatives.clone();
    // Keep in sync with server CLOSE_LOGIT_MARGIN (6.0).
    let close = alternatives.len() >= 2
        && (alternatives[0].logit - alternatives[1].logit).abs() < 6.0;
    let margin = if alternatives.len() >= 2 {
        Some(alternatives[0].logit - alternatives[1].logit)
    } else {
        None
    };

    view! {
        <div style="border:1px solid #333; border-radius:3px; padding:0.5rem; background:#161616;">
            <div style="font-size:0.8rem; color:#aaa; margin-bottom:0.35rem;">
                {format!(
                    "Before grammar · step {index} · chose {:?}{}",
                    step.chosen.text,
                    margin.map(|m| format!(" · top-2 Δlogit={m:.2}")).unwrap_or_default()
                )}
            </div>
            <LogitTable
                tokens=alternatives
                constrained_ids=constrained_ids
                highlight_close=close
                flag_non_grammar=true
            />
        </div>
    }
}

#[component]
fn LogitTable(
    tokens: Vec<TokenWithProb>,
    constrained_ids: std::collections::HashSet<u32>,
    highlight_close: bool,
    flag_non_grammar: bool,
) -> impl IntoView {
    view! {
        <table style="width:100%; font-family:monospace; font-size:0.8rem; border-collapse:collapse; margin-top:0.25rem;">
            <thead>
                <tr style="color:#888;">
                    <th style="text-align:left; padding:2px 6px;">"Token"</th>
                    <th style="text-align:right; padding:2px 6px;">"Prob"</th>
                    <th style="text-align:right; padding:2px 6px;">"Logit"</th>
                    <th style="text-align:left; padding:2px 6px;">"Flags"</th>
                </tr>
            </thead>
            <tbody>
                {tokens.into_iter().enumerate().map(|(i, t)| {
                    let non_grammar = flag_non_grammar && !constrained_ids.contains(&t.token_id);
                    let close_row = highlight_close && i < 2;
                    let bg = if non_grammar {
                        "background:rgba(244,67,54,0.18);"
                    } else if close_row {
                        "background:rgba(255,213,79,0.12);"
                    } else {
                        ""
                    };
                    let mut flags = Vec::new();
                    if non_grammar {
                        flags.push("NON-GRAMMAR");
                    }
                    if close_row {
                        flags.push("CLOSE");
                    }
                    let flag_text = flags.join(" · ");
                    let flag_color = if non_grammar { "#ff8a80" } else if close_row { "#ffd54f" } else { "#666" };
                    view! {
                        <tr style=bg>
                            <td style="padding:2px 6px;">{format!("{:?}", t.text)}</td>
                            <td style="text-align:right; padding:2px 6px;">{format!("{:.4}", t.probability)}</td>
                            <td style="text-align:right; padding:2px 6px;">{format!("{:.3}", t.logit)}</td>
                            <td style=format!("padding:2px 6px; color:{flag_color}; font-size:0.75rem;")>{flag_text}</td>
                        </tr>
                    }
                }).collect_view()}
            </tbody>
        </table>
    }
}

fn amendment_view(a: &ValidationAmendment) -> AnyView {
    match a {
        ValidationAmendment::Description {
            category_name,
            old_description,
            new_description,
            rationale,
            message_id,
        } => view! {
            <li style="margin-bottom:0.6rem;">
                <div style="font-weight:bold;">{format!("Description · {category_name} (msg {message_id})")}</div>
                <div style="color:#f44336; font-size:0.8rem;">{format!("− {old_description}")}</div>
                <div style="color:#4caf50; font-size:0.8rem;">{format!("+ {new_description}")}</div>
                <div style="color:#aaa; font-size:0.8rem;">{rationale.clone()}</div>
            </li>
        }
        .into_any(),
        ValidationAmendment::ReassignValidation {
            example_id,
            example_text,
            from_category,
            to_category,
            rationale,
            ..
        } => view! {
            <li style="margin-bottom:0.6rem;">
                <div style="font-weight:bold;">{format!("Reassign validation · example {example_id}")}</div>
                <div style="font-size:0.8rem; font-family:monospace;">{example_text.clone()}</div>
                <div style="font-size:0.85rem;">{format!("{from_category} → {to_category}")}</div>
                <div style="color:#aaa; font-size:0.8rem;">{rationale.clone()}</div>
            </li>
        }
        .into_any(),
        ValidationAmendment::RemoveValidation {
            example_id,
            example_text,
            category_name,
            rationale,
            ..
        } => view! {
            <li style="margin-bottom:0.6rem;">
                <div style="font-weight:bold;">{format!("Remove validation · example {example_id} ({category_name})")}</div>
                <div style="font-size:0.8rem; font-family:monospace;">{example_text.clone()}</div>
                <div style="color:#aaa; font-size:0.8rem;">{rationale.clone()}</div>
            </li>
        }
        .into_any(),
    }
}
