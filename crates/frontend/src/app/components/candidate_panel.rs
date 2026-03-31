use std::cmp::Ordering;

use inference_types::{CategoryTopToken, StepCandidates, TokenWithProb};
use leptos::prelude::*;

#[component]
pub fn CandidatePanel(step: StepCandidates) -> impl IntoView {
    let constrained = step.top_constrained.clone();
    let alternatives = step.top_alternatives.clone();
    let category_top = step.category_top_tokens.clone();

    view! {
        <div style="margin-top:1rem; padding:0.5rem; background:#1e1e1e; border-radius:4px;">
            <h3 style="margin:0 0 0.5rem">"Candidates at this step"</h3>

            <details open>
                <summary>"After grammar mask (constrained)"</summary>
                <CandidateTable tokens=constrained />
            </details>

            <details>
                <summary>"Before grammar mask (all)"</summary>
                <CandidateTable tokens=alternatives />
            </details>

            {if !category_top.is_empty() {
                view! {
                    <details open>
                        <summary>"Best token each category"</summary>
                        <CategoryTopTable entries=category_top />
                    </details>
                }.into_any()
            } else {
                view! { <div></div> }.into_any()
            }}
        </div>
    }
}

#[component]
fn CandidateTable(tokens: Vec<TokenWithProb>) -> impl IntoView {
    // Tokens arrive sorted by probability (total score = logit + embedding_logit).
    // Find the row that would be #1 by raw logit alone.
    let top_raw_idx = tokens
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.logit
                .partial_cmp(&b.logit)
                .unwrap_or(Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Highlight only when the embedding changed who's on top.
    let embedding_changed = top_raw_idx != 0;

    view! {
        <table style="width:100%; font-family:monospace; font-size:0.85rem; border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="text-align:left; padding:2px 6px">"Token"</th>
                    <th style="text-align:right; padding:2px 6px">"Prob"</th>
                    <th style="text-align:right; padding:2px 6px">"Logit"</th>
                    <th style="text-align:right; padding:2px 6px">"Emb. Logit"</th>
                </tr>
            </thead>
            <tbody>
                {tokens.into_iter().enumerate().map(|(i, t)| {
                    let text = format!("{:?}", t.text);
                    let prob = format!("{:.4}", t.probability);
                    let logit = format!("{:.4}", t.logit);
                    let emb_logit = format!("{:.4}", t.embedding_logit);
                    let bg = if embedding_changed {
                        if i == 0 {
                            // Promoted to top by embedding bias
                            "background:rgba(255,200,0,0.12);"
                        } else if i == top_raw_idx {
                            // Would have been #1 by raw logit, pushed down
                            "background:rgba(255,100,0,0.12);"
                        } else {
                            ""
                        }
                    } else {
                        ""
                    };
                    view! {
                        <tr style=bg>
                            <td style="padding:2px 6px">{text}</td>
                            <td style="text-align:right; padding:2px 6px">{prob}</td>
                            <td style="text-align:right; padding:2px 6px">{logit}</td>
                            <td style="text-align:right; padding:2px 6px">{emb_logit}</td>
                        </tr>
                    }
                }).collect_view()}
            </tbody>
        </table>
    }
}

#[component]
fn CategoryTopTable(entries: Vec<CategoryTopToken>) -> impl IntoView {
    // Find which row is #1 by total score (logit + embedding_logit)
    // and which would be #1 by raw logit alone.
    let top_total_idx = entries
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let sa = a.best_token.logit + a.best_token.embedding_logit;
            let sb = b.best_token.logit + b.best_token.embedding_logit;
            sa.partial_cmp(&sb).unwrap_or(Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let top_raw_idx = entries
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.best_token
                .logit
                .partial_cmp(&b.best_token.logit)
                .unwrap_or(Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let embedding_changed = top_total_idx != top_raw_idx;

    view! {
        <table style="width:100%; font-family:monospace; font-size:0.85rem; border-collapse:collapse;">
            <thead>
                <tr>
                    <th style="text-align:left; padding:2px 6px">"Category"</th>
                    <th style="text-align:left; padding:2px 6px">"Token"</th>
                    <th style="text-align:right; padding:2px 6px">"Total Score"</th>
                    <th style="text-align:right; padding:2px 6px">"Logit"</th>
                    <th style="text-align:right; padding:2px 6px">"Emb. Logit"</th>
                </tr>
            </thead>
            <tbody>
                {entries.into_iter().enumerate().map(|(i, e)| {
                    let cat = e.category_name;
                    let text = format!("{:?}", e.best_token.text);
                    let total = format!("{:.4}", e.best_token.logit + e.best_token.embedding_logit);
                    let logit = format!("{:.4}", e.best_token.logit);
                    let emb_logit = format!("{:.4}", e.best_token.embedding_logit);
                    let bg = if embedding_changed {
                        if i == top_total_idx {
                            "background:rgba(255,200,0,0.12);"
                        } else if i == top_raw_idx {
                            "background:rgba(255,100,0,0.12);"
                        } else {
                            ""
                        }
                    } else {
                        ""
                    };
                    view! {
                        <tr style=bg>
                            <td style="padding:2px 6px">{cat}</td>
                            <td style="padding:2px 6px">{text}</td>
                            <td style="text-align:right; padding:2px 6px">{total}</td>
                            <td style="text-align:right; padding:2px 6px">{logit}</td>
                            <td style="text-align:right; padding:2px 6px">{emb_logit}</td>
                        </tr>
                    }
                }).collect_view()}
            </tbody>
        </table>
    }
}
