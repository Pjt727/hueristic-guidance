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
                {tokens.into_iter().map(|t| {
                    let text = format!("{:?}", t.text);
                    let prob = format!("{:.4}", t.probability);
                    let logit = format!("{:.4}", t.logit);
                    let emb_logit = format!("{:.4}", t.embedding_logit);
                    view! {
                        <tr>
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
                {entries.into_iter().map(|e| {
                    let cat = e.category_name;
                    let text = format!("{:?}", e.best_token.text);
                    let total = format!("{:.4}", e.best_token.logit + e.best_token.embedding_logit);
                    let logit = format!("{:.4}", e.best_token.logit);
                    let emb_logit = format!("{:.4}", e.best_token.embedding_logit);
                    view! {
                        <tr>
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
