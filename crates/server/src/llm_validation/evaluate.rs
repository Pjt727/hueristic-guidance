use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use inference::{GrammarFlow, InferenceEngine, InferenceEvent};
use inference_types::{ClassifierBackend, ConfusionCase, StepCandidates};
use tokio::sync::{mpsc, Semaphore};

use super::dataset::ValidationDataset;

/// Max concurrent OpenAI classification requests.
const OPENAI_CLASSIFY_CONCURRENCY: usize = 5;

/// Unconstrained top-1 vs top-2 logit gap below this → "close / ambiguous".
/// Gaps like 18 vs 16 (Δ=2) still count as close.
const CLOSE_LOGIT_MARGIN: f32 = 6.0;

#[derive(Clone)]
pub struct ExampleEvalResult {
    pub example_id: i32,
    pub example_text: String,
    pub expected_category: String,
    pub expected_message_id: i32,
    pub predicted_category: Option<String>,
    pub predicted_message_id: Option<i32>,
    pub success: bool,
    pub decision_steps: Vec<StepCandidates>,
    pub close_top_logits: bool,
    pub non_grammar_top_logits: bool,
    pub top_logit_margin: Option<f32>,
}

pub struct EvalReport {
    pub results: Vec<ExampleEvalResult>,
    pub success_count: usize,
    pub cancelled: bool,
}

impl EvalReport {
    pub fn total(&self) -> usize {
        self.results.len()
    }

    pub fn accuracy_pct(&self) -> f64 {
        if self.results.is_empty() {
            0.0
        } else {
            (self.success_count as f64) * 100.0 / (self.results.len() as f64)
        }
    }

    pub fn confusions(&self) -> Vec<&ExampleEvalResult> {
        self.results.iter().filter(|r| !r.success).collect()
    }
}

pub async fn evaluate_dataset(
    engine: &Arc<InferenceEngine>,
    brand_name: &str,
    dataset: &ValidationDataset,
    on_example: Option<mpsc::Sender<(u32, ExampleEvalResult)>>,
    round: u32,
    backend: ClassifierBackend,
    cancel: Option<Arc<AtomicBool>>,
) -> anyhow::Result<EvalReport> {
    match backend {
        ClassifierBackend::Local => {
            evaluate_local(engine, brand_name, dataset, on_example, round, cancel).await
        }
        ClassifierBackend::Openai => {
            evaluate_openai(brand_name, dataset, on_example, round, cancel).await
        }
    }
}

fn cancelled(cancel: &Option<Arc<AtomicBool>>) -> bool {
    cancel
        .as_ref()
        .is_some_and(|c| c.load(Ordering::SeqCst))
}

async fn evaluate_local(
    engine: &Arc<InferenceEngine>,
    brand_name: &str,
    dataset: &ValidationDataset,
    mut on_example: Option<mpsc::Sender<(u32, ExampleEvalResult)>>,
    round: u32,
    cancel: Option<Arc<AtomicBool>>,
) -> anyhow::Result<EvalReport> {
    let vc_messages: Vec<_> = dataset
        .messages
        .iter()
        .map(|m| m.vc_message.clone())
        .collect();
    let grammar_flow = GrammarFlow::new(brand_name, &vc_messages)?;
    let maps = CategoryMaps::from_dataset(dataset);

    let mut results = Vec::with_capacity(dataset.examples.len());
    let mut success_count = 0usize;
    let mut was_cancelled = false;

    for example in &dataset.examples {
        if cancelled(&cancel) {
            was_cancelled = true;
            break;
        }

        let Some(&expected_message_id) = dataset.expected.get(&example.id) else {
            continue;
        };
        let expected_category = maps.expected_category(expected_message_id);

        // LLM-only: no embedding biases.
        let mut rx = engine
            .generate(example.text.clone(), grammar_flow.clone(), vec![])
            .await;

        let mut full_text: Option<String> = None;
        let mut sampled_steps = Vec::new();
        while let Some(event) = rx.recv().await {
            match event {
                InferenceEvent::Done { full_text: ft, .. } => {
                    full_text = Some(ft);
                    break;
                }
                InferenceEvent::Error { message } => {
                    tracing::warn!(
                        example_id = example.id,
                        error = %message,
                        "llm validation inference error"
                    );
                    break;
                }
                InferenceEvent::Token(step) => {
                    if !step.top_alternatives.is_empty() {
                        sampled_steps.push(step);
                    }
                }
            }
        }

        let result = finish_example(
            example.id,
            &example.text,
            expected_message_id,
            &expected_category,
            full_text.as_deref(),
            &maps,
            select_decision_steps(sampled_steps),
        );
        if result.success {
            success_count += 1;
        }
        if let Some(tx) = on_example.as_mut() {
            let _ = tx.send((round, result.clone())).await;
        }
        results.push(result);
    }

    Ok(EvalReport {
        results,
        success_count,
        cancelled: was_cancelled,
    })
}

async fn evaluate_openai(
    brand_name: &str,
    dataset: &ValidationDataset,
    on_example: Option<mpsc::Sender<(u32, ExampleEvalResult)>>,
    round: u32,
    cancel: Option<Arc<AtomicBool>>,
) -> anyhow::Result<EvalReport> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY is required for OpenAI classification"))?;
    if api_key.is_empty() {
        anyhow::bail!("OPENAI_API_KEY is required for OpenAI classification");
    }
    let model =
        std::env::var("OPENAI_CLASSIFY_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let vc_messages: Vec<_> = dataset
        .messages
        .iter()
        .map(|m| m.vc_message.clone())
        .collect();
    let grammar_flow = GrammarFlow::new(brand_name, &vc_messages)?;
    let system_prompt = openai_classify_system_prompt(grammar_flow.get_system_prompt());
    // Stable key so concurrent classify calls for this catalog share cache routing.
    let prompt_cache_key = prompt_cache_key_for(&system_prompt);
    let maps = CategoryMaps::from_dataset(dataset);
    let client = reqwest::Client::new();
    let sem = Arc::new(Semaphore::new(OPENAI_CLASSIFY_CONCURRENCY));

    let mut work = Vec::new();
    for example in &dataset.examples {
        let Some(&expected_message_id) = dataset.expected.get(&example.id) else {
            continue;
        };
        work.push((
            example.id,
            example.text.clone(),
            expected_message_id,
            maps.expected_category(expected_message_id),
        ));
    }

    let mut handles = Vec::with_capacity(work.len());
    for (idx, (example_id, text, expected_message_id, expected_category)) in
        work.into_iter().enumerate()
    {
        if cancelled(&cancel) {
            break;
        }
        let sem = sem.clone();
        let client = client.clone();
        let api_key = api_key.clone();
        let model = model.clone();
        let system_prompt = system_prompt.clone();
        let prompt_cache_key = prompt_cache_key.clone();
        let maps = maps.clone();
        let cancel = cancel.clone();

        handles.push(tokio::spawn(async move {
            if cancelled(&cancel) {
                return Ok::<_, anyhow::Error>(None);
            }
            let _permit = sem
                .acquire()
                .await
                .map_err(|e| anyhow::anyhow!("semaphore closed: {e}"))?;
            if cancelled(&cancel) {
                return Ok(None);
            }
            let full_text = openai_classify(
                &client,
                &api_key,
                &model,
                &system_prompt,
                &prompt_cache_key,
                &text,
            )
            .await?;
            let result = finish_example(
                example_id,
                &text,
                expected_message_id,
                &expected_category,
                Some(&full_text),
                &maps,
                vec![],
            );
            Ok(Some((idx, result)))
        }));
    }

    let was_cancelled = cancelled(&cancel);
    let mut indexed = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(Ok(Some(pair))) => indexed.push(pair),
            Ok(Ok(None)) => {}
            Ok(Err(e)) => return Err(e),
            Err(e) => return Err(anyhow::anyhow!("classify task join error: {e}")),
        }
    }
    indexed.sort_by_key(|(idx, _)| *idx);

    let mut results = Vec::with_capacity(indexed.len());
    let mut success_count = 0usize;
    for (_, result) in indexed {
        if result.success {
            success_count += 1;
        }
        if let Some(tx) = &on_example {
            let _ = tx.send((round, result.clone())).await;
        }
        results.push(result);
    }

    Ok(EvalReport {
        results,
        success_count,
        cancelled: was_cancelled,
    })
}

fn openai_classify_system_prompt(base: &str) -> String {
    format!(
        "{base}\n\
         ---\n\
         Chat API notes: pick exactly one category name from the Approved Responses List above. \
         Reply with a single line `Category: <exact name>` and nothing else. \
         Do not invent categories or add explanation."
    )
}

fn prompt_cache_key_for(system_prompt: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    system_prompt.hash(&mut hasher);
    format!("llm-val-classify-{:016x}", hasher.finish())
}

/// Explicit `prompt_cache_breakpoint` / `prompt_cache_options` are GPT-5.6+ only;
/// earlier models 400 if those fields are present.
fn model_supports_explicit_prompt_cache(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    let Some(rest) = m.strip_prefix("gpt-5.") else {
        return false;
    };
    let maj: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    maj.parse::<u32>().is_ok_and(|n| n >= 6)
}

async fn openai_classify(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    system_prompt: &str,
    prompt_cache_key: &str,
    user_text: &str,
) -> anyhow::Result<String> {
    // Keep the category catalog in a stable system prefix; only the HCP text varies.
    // prompt_cache_key routes concurrent requests for the same catalog onto one cache.
    let mut body = serde_json::json!({
        "model": model,
        "temperature": 0.0,
        "prompt_cache_key": prompt_cache_key,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_text }
        ]
    });

    if model_supports_explicit_prompt_cache(model) {
        // Cache only through the system prompt; do not write each unique user example.
        body["prompt_cache_options"] = serde_json::json!({ "mode": "explicit" });
        body["messages"] = serde_json::json!([
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": system_prompt,
                    "prompt_cache_breakpoint": { "mode": "explicit" }
                }]
            },
            { "role": "user", "content": user_text }
        ]);
    }

    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?
        .error_for_status()?;
    let value: serde_json::Value = resp.json().await?;
    if let Some(cached) = value["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64() {
        tracing::debug!(cached_tokens = cached, model, "openai classify cache usage");
    }
    let content = value["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing OpenAI classify content"))?
        .to_string();
    Ok(content)
}

fn select_decision_steps(sampled: Vec<StepCandidates>) -> Vec<StepCandidates> {
    let with_category: Vec<_> = sampled
        .iter()
        .filter(|s| !s.category_top_tokens.is_empty() || s.decision_diagnostics.is_some())
        .cloned()
        .collect();
    if !with_category.is_empty() {
        return with_category.into_iter().take(3).collect();
    }
    sampled.into_iter().take(2).collect()
}

fn analyze_decision_steps(steps: &[StepCandidates]) -> (bool, bool, Option<f32>) {
    let mut close = false;
    let mut non_grammar = false;
    let mut margin = None;
    for step in steps {
        let (c, ng, m) = analyze_step(step);
        close |= c;
        non_grammar |= ng;
        if margin.is_none() {
            margin = m;
        }
    }
    (close, non_grammar, margin)
}

fn analyze_step(step: &StepCandidates) -> (bool, bool, Option<f32>) {
    let margin = if step.top_alternatives.len() >= 2 {
        Some(step.top_alternatives[0].logit - step.top_alternatives[1].logit)
    } else {
        None
    };
    let close = margin.is_some_and(|m| m.abs() < CLOSE_LOGIT_MARGIN);

    let constrained_ids: std::collections::HashSet<_> = step
        .top_constrained
        .iter()
        .map(|t| t.token_id)
        .collect();
    let non_grammar = step
        .top_alternatives
        .iter()
        .take(5)
        .any(|t| !constrained_ids.contains(&t.token_id));

    (close, non_grammar, margin)
}

fn finish_example(
    example_id: i32,
    example_text: &str,
    expected_message_id: i32,
    expected_category: &str,
    full_text: Option<&str>,
    maps: &CategoryMaps,
    decision_steps: Vec<StepCandidates>,
) -> ExampleEvalResult {
    let predicted_category = full_text.and_then(|ft| match_category_prefix(ft, &maps.category_to_id));
    let predicted_message_id = predicted_category
        .as_ref()
        .and_then(|name| maps.category_to_id.get(name).copied());
    let success = predicted_message_id == Some(expected_message_id);
    let (close_top_logits, non_grammar_top_logits, top_logit_margin) =
        analyze_decision_steps(&decision_steps);
    ExampleEvalResult {
        example_id,
        example_text: example_text.to_string(),
        expected_category: expected_category.to_string(),
        expected_message_id,
        predicted_category,
        predicted_message_id,
        success,
        decision_steps,
        close_top_logits,
        non_grammar_top_logits,
        top_logit_margin,
    }
}

#[derive(Clone)]
struct CategoryMaps {
    id_to_category: HashMap<i32, String>,
    category_to_id: HashMap<String, i32>,
}

impl CategoryMaps {
    fn from_dataset(dataset: &ValidationDataset) -> Self {
        Self {
            id_to_category: dataset
                .messages
                .iter()
                .map(|m| (m.id, m.vc_message.category.clone()))
                .collect(),
            category_to_id: dataset
                .messages
                .iter()
                .map(|m| (m.vc_message.category.clone(), m.id))
                .collect(),
        }
    }

    fn expected_category(&self, message_id: i32) -> String {
        self.id_to_category
            .get(&message_id)
            .cloned()
            .unwrap_or_else(|| format!("message:{message_id}"))
    }
}

pub fn confusions_from_report(report: &EvalReport) -> Vec<ConfusionCase> {
    report
        .confusions()
        .into_iter()
        .map(|r| ConfusionCase {
            example_id: r.example_id,
            example_text: r.example_text.clone(),
            expected_category: r.expected_category.clone(),
            expected_message_id: r.expected_message_id,
            predicted_category: r.predicted_category.clone(),
            predicted_message_id: r.predicted_message_id,
            analysis: None,
            suggested_amendments: vec![],
            decision_steps: r.decision_steps.clone(),
            close_top_logits: r.close_top_logits,
            non_grammar_top_logits: r.non_grammar_top_logits,
            top_logit_margin: r.top_logit_margin,
            is_repeat_offender: false,
            recurrence_count: 1,
            prior_amendment_failures: vec![],
        })
        .collect()
}

/// Aggregate correct examples by expected category, including close-logit counts.
pub fn passing_stats_from_report(
    report: &EvalReport,
) -> (Vec<inference_types::PassingCategoryStat>, usize) {
    use std::collections::BTreeMap;
    let mut by_cat: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut passing_close = 0usize;
    for r in &report.results {
        if !r.success {
            continue;
        }
        let entry = by_cat.entry(r.expected_category.clone()).or_insert((0, 0));
        entry.0 += 1;
        if r.close_top_logits {
            entry.1 += 1;
            passing_close += 1;
        }
    }
    let stats = by_cat
        .into_iter()
        .map(|(category_name, (pass_count, close_logit_count))| {
            inference_types::PassingCategoryStat {
                category_name,
                pass_count,
                close_logit_count,
            }
        })
        .collect();
    (stats, passing_close)
}

fn match_category_prefix(
    full_text: &str,
    category_to_id: &HashMap<String, i32>,
) -> Option<String> {
    let ft_norm = {
        let s = full_text.trim_start();
        let s = s.strip_prefix("Category:").unwrap_or(s);
        s.trim_start()
    };
    category_to_id
        .keys()
        .filter(|name| ft_norm.starts_with(name.as_str()))
        .max_by_key(|name| name.len())
        .cloned()
}
