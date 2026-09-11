use inference_types::{BulkTestEvent, InferenceEvent, StepCandidates};
use leptos::prelude::*;
use wasm_bindgen::{JsCast, closure::Closure};
use web_sys::{EventSource, MessageEvent};

/// GET /agents — returns the list of agents (id + name) that have VC messages.
pub async fn fetch_agents() -> Result<Vec<inference_types::AgentInfo>, String> {
    let resp = gloo_net::http::Request::get("/agents")
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }

    resp.json::<Vec<inference_types::AgentInfo>>()
        .await
        .map_err(|e| e.to_string())
}

/// GET /agents/:agent_id/system-prompt — returns the rendered system prompt text.
pub async fn fetch_system_prompt(agent_id: i32) -> Result<String, String> {
    let resp = gloo_net::http::Request::get(&format!("/agents/{agent_id}/system-prompt"))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }

    resp.text().await.map_err(|e| e.to_string())
}

/// POST /infer — creates a session for the given agent and prompt.
/// Returns the session_id string on success.
pub async fn start_inference(prompt: String, agent_id: i32) -> Result<String, String> {
    let body = serde_json::json!({ "prompt": prompt, "agent_id": agent_id });
    let resp = gloo_net::http::Request::post("/infer")
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    json["session_id"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| "missing session_id in response".to_string())
}

/// Opens an SSE connection to GET /infer/stream/:session_id.
/// Registers onmessage/onerror callbacks that update Leptos signals directly.
/// The EventSource is kept alive via `mem::forget` until Done/Error.
pub fn open_sse_stream(
    session_id: String,
    set_steps: WriteSignal<Vec<StepCandidates>>,
    set_status: WriteSignal<String>,
    set_streaming: WriteSignal<bool>,
    set_llm_latency_ms: WriteSignal<Option<u64>>,
) {
    let url = format!("/infer/stream/{session_id}");
    let es = match EventSource::new(&url) {
        Ok(es) => es,
        Err(e) => {
            set_status.set(format!("EventSource failed: {:?}", e));
            set_streaming.set(false);
            return;
        }
    };

    // -- onmessage ----------------------------------------------------------
    let es_done = es.clone();
    let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
        let data = e.data().as_string().unwrap_or_default();
        match serde_json::from_str::<InferenceEvent>(&data) {
            Ok(InferenceEvent::Token(step)) => {
                set_steps.update(|v| v.push(step));
            }
            Ok(InferenceEvent::Done {
                latency_ms,
                category_latency_ms,
                ..
            }) => {
                set_llm_latency_ms.set(Some(category_latency_ms.unwrap_or(latency_ms)));
                set_status.set("Done".to_string());
                set_streaming.set(false);
                es_done.close();
            }
            Ok(InferenceEvent::Error { message }) => {
                set_status.set(format!("Inference error: {message}"));
                set_streaming.set(false);
                es_done.close();
            }
            Err(_) => {} // ignore unparseable frames
        }
    });
    es.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();

    // -- onerror ------------------------------------------------------------
    let es_err = es.clone();
    let on_error = Closure::<dyn FnMut(_)>::new(move |_: web_sys::Event| {
        set_status.set("Stream connection error".to_string());
        set_streaming.set(false);
        es_err.close();
    });
    es.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    on_error.forget();

    // Keep the EventSource alive; it closes itself when generation finishes
    std::mem::forget(es);
}

/// Summary of one stored bulk test run.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct BulkTestRunSummary {
    pub id: i64,
    pub agent_id: i64,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub total: Option<i64>,
    pub success_count: Option<i64>,
}

/// GET /bulk-tests — list the 50 most recent runs.
pub async fn fetch_bulk_test_runs() -> Result<Vec<BulkTestRunSummary>, String> {
    let resp = gloo_net::http::Request::get("/bulk-tests")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json::<Vec<BulkTestRunSummary>>()
        .await
        .map_err(|e| e.to_string())
}

/// GET /bulk-tests/{run_id} — load all results for a past run.
pub async fn fetch_bulk_test_run(run_id: i64) -> Result<Vec<TestResult>, String> {
    let resp = gloo_net::http::Request::get(&format!("/bulk-tests/{run_id}"))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    // The server returns the same shape as TestResult.
    #[derive(serde::Deserialize)]
    struct Row {
        example_id: i32,
        example_text: String,
        chosen_category: Option<String>,
        correct_categories: Vec<String>,
        success: bool,
        steps: Vec<inference_types::StepCandidates>,
        #[serde(default)]
        classifier_results: Vec<inference_types::ClassifierMethodResult>,
    }
    let rows = resp.json::<Vec<Row>>().await.map_err(|e| e.to_string())?;
    Ok(rows
        .into_iter()
        .map(|r| TestResult {
            example_id: r.example_id,
            example_text: r.example_text,
            chosen_category: r.chosen_category,
            correct_categories: r.correct_categories,
            success: r.success,
            steps: r.steps,
            classifier_results: r.classifier_results,
        })
        .collect())
}

/// POST /bulk-test — creates a bulk test run for the given agent.
/// Returns `(bulk_test_id, run_id)` on success.
pub async fn start_bulk_test(agent_id: i32) -> Result<(String, i64), String> {
    let body = serde_json::json!({ "agent_id": agent_id });
    let resp = gloo_net::http::Request::post("/bulk-test")
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let bulk_test_id = json["bulk_test_id"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| "missing bulk_test_id in response".to_string())?;
    let run_id = json["run_id"]
        .as_i64()
        .ok_or_else(|| "missing run_id in response".to_string())?;
    Ok((bulk_test_id, run_id))
}

/// A single completed bulk test result (extracted from BulkTestEvent::Result).
#[derive(Clone, Debug)]
pub struct TestResult {
    pub example_id: i32,
    pub example_text: String,
    pub chosen_category: Option<String>,
    pub correct_categories: Vec<String>,
    pub success: bool,
    pub steps: Vec<StepCandidates>,
    pub classifier_results: Vec<inference_types::ClassifierMethodResult>,
}

/// POST /bulk-tests/{run_id}/apply-weights — saves per-category kappa values to the DB.
pub async fn apply_weights(
    run_id: i64,
    weights: &std::collections::HashMap<String, f64>,
) -> Result<ApplyWeightsResponse, String> {
    let body = serde_json::json!({ "weights": weights });
    let resp = gloo_net::http::Request::post(&format!("/bulk-tests/{run_id}/apply-weights"))
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json::<ApplyWeightsResponse>()
        .await
        .map_err(|e| e.to_string())
}

/// Response from POST /bulk-tests/{run_id}/apply-weights.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct ApplyWeightsResponse {
    pub updated: usize,
    pub unmatched_categories: Vec<String>,
}

/// POST /bulk-tests/{run_id}/optimize — returns optimal per-category weights.
pub async fn optimize_weights(run_id: i64) -> Result<OptimizeResponse, String> {
    let resp = gloo_net::http::Request::post(&format!("/bulk-tests/{run_id}/optimize"))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json::<OptimizeResponse>()
        .await
        .map_err(|e| e.to_string())
}

/// Response from POST /bulk-tests/{run_id}/optimize.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct OptimizeResponse {
    pub weights: std::collections::HashMap<String, f64>,
    pub examples_used: usize,
    pub examples_skipped: usize,
    /// Server-computed accuracy with default kappa=10.0.
    #[serde(default)]
    pub baseline_accuracy: Option<AccuracyReport>,
    /// Server-computed accuracy with kappa=0 (no embedding bias).
    #[serde(default)]
    pub no_embedding_accuracy: Option<AccuracyReport>,
    /// Server-computed accuracy with the optimised kappa values.
    #[serde(default)]
    pub optimized_accuracy: Option<AccuracyReport>,
    /// Classifier ensemble optimization results.
    #[serde(default)]
    pub ensemble_optimization: Option<EnsembleOptimization>,
    #[serde(default)]
    pub routing_metrics: RoutingMetrics,
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
pub struct RoutingMetrics {
    pub examples_analyzed: usize,
    pub force_gate_count: usize,
    pub force_gate_rate_pct: f64,
    pub soft_override_count: usize,
    pub forced_override_count: usize,
    pub override_correct: usize,
    pub override_accuracy_pct: f64,
    pub forced_correct: usize,
    pub forced_accuracy_pct: f64,
    pub raw_correct: usize,
    pub raw_accuracy_pct: f64,
    pub final_correct: usize,
    pub final_accuracy_pct: f64,
}

/// Accuracy report from the server-side kappa evaluation.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct AccuracyReport {
    pub correct: usize,
    pub total: usize,
    pub accuracy_pct: f64,
    #[serde(default)]
    pub per_category: std::collections::HashMap<String, CategoryAccuracyInfo>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct CategoryAccuracyInfo {
    pub correct: usize,
    pub total: usize,
    pub accuracy_pct: f64,
}

/// Classifier ensemble optimization results (TF-IDF vs embedding weights + LLM analysis).
#[derive(Clone, Debug, serde::Deserialize)]
pub struct EnsembleOptimization {
    pub examples_analyzed: usize,
    pub baseline_accuracy: std::collections::HashMap<String, f64>,
    pub weight_search: Vec<WeightPoint>,
    pub optimal_weights: WeightPoint,
    pub temperature_search: Vec<TemperaturePoint>,
    pub optimal_temperature: TemperaturePoint,
    pub llm_accuracy_pct: Option<f64>,
    pub llm_category_stats: std::collections::HashMap<String, LlmCategoryStats>,
    pub category_bias_offsets: Vec<CategoryBiasOffset>,
    pub bias_corrected_accuracy_pct: Option<f64>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct WeightPoint {
    pub tfidf_weight: f64,
    pub embedding_weight: f64,
    pub correct: usize,
    pub total: usize,
    pub accuracy_pct: f64,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct TemperaturePoint {
    pub temperature: f64,
    pub correct: usize,
    pub total: usize,
    pub accuracy_pct: f64,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct LlmCategoryStats {
    pub avg_logit: f64,
    pub avg_probability: f64,
    pub times_chosen: usize,
    pub times_correct: usize,
    pub sample_count: usize,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct CategoryBiasOffset {
    pub category_name: String,
    pub bias_offset: f64,
    pub sample_count: usize,
}

/// Opens an SSE connection to GET /bulk-test/stream/:bulk_test_id.
/// Appends a `TestResult` to `set_results` for each completed test case.
/// Calls `set_status` and `set_running(false)` when Done or Error.
pub fn open_bulk_test_stream(
    bulk_test_id: String,
    set_results: WriteSignal<Vec<TestResult>>,
    set_status: WriteSignal<String>,
    set_running: WriteSignal<bool>,
    set_total: WriteSignal<usize>,
) {
    let url = format!("/bulk-test/stream/{bulk_test_id}");
    let es = match EventSource::new(&url) {
        Ok(es) => es,
        Err(e) => {
            set_status.set(format!("EventSource failed: {:?}", e));
            set_running.set(false);
            return;
        }
    };

    let es_done = es.clone();
    let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
        let data = e.data().as_string().unwrap_or_default();
        match serde_json::from_str::<BulkTestEvent>(&data) {
            Ok(BulkTestEvent::Result {
                example_id,
                example_text,
                chosen_category,
                correct_categories,
                success,
                steps,
                classifier_results,
            }) => {
                set_results.update(|v| {
                    v.push(TestResult {
                        example_id,
                        example_text,
                        chosen_category,
                        correct_categories,
                        success,
                        steps,
                        classifier_results,
                    })
                });
            }
            Ok(BulkTestEvent::Done {
                total,
                success_count,
            }) => {
                set_total.set(total);
                set_status.set(format!(
                    "Done — {success_count}/{total} ({:.0}%)",
                    if total == 0 {
                        0.0
                    } else {
                        success_count as f64 / total as f64 * 100.0
                    }
                ));
                set_running.set(false);
                es_done.close();
            }
            Ok(BulkTestEvent::Error { message }) => {
                set_status.set(format!("Bulk test error: {message}"));
                set_running.set(false);
                es_done.close();
            }
            Err(_) => {}
        }
    });
    es.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();

    let es_err = es.clone();
    let on_error = Closure::<dyn FnMut(_)>::new(move |_: web_sys::Event| {
        set_status.set("Bulk test stream connection error".to_string());
        set_running.set(false);
        es_err.close();
    });
    es.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    on_error.forget();

    std::mem::forget(es);
}

// ---------------------------------------------------------------------------
// Classifier API
// ---------------------------------------------------------------------------

/// GET /classify/methods — list classifier method names available on the server.
pub async fn fetch_classifier_methods() -> Result<Vec<String>, String> {
    let resp = gloo_net::http::Request::get("/classify/methods")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json::<Vec<String>>().await.map_err(|e| e.to_string())
}

/// POST /classify — classify a single query with all available methods.
pub async fn classify_single(
    prompt: &str,
    agent_id: i32,
) -> Result<Vec<inference_types::ClassifierMethodResult>, String> {
    #[derive(serde::Deserialize)]
    struct ClassificationResultRaw {
        method_name: String,
        chosen_category: String,
        confidence: f64,
        latency_ms: u64,
        scores: Vec<inference_types::ClassifierScore>,
    }
    #[derive(serde::Deserialize)]
    struct Resp {
        results: Vec<ClassificationResultRaw>,
    }
    let body = serde_json::json!({ "prompt": prompt, "agent_id": agent_id });
    let resp = gloo_net::http::Request::post("/classify")
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let r: Resp = resp.json().await.map_err(|e| e.to_string())?;
    Ok(r.results
        .into_iter()
        .map(|x| inference_types::ClassifierMethodResult {
            method_name: x.method_name,
            chosen_category: x.chosen_category,
            confidence: x.confidence,
            latency_ms: x.latency_ms,
            scores: x.scores,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// LLM validation loop
// ---------------------------------------------------------------------------

pub async fn fetch_agent_versions() -> Result<Vec<inference_types::AgentVersionInfo>, String> {
    let resp = gloo_net::http::Request::get("/agent-versions")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    resp.json().await.map_err(|e| e.to_string())
}

pub async fn start_llm_validation(
    agent_id: i32,
    version_id: i32,
    classifier_backend: &str,
) -> Result<String, String> {
    let body = serde_json::json!({
        "agent_id": agent_id,
        "version_id": version_id,
        "classifier_backend": classifier_backend,
    });
    let resp = gloo_net::http::Request::post("/llm-validation")
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    json["session_id"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| "missing session_id".into())
}

pub async fn decide_llm_validation(session_id: &str, approve: bool) -> Result<String, String> {
    let body = serde_json::json!({ "approve": approve });
    let resp = gloo_net::http::Request::post(&format!("/llm-validation/{session_id}/decision"))
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    json["session_id"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| "missing session_id".into())
}

pub async fn cancel_llm_validation(session_id: &str) -> Result<(), String> {
    let resp = gloo_net::http::Request::post(&format!("/llm-validation/{session_id}/cancel"))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ProposalState {
    pub best_round: u32,
    pub best_accuracy_pct: f64,
    pub amendments: Vec<inference_types::ValidationAmendment>,
    pub executive_summary: Option<inference_types::AnalysisExecutiveSummary>,
}

pub fn open_llm_validation_stream(
    session_id: String,
    set_status: WriteSignal<String>,
    set_running: WriteSignal<bool>,
    set_rounds: WriteSignal<Vec<inference_types::ValidationRoundSummary>>,
    set_proposal: WriteSignal<Option<ProposalState>>,
    set_pending_session: WriteSignal<Option<String>>,
    decision_session: String,
) {
    let url = format!("/llm-validation/stream/{session_id}");
    let es = match EventSource::new(&url) {
        Ok(es) => es,
        Err(e) => {
            set_status.set(format!("EventSource failed: {:?}", e));
            set_running.set(false);
            return;
        }
    };

    let es_done = es.clone();
    let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
        let data = e.data().as_string().unwrap_or_default();
        match serde_json::from_str::<inference_types::LlmValidationEvent>(&data) {
            Ok(inference_types::LlmValidationEvent::Status { message }) => {
                set_status.set(message);
            }
            Ok(inference_types::LlmValidationEvent::RoundStarted { round }) => {
                set_status.set(format!("Round {round} started…"));
            }
            Ok(inference_types::LlmValidationEvent::ExampleResult {
                round,
                example_id,
                success,
                expected_category,
                predicted_category,
                close_top_logits,
                non_grammar_top_logits,
            }) => {
                let mark = if success { "✓" } else { "✗" };
                let mut flags = String::new();
                if close_top_logits {
                    flags.push_str(" [close logits]");
                }
                if non_grammar_top_logits {
                    flags.push_str(" [non-grammar top]");
                }
                set_status.set(format!(
                    "R{round} {mark} #{example_id} expected={expected_category} got={predicted_category:?}{flags}"
                ));
            }
            Ok(inference_types::LlmValidationEvent::RoundCompleted { summary }) => {
                set_status.set(format!(
                    "Round {} done: {:.1}% — {} confusion(s)",
                    summary.round,
                    summary.accuracy_pct,
                    summary.confusions.len()
                ));
                set_rounds.update(|v| {
                    if let Some(existing) = v.iter_mut().find(|r| r.round == summary.round) {
                        *existing = summary;
                    } else {
                        v.push(summary);
                    }
                });
            }
            Ok(inference_types::LlmValidationEvent::Analyzing { confusion_count }) => {
                set_status.set(format!("Analyzing {confusion_count} confusions with OpenAI…"));
            }
            Ok(inference_types::LlmValidationEvent::ProposalReady {
                best_round,
                best_accuracy_pct,
                amendments,
                rounds,
                executive_summary,
            }) => {
                set_rounds.set(rounds);
                set_proposal.set(Some(ProposalState {
                    best_round,
                    best_accuracy_pct,
                    amendments,
                    executive_summary,
                }));
                set_pending_session.set(Some(decision_session.clone()));
                set_status.set(format!(
                    "Best round {best_round} at {best_accuracy_pct:.1}% — review amendments"
                ));
                set_running.set(false);
                es_done.close();
            }
            Ok(inference_types::LlmValidationEvent::Applied { applied_count }) => {
                set_status.set(format!("Applied {applied_count} amendments to Postgres"));
            }
            Ok(inference_types::LlmValidationEvent::DeniedRetest {
                total,
                success_count,
                accuracy_pct,
            }) => {
                set_status.set(format!(
                    "Denied retest: {success_count}/{total} ({accuracy_pct:.1}%)"
                ));
                set_proposal.set(None);
                set_pending_session.set(None);
            }
            Ok(inference_types::LlmValidationEvent::Cancelled { message }) => {
                set_status.set(format!("Cancelled: {message}"));
                set_running.set(false);
                set_pending_session.set(None);
                es_done.close();
            }
            Ok(inference_types::LlmValidationEvent::Done) => {
                set_running.set(false);
                es_done.close();
            }
            Ok(inference_types::LlmValidationEvent::Error { message }) => {
                set_status.set(format!("Error: {message}"));
                set_running.set(false);
                es_done.close();
            }
            Err(_) => {}
        }
    });
    es.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();

    let es_err = es.clone();
    let on_error = Closure::<dyn FnMut(web_sys::Event)>::new(move |_e| {
        set_status.set("SSE connection error".into());
        set_running.set(false);
        es_err.close();
    });
    es.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    on_error.forget();
    std::mem::forget(es);
}
