use inference_types::{BulkTestEvent, InferenceEvent, StepCandidates};
use leptos::prelude::*;
use wasm_bindgen::{JsCast, closure::Closure};
use web_sys::{EventSource, MessageEvent};

/// GET /agents — returns the list of agent IDs that have VC messages.
pub async fn fetch_agents() -> Result<Vec<i32>, String> {
    let resp = gloo_net::http::Request::get("/agents")
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }

    resp.json::<Vec<i32>>().await.map_err(|e| e.to_string())
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
            Ok(InferenceEvent::Done { .. }) => {
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
    resp.json::<Vec<BulkTestRunSummary>>().await.map_err(|e| e.to_string())
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
        })
        .collect())
}

/// POST /bulk-test — creates a bulk test run for the given agent.
/// Returns the bulk_test_id string on success.
pub async fn start_bulk_test(agent_id: i32) -> Result<String, String> {
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
    json["bulk_test_id"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| "missing bulk_test_id in response".to_string())
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
            }) => {
                set_results.update(|v| {
                    v.push(TestResult {
                        example_id,
                        example_text,
                        chosen_category,
                        correct_categories,
                        success,
                        steps,
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
