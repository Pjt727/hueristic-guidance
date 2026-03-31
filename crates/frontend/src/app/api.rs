use inference_types::{InferenceEvent, StepCandidates};
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
