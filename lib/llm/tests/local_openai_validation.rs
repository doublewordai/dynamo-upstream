// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The loopback OpenAI endpoint that worker validation gates check.
//!
//! Each test runs in a fresh subprocess: gates are read from the environment
//! when the `DistributedRuntime` starts, and the TCP request plane listener is
//! process-global.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use dynamo_llm::backend::ExecutionContext;
use dynamo_llm::discovery::{ModelManager, ModelWatcher};
use dynamo_llm::entrypoint::input::{endpoint, local_openai};
use dynamo_llm::entrypoint::{EngineConfig, RouterConfig};
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_llm::local_model::{LocalModel, LocalModelBuilder};
use dynamo_llm::model_type::{ModelInput, ModelType};
use dynamo_llm::namespace::NamespaceFilter;
use dynamo_llm::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};
use dynamo_runtime::discovery::{DiscoveryQuery, EventTransportKind};
use dynamo_runtime::distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode};
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::storage::kv;
use dynamo_runtime::{CancellationToken, DistributedRuntime, Runtime};
use serde_json::{Value, json};

const MODEL_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1";
const MODEL: &str = "validation-parity";
const WINDOW: Duration = Duration::from_secs(30);
/// What the scripted engine answers: reasoning (deepseek_r1 starts in it),
/// then a hermes tool call.
const REPLY: &str = "The user wants the weather in Paris.</think>\
                     <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
const TEST_ENV: &str = "DYNAMO_LOCAL_OPENAI_TEST";
const SCRATCH_ENV: &str = "DYNAMO_LOCAL_OPENAI_TEST_SCRATCH";

/// Re-run test `name` alone in a child process with `envs` set. Returns false
/// when already in that child, which should then run the test body.
async fn in_fresh_process(name: &str, envs: &[(&str, String)]) -> bool {
    if std::env::var(TEST_ENV).as_deref() == Ok(name) {
        dynamo_runtime::logging::init();
        return false;
    }
    let output = tokio::time::timeout(
        Duration::from_secs(120),
        tokio::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture"])
            .env(TEST_ENV, name)
            .env("DYN_TCP_RPC_HOST", "127.0.0.1")
            .env("DYN_TCP_RPC_PORT", "0")
            .env("DYN_TCP_RESPONSE_STREAM_HOST", "127.0.0.1")
            .env("DYN_TCP_RESPONSE_STREAM_PORT", "0")
            .env("DYN_SYSTEM_PORT", "0")
            .env_remove("DYN_WORKER_VALIDATION_GATES")
            .env_remove("DYN_WORKER_VALIDATION_HOLD")
            .env_remove("DYN_WORKER_VALIDATION_LOCAL_PORT")
            .envs(envs.iter().map(|(key, value)| (*key, value.as_str())))
            .kill_on_drop(true)
            .output(),
    )
    .await
    .expect("test subprocess must finish")
    .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("1 passed; 0 failed"),
        "subprocess must run the selected test: {stdout}"
    );
    true
}

/// A gate that records the environment it was given, then holds until the
/// test writes `release`, so the endpoint stays up while the test uses it.
fn recording_gate(scratch: &Path) -> String {
    let script = format!(
        "printf '%s\\n%s\\n' \"$OPENAI_BASE_URL\" \"$DYN_VALIDATION_MODEL\" > {dir}/env.tmp \
         && mv {dir}/env.tmp {dir}/env \
         && while [ ! -f {dir}/release ]; do sleep 0.05; done",
        dir = scratch.display()
    );
    json!({"record": {"exec": ["sh", "-c", script], "timeout_secs": 120}}).to_string()
}

/// The gate's `(OPENAI_BASE_URL, DYN_VALIDATION_MODEL)`, once it has started.
async fn gate_env(scratch: &Path) -> (String, String) {
    let path = scratch.join("env");
    tokio::time::timeout(WINDOW, async {
        loop {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                let mut lines = contents.lines().map(str::to_string);
                return (lines.next().unwrap(), lines.next().unwrap());
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the gate must start")
}

fn scratch() -> PathBuf {
    PathBuf::from(std::env::var(SCRATCH_ENV).unwrap())
}

fn config() -> DistributedConfig {
    DistributedConfig {
        discovery_backend: DiscoveryBackend::KvStore(kv::Selector::Memory),
        nats_config: None,
        request_plane: RequestPlaneMode::Tcp,
        response_plane: None,
        event_transport_kind: EventTransportKind::Zmq,
    }
}

fn model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_PATH)
}

async fn local_model(name: &str) -> LocalModel {
    LocalModelBuilder::default()
        .model_path(model_path())
        .source_path(model_path())
        .model_name(Some(name.to_string()))
        .runtime_config(ModelRuntimeConfig {
            tool_call_parser: Some("hermes".to_string()),
            reasoning_parser: Some("deepseek_r1".to_string()),
            ..Default::default()
        })
        // Keep local file paths in the card: frontend and worker share a disk.
        .self_host_metadata(false)
        .build()
        .await
        .unwrap()
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(WINDOW)
        .build()
        .unwrap()
}

fn chat_request(stream: bool) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
        "max_tokens": 128,
        "stream": stream,
        "nvext": {"annotations": ["output_replay_id:parity"]},
    })
}

/// POST a chat request and return the response bodies: one for a unary
/// request, one per SSE event for a streaming one.
async fn chat(base_url: &str, stream: bool) -> (reqwest::StatusCode, Vec<Value>) {
    let response = client()
        .post(format!("{base_url}/chat/completions"))
        .json(&chat_request(stream))
        .send()
        .await
        .unwrap();
    let status = response.status();
    let text = response.text().await.unwrap();
    if !stream || !status.is_success() {
        return (status, vec![serde_json::from_str(&text).unwrap()]);
    }
    let events = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    (status, events)
}

/// Drop the fields that differ between any two requests.
fn normalize(mut body: Value) -> Value {
    if let Some(object) = body.as_object_mut() {
        object.remove("id");
        object.remove("created");
    }
    if let Some(choices) = body["choices"].as_array_mut() {
        for choice in choices {
            for part in ["message", "delta"] {
                let calls = choice
                    .get_mut(part)
                    .and_then(|part| part.get_mut("tool_calls"))
                    .and_then(Value::as_array_mut);
                for call in calls.into_iter().flatten() {
                    if let Some(call) = call.as_object_mut() {
                        call.remove("id");
                    }
                }
            }
        }
    }
    body
}

/// The frontend: discovery, the default chat processor, an HTTP service.
async fn start_frontend(
    drt: &DistributedRuntime,
    cancel: &CancellationToken,
    model: &str,
) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let service = HttpService::builder()
        .host("127.0.0.1")
        .port(port)
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .build()
        .unwrap();
    let manager: Arc<ModelManager> = service.state().manager_clone();
    let stream = drt
        .discovery()
        .list_and_watch(DiscoveryQuery::AllModels, Some(cancel.clone()))
        .await
        .unwrap();
    let mut watcher = ModelWatcher::new(
        drt.clone(),
        manager.clone(),
        RouterConfig {
            router_mode: dynamo_runtime::pipeline::RouterMode::RoundRobin,
            ..Default::default()
        },
        0,
        None,
        None,
        None,
        service.state().metrics_clone(),
    );
    // As `--model-path` does for a frontend sharing the worker's disk.
    watcher.set_local_model_path(Some(model_path()));
    let watcher = Arc::new(watcher);
    tokio::spawn(watcher.watch(stream, NamespaceFilter::Global));
    let _http = service.spawn_with_listener(cancel.clone(), listener).await;
    tokio::time::timeout(WINDOW, async {
        while !manager.is_model_ready_to_serve(model) {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the frontend must discover the worker");
    format!("http://127.0.0.1:{port}/v1")
}

/// The frontend and the loopback endpoint answer the same request to the same
/// mocker worker identically, with reasoning and tool calls parsed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn loopback_matches_frontend_on_the_mocker() {
    let dir = tempfile::tempdir().unwrap();
    let name = "loopback_matches_frontend_on_the_mocker";
    if in_fresh_process(
        name,
        &[
            (SCRATCH_ENV, dir.path().display().to_string()),
            ("DYN_WORKER_VALIDATION_GATES", recording_gate(dir.path())),
            // Publish the card so the frontend can serve the same worker for
            // comparison; /health stays held until release.
            ("DYN_WORKER_VALIDATION_HOLD", "health".to_string()),
        ],
    )
    .await
    {
        return;
    }
    let scratch = scratch();

    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), config())
        .await
        .unwrap();
    let model = local_model(MODEL).await;

    // Script the mocker's output for requests annotated `output_replay_id:parity`.
    let reply = model
        .card()
        .tokenizer()
        .unwrap()
        .encode(REPLY)
        .unwrap()
        .token_ids()
        .to_vec();
    let trace = scratch.join("replay.jsonl");
    std::fs::write(
        &trace,
        json!({"request_id": "parity", "output_length": reply.len(), "output_token_ids": reply})
            .to_string(),
    )
    .unwrap();
    let mocker = dynamo_llm::mocker::make_mocker_engine(
        drt.clone(),
        "parity.mocker.generate".parse().unwrap(),
        dynamo_mocker::common::protocols::MockerConfig::from_value(json!({
            "engine": {
                "block_size": 16,
                "num_gpu_blocks": 1024,
                "max_num_batched_tokens": 1024,
                "speedup_ratio": 1000.0,
                "enable_prefix_caching": false,
            },
            "dynamo": {"response_replay_trace_path": trace},
        }))
        .unwrap(),
    )
    .await
    .unwrap();

    // The worker, exactly as `python -m dynamo.mocker` runs it.
    let worker = tokio::spawn(endpoint::run(
        drt.clone(),
        "dyn://parity.mocker.generate".to_string(),
        EngineConfig::InProcessTokens {
            engine: mocker,
            model: Box::new(model),
            is_prefill: false,
            is_decode: false,
        },
    ));

    let cancel = CancellationToken::new();
    let frontend = start_frontend(&drt, &cancel, MODEL).await;
    let (loopback, gate_model) = gate_env(&scratch).await;
    assert_eq!(gate_model, MODEL);
    assert!(
        loopback.starts_with("http://127.0.0.1:") && loopback.ends_with("/v1"),
        "{loopback}"
    );

    for stream in [false, true] {
        let (status, from_frontend) = chat(&frontend, stream).await;
        assert!(status.is_success(), "{status}: {from_frontend:?}");
        let (status, from_loopback) = chat(&loopback, stream).await;
        assert!(status.is_success(), "{status}: {from_loopback:?}");
        let from_frontend: Vec<_> = from_frontend.into_iter().map(normalize).collect();
        let from_loopback: Vec<_> = from_loopback.into_iter().map(normalize).collect();
        assert_eq!(
            from_loopback, from_frontend,
            "stream={stream}: loopback output differs from the frontend's"
        );

        // Parity is only meaningful if the parsers ran.
        let mut reasoning = String::new();
        let mut tool_names = Vec::new();
        for body in &from_frontend {
            let choice = &body["choices"][0];
            let part = if stream {
                &choice["delta"]
            } else {
                &choice["message"]
            };
            reasoning.push_str(part["reasoning_content"].as_str().unwrap_or_default());
            for call in part["tool_calls"].as_array().into_iter().flatten() {
                if let Some(name) = call["function"]["name"].as_str() {
                    tool_names.push(name.to_string());
                }
            }
        }
        assert_eq!(
            reasoning.trim(),
            "The user wants the weather in Paris.",
            "{from_frontend:?}"
        );
        assert_eq!(tool_names, ["get_weather"], "{from_frontend:?}");
    }

    // A passed gate releases the worker and closes the endpoint.
    std::fs::write(scratch.join("release"), "").unwrap();
    let validation = drt.worker_validation().unwrap();
    tokio::time::timeout(WINDOW, validation.wait_released())
        .await
        .expect("the gate must pass");
    tokio::time::timeout(WINDOW, async {
        while client()
            .get(format!("{loopback}/models"))
            .send()
            .await
            .is_ok()
        {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the endpoint must close once the worker is released");

    cancel.cancel();
    worker.abort();
    runtime.shutdown();
}

/// Without gates, no endpoint is served and no local engine is registered.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ungated_worker_opens_no_listener() {
    let name = "ungated_worker_opens_no_listener";
    let port = validation_port();
    if in_fresh_process(
        name,
        &[("DYN_WORKER_VALIDATION_LOCAL_PORT", port.to_string())],
    )
    .await
    {
        return;
    }

    // Unique, so another process serving the shared fixture cannot match.
    let model = format!("ungated-{}", std::process::id());
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), config())
        .await
        .unwrap();
    assert!(drt.worker_validation().is_none());
    let worker = tokio::spawn(endpoint::run(
        drt.clone(),
        "dyn://ungated.worker.generate".to_string(),
        EngineConfig::InProcessTokens {
            engine: Arc::new(Scripted),
            model: Box::new(local_model(&model).await),
            is_prefill: false,
            is_decode: false,
        },
    ));
    let cancel = CancellationToken::new();
    start_frontend(&drt, &cancel, &model).await;

    // Another process may have taken the port since the parent picked it, so
    // only a listener serving this worker's model counts.
    if let Ok(response) = client()
        .get(format!("http://127.0.0.1:{port}/v1/models"))
        .send()
        .await
    {
        let body = response.text().await.unwrap_or_default();
        assert!(
            !body.contains(&model),
            "the validation endpoint must not be served: {body}"
        );
    }
    assert!(drt.local_endpoint_registry().get("generate").is_none());

    cancel.cancel();
    worker.abort();
    runtime.shutdown();
}

/// Before the worker's engine is registered the endpoint answers 503 and the
/// gates wait; registering the engine starts them with the endpoint's URL.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn endpoint_is_unavailable_until_the_engine_registers() {
    let dir = tempfile::tempdir().unwrap();
    let name = "endpoint_is_unavailable_until_the_engine_registers";
    let port = validation_port();
    if in_fresh_process(
        name,
        &[
            (SCRATCH_ENV, dir.path().display().to_string()),
            ("DYN_WORKER_VALIDATION_GATES", recording_gate(dir.path())),
            ("DYN_WORKER_VALIDATION_LOCAL_PORT", port.to_string()),
        ],
    )
    .await
    {
        return;
    }
    let scratch = scratch();

    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), config())
        .await
        .unwrap();
    let endpoint = drt
        .namespace("held")
        .unwrap()
        .component("worker")
        .unwrap()
        .endpoint("generate");
    let mut card = local_model(MODEL).await.into_card();
    card.model_input = ModelInput::Tokens;
    card.model_type = ModelType::Chat | ModelType::Completions;
    local_openai::serve_if_gated(&endpoint, &card)
        .await
        .unwrap();
    // A second card, such as a LoRA adapter, gets no second endpoint.
    local_openai::serve_if_gated(&endpoint, &card)
        .await
        .unwrap();

    let base_url = format!("http://127.0.0.1:{port}/v1");
    let (status, body) = chat(&base_url, false).await;
    assert_eq!(status, reqwest::StatusCode::SERVICE_UNAVAILABLE, "{body:?}");
    assert_eq!(
        drt.worker_validation().unwrap().status_json()["gates"]["record"]["state"],
        "pending"
    );

    drt.local_endpoint_registry()
        .register("generate".to_string(), Arc::new(JsonScripted));
    let (url, model) = gate_env(&scratch).await;
    assert_eq!(url, base_url);
    assert_eq!(model, MODEL);
    let (status, body) = chat(&base_url, false).await;
    assert!(status.is_success(), "{status}: {body:?}");
    assert_eq!(
        body[0]["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );

    std::fs::write(scratch.join("release"), "").unwrap();
    runtime.shutdown();
}

/// The fixed validation port: picked by the parent, inherited by the child.
fn validation_port() -> u16 {
    std::env::var("DYN_WORKER_VALIDATION_LOCAL_PORT")
        .ok()
        .and_then(|port| port.parse().ok())
        .unwrap_or_else(free_port)
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Token engine that answers [`REPLY`] in one chunk.
struct Scripted;

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for Scripted
{
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (_request, context) = input.into_parts();
        let card = local_model(MODEL).await.into_card();
        let token_ids = card.tokenizer()?.encode(REPLY)?.token_ids().to_vec();
        let output = LLMEngineOutput {
            token_ids,
            ..LLMEngineOutput::stop()
        };
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([Annotated::from_data(output)])),
            context.context(),
        ))
    }
}

/// [`Scripted`] behind the JSON surface workers register locally.
struct JsonScripted;

#[async_trait]
impl AsyncEngine<SingleIn<Value>, ManyOut<Annotated<Value>>, Error> for JsonScripted {
    async fn generate(&self, input: SingleIn<Value>) -> Result<ManyOut<Annotated<Value>>, Error> {
        let (request, context) = input.into_parts();
        let request: PreprocessedRequest = serde_json::from_value(request)?;
        let engine: ExecutionContext = Arc::new(Scripted);
        let stream = engine.generate(context.map(|_| request)).await?;
        let context = stream.context();
        Ok(ResponseStream::new(
            Box::pin(futures::StreamExt::map(stream, |annotated| {
                annotated.map_data(|output| serde_json::to_value(output).map_err(|e| e.to_string()))
            })),
            context,
        ))
    }
}
