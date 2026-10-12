// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loopback OpenAI endpoint for worker validation gates.
//!
//! A gated worker is unreachable by design: its card is held, so no frontend
//! routes to it. Its gates still need to send it OpenAI requests that go
//! through the frontend's preprocessing, tool-call and reasoning parsers. This
//! module serves the worker's model on `127.0.0.1` with the same pipeline the
//! in-process token HTTP input builds (`EngineConfig::InProcessTokens`), and
//! starts the gates once it is serving.
//!
//! The engine behind it is the worker's own local engine, looked up per
//! request in the `LocalEndpointRegistry` and reached through
//! [`RegistryEngine`], which serializes requests and responses the same way the
//! network transport does. Requests before the engine is registered get 503.
//!
//! Nothing here runs unless `DYN_WORKER_VALIDATION_GATES` is set.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use anyhow::Context as _;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::config::environment_names::runtime::validation as env_validation;
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::local_endpoint_registry::{LocalAsyncEngine, LocalEndpointRegistry};
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::validation::{GateEnv, WorkerValidation};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::backend::ExecutionContext;
use crate::http::service::service_v2::HttpService;
use crate::model_card::ModelDeploymentCard;
use crate::model_type::ModelInput;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};

/// Serve `card` on the loopback endpoint and start the gates, if this worker is
/// gated and `card` is the first one the endpoint can serve.
///
/// Binds before returning, so a bad `DYN_WORKER_VALIDATION_LOCAL_PORT` or a
/// port in use fails model registration instead of leaving the worker held
/// with no gates running.
///
/// A card the endpoint can't serve (prefill, encode, embeddings, text input)
/// still starts the gates, without `OPENAI_BASE_URL`: `exec` gates that don't
/// need the endpoint can pass, and anything else fails with a reason instead
/// of holding the worker silently.
pub async fn serve_if_gated(endpoint: &Endpoint, card: &ModelDeploymentCard) -> anyhow::Result<()> {
    let drt = endpoint.drt();
    let Some(validation) = drt.worker_validation() else {
        return Ok(());
    };
    if !validation.claim_local_endpoint() {
        tracing::info!(
            model = %card.display_name,
            "Validation gates already run against an earlier card; this one is not checked"
        );
        return Ok(());
    }
    if !servable(card) {
        tracing::warn!(
            model = %card.display_name,
            "Card has no token chat or completions surface; validation gates run without OPENAI_BASE_URL"
        );
        tokio::spawn(start_gates_when_registered(
            Arc::clone(validation),
            drt.local_endpoint_registry().clone(),
            endpoint.name().to_string(),
            GateEnv {
                openai_base_url: None,
                model: Some(card.display_name.clone()),
            },
            drt.child_token(),
            CancellationToken::new(),
        ));
        return Ok(());
    }

    match serve(endpoint, card, validation).await {
        Ok(()) => Ok(()),
        Err(error) => {
            // Let a later card try; this one failed before anything started.
            validation.release_local_endpoint_claim();
            Err(error)
        }
    }
}

async fn serve(
    endpoint: &Endpoint,
    card: &ModelDeploymentCard,
    validation: &Arc<WorkerValidation>,
) -> anyhow::Result<()> {
    let drt = endpoint.drt();
    let port = local_port()?;
    let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port)))
        .await
        .with_context(|| {
            format!(
                "binding the validation endpoint on 127.0.0.1:{port} \
                 (set {} to change it)",
                env_validation::DYN_WORKER_VALIDATION_LOCAL_PORT
            )
        })?;
    let addr = listener.local_addr()?;

    let cancel = drt.primary_token().child_token();
    let http_service = HttpService::builder()
        .host(Ipv4Addr::LOCALHOST.to_string())
        .port(addr.port())
        .cancel_token(Some(cancel.clone()))
        // `DYN_ENABLE_RL` on a worker is for its own RL endpoint, not this one.
        .enable_rl(false)
        .build()?;
    let engine: ExecutionContext = Arc::new(RegistryEngine::new(
        drt.local_endpoint_registry().clone(),
        endpoint.name(),
    ));
    super::http::serve_in_process_tokens(&http_service, card, engine).await?;

    tracing::info!(
        %addr,
        model = %card.display_name,
        endpoint = endpoint.name(),
        "Serving the validation endpoint"
    );
    let serving = http_service
        .spawn_with_listener(cancel.clone(), listener)
        .await;
    tokio::spawn(async move {
        if let Ok(Err(error)) = serving.await {
            tracing::error!(%error, "Validation endpoint stopped");
        }
    });

    let env = GateEnv {
        openai_base_url: Some(format!("http://{addr}/v1")),
        model: Some(card.display_name.clone()),
    };
    tokio::spawn(start_gates_when_registered(
        Arc::clone(validation),
        drt.local_endpoint_registry().clone(),
        endpoint.name().to_string(),
        env,
        drt.child_token(),
        cancel,
    ));
    Ok(())
}

/// Whether the loopback endpoint can serve `card`: a base model that takes
/// tokens and has a chat or completions surface. Prefill, encode, embedding
/// and text-input workers, and LoRA adapters, are not served.
fn servable(card: &ModelDeploymentCard) -> bool {
    card.model_input == ModelInput::Tokens
        && (card.model_type.supports_chat() || card.model_type.supports_completions())
        && card.lora.is_none()
}

fn local_port() -> anyhow::Result<u16> {
    let name = env_validation::DYN_WORKER_VALIDATION_LOCAL_PORT;
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => value
            .trim()
            .parse()
            .with_context(|| format!("{name} must be a port number, got {value:?}")),
        _ => Ok(0),
    }
}

/// Start the gates once the worker's engine is registered, so their first
/// request does not race it. Stop the endpoint once the worker is released:
/// gates run once, and a released worker takes traffic through its frontend.
/// A failed gate keeps it up, for debugging the held worker.
async fn start_gates_when_registered(
    validation: Arc<WorkerValidation>,
    registry: LocalEndpointRegistry,
    endpoint_name: String,
    env: GateEnv,
    gates_cancel: CancellationToken,
    endpoint_cancel: CancellationToken,
) {
    let mut registrations = registry.subscribe();
    tokio::select! {
        result = registrations.wait_for(|_| registry.get(&endpoint_name).is_some()) => {
            if result.is_err() {
                return;
            }
        }
        _ = endpoint_cancel.cancelled() => return,
    }
    validation.start(env, gates_cancel);
    tokio::select! {
        _ = validation.wait_released() => endpoint_cancel.cancel(),
        _ = endpoint_cancel.cancelled() => {}
    }
}

/// Token engine that forwards to a local engine looked up by endpoint name on
/// every request. It is the inverse of the JSON adapters workers register:
/// requests are serialized and responses deserialized with serde, exactly as
/// over the request plane, so the parsers see what a frontend would see.
pub struct RegistryEngine {
    registry: LocalEndpointRegistry,
    endpoint_name: String,
}

impl RegistryEngine {
    pub fn new(registry: LocalEndpointRegistry, endpoint_name: impl Into<String>) -> Self {
        Self {
            registry,
            endpoint_name: endpoint_name.into(),
        }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for RegistryEngine
{
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let Some(engine) = self.registry.get(&self.endpoint_name) else {
            return Err(DynamoError::builder()
                .error_type(ErrorType::Unavailable)
                .message(format!(
                    "local engine for endpoint '{}' is not registered yet",
                    self.endpoint_name
                ))
                .build()
                .into());
        };
        let (request, context) = input.into_parts();
        let request = serde_json::to_value(&request).context("serializing request")?;
        let stream = engine.generate(context.map(|_| request)).await?;
        let context = stream.context();
        let stream = stream.map(|annotated| {
            annotated.map_data(|value| {
                serde_json::from_value::<LLMEngineOutput>(value)
                    .map_err(|error| format!("invalid engine output: {error}"))
            })
        });
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

/// JSON-shaped wrapper around a token engine, for registering it in the
/// `LocalEndpointRegistry` so [`RegistryEngine`] can reach it.
pub(crate) struct JsonEngine {
    inner: ExecutionContext,
}

impl JsonEngine {
    pub(crate) fn local(inner: ExecutionContext) -> LocalAsyncEngine {
        Arc::new(Self { inner })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<serde_json::Value>, ManyOut<Annotated<serde_json::Value>>, Error>
    for JsonEngine
{
    async fn generate(
        &self,
        input: SingleIn<serde_json::Value>,
    ) -> Result<ManyOut<Annotated<serde_json::Value>>, Error> {
        let (request, context) = input.into_parts();
        let request: PreprocessedRequest =
            serde_json::from_value(request).context("deserializing request")?;
        let stream = self.inner.generate(context.map(|_| request)).await?;
        let context = stream.context();
        let stream = stream.map(|annotated| {
            annotated.map_data(|output| serde_json::to_value(output).map_err(|e| e.to_string()))
        });
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serves_only_token_base_models_with_an_openai_surface() {
        use crate::model_card::LoraInfo;
        use crate::model_type::ModelType;

        let mut card = ModelDeploymentCard::with_name_only("m");
        card.model_input = ModelInput::Tokens;
        card.model_type = ModelType::Chat | ModelType::Completions;
        assert!(servable(&card));

        let mut prefill = card.clone();
        prefill.model_type = ModelType::empty();
        assert!(!servable(&prefill));

        let mut text = card.clone();
        text.model_input = ModelInput::Text;
        assert!(!servable(&text));

        let mut lora = card.clone();
        lora.lora = Some(LoraInfo {
            name: "adapter".into(),
            max_gpu_lora_count: None,
        });
        assert!(!servable(&lora));
    }

    #[test]
    fn port_defaults_to_ephemeral_and_rejects_garbage() {
        let name = env_validation::DYN_WORKER_VALIDATION_LOCAL_PORT;
        temp_env::with_var(name, None::<&str>, || assert_eq!(local_port().unwrap(), 0));
        temp_env::with_var(name, Some(""), || assert_eq!(local_port().unwrap(), 0));
        temp_env::with_var(name, Some("18181"), || {
            assert_eq!(local_port().unwrap(), 18181)
        });
        temp_env::with_var(name, Some("http"), || assert!(local_port().is_err()));
    }

    #[tokio::test]
    async fn registry_engine_is_unavailable_until_registered() {
        let registry = LocalEndpointRegistry::new();
        let engine = RegistryEngine::new(registry.clone(), "generate");
        let request = PreprocessedRequest::builder()
            .model("m".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap();

        let Err(error) = engine.generate(SingleIn::new(request.clone())).await else {
            panic!("no engine is registered");
        };
        assert!(crate::http::service::metrics::request_was_unavailable(
            error.as_ref()
        ));

        // Round trip through the JSON surface every worker registers.
        let echo: ExecutionContext = Arc::new(EchoTokens);
        registry.register("generate".into(), JsonEngine::local(echo));
        let outputs: Vec<_> = engine
            .generate(SingleIn::new(request))
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].data.as_ref().unwrap().token_ids, vec![1, 2, 3]);
    }

    struct EchoTokens;

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for EchoTokens
    {
        async fn generate(
            &self,
            input: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let (request, context) = input.into_parts();
            let output = LLMEngineOutput {
                token_ids: request.token_ids.to_vec(),
                ..LLMEngineOutput::stop()
            };
            Ok(ResponseStream::new(
                Box::pin(futures::stream::iter([Annotated::from_data(output)])),
                context.context(),
            ))
        }
    }
}
