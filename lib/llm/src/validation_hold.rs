// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Holds model-card discovery writes until worker validation releases.
//!
//! Under etcd discovery the frontend routes to any endpoint that has an
//! admitted model card, whatever the pod's readiness. So a worker held by
//! validation gates (`DYN_WORKER_VALIDATION_HOLD=card`, the default) must not
//! write its card. [`register_model_card`](crate::local_model::register_model_card)
//! returns as usual, and the card waits here instead.
//!
//! While a card is held, this map is its state: taint updates edit it and
//! unregister removes it. On release, each held card is written in
//! registration order, with retry; a final failure fails the worker's
//! `publish` gate and the worker stays held.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::Context;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::{Discovery, DiscoveryInstance, DiscoverySpec};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::validation::HoldMode;
use std::sync::Arc;

/// Attempts at writing one held card on release before failing closed.
const PUBLISH_ATTEMPTS: u32 = 3;
const PUBLISH_RETRY_DELAY: Duration = Duration::from_secs(1);

/// One held card, identified by its discovery instance and path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct HeldKey {
    connection_id: u64,
    namespace: String,
    component: String,
    endpoint: String,
    model_suffix: Option<String>,
}

impl HeldKey {
    fn new(endpoint: &Endpoint, model_suffix: Option<&str>) -> Self {
        let id = endpoint.id();
        Self {
            connection_id: endpoint.drt().connection_id(),
            namespace: id.namespace,
            component: id.component,
            endpoint: id.name,
            model_suffix: model_suffix.map(str::to_string),
        }
    }
}

impl std::fmt::Display for HeldKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{}/{}/{:x}",
            self.namespace, self.component, self.endpoint, self.connection_id
        )?;
        if let Some(suffix) = &self.model_suffix {
            write!(f, "/{suffix}")?;
        }
        Ok(())
    }
}

struct HeldCard {
    discovery: Arc<dyn Discovery>,
    spec: DiscoverySpec,
}

/// Cards held by every runtime in this process. The lock is never held across
/// a discovery write: a release copies the card out, writes it, then checks
/// the map again, so a taint update or unregister that lands mid-write is
/// honoured (see [`publish_held`]).
static HELD: LazyLock<tokio::sync::Mutex<HashMap<HeldKey, HeldCard>>> =
    LazyLock::new(Default::default);

/// A card written by the release in progress, and the discovery it was
/// written to.
type PublishedCard = (Arc<dyn Discovery>, DiscoveryInstance);

/// Cards written by the release in progress. If a later card's write fails,
/// these are unregistered again, so a worker whose `publish` gate failed has
/// no routable card.
static PUBLISHED: LazyLock<tokio::sync::Mutex<Vec<PublishedCard>>> =
    LazyLock::new(Default::default);

/// Hold `spec` if this worker's validation requires it. Returns the spec back
/// when it should be written now: no gates, `hold=health`, or already released.
///
/// Discovery refuses a second, different model on the same endpoint path. The
/// held map applies the same rule, so a worker can't park two models under
/// one key and have one silently dropped.
pub(crate) async fn hold_card(
    endpoint: &Endpoint,
    model_suffix: Option<&str>,
    spec: DiscoverySpec,
) -> anyhow::Result<Option<DiscoverySpec>> {
    let drt = endpoint.drt();
    let Some(validation) = drt.worker_validation() else {
        return Ok(Some(spec));
    };
    if validation.hold_mode() != HoldMode::Card || validation.is_released() {
        return Ok(Some(spec));
    }

    let key = HeldKey::new(endpoint, model_suffix);
    {
        let mut held = HELD.lock().await;
        if let Some(existing) = held.get(&key) {
            let (existing_name, requested_name) =
                (display_name(&existing.spec), display_name(&spec));
            if existing_name != requested_name {
                let id = endpoint.id();
                anyhow::bail!(
                    "Cannot register model '{requested_name}' on endpoint \
                     '{}/{}/{}': a different model '{existing_name}' is already registered there",
                    id.namespace,
                    id.component,
                    id.name
                );
            }
        }
        held.insert(
            key.clone(),
            HeldCard {
                discovery: drt.discovery(),
                spec,
            },
        );
    }

    let hook_key = key.clone();
    let added = validation.add_release_hook(Box::new(move || {
        Box::pin(async move { publish_held(&hook_key).await })
    }));
    if !added {
        // Released between the check above and now: write it directly.
        return Ok(HELD.lock().await.remove(&key).map(|held| held.spec));
    }
    // Whatever fails the release, no card written by it may stay routable.
    validation.add_rollback_hook(Box::new(|| Box::pin(rollback_published())));
    tracing::info!(card = %key, "Holding model card until worker validation passes");
    Ok(None)
}

fn display_name(spec: &DiscoverySpec) -> String {
    match spec {
        DiscoverySpec::Model { card_json, .. } => card_json
            .get("display_name")
            .and_then(|name| name.as_str())
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

/// Write a held card. The card may have been unregistered while held, or
/// already written by an earlier hook for the same key.
///
/// The write happens with the map unlocked. If the card changed meanwhile (a
/// taint update), it's written again; if it was unregistered, the write is
/// undone. On a final failure every card this release has written so far is
/// unregistered, and the error fails the worker's `publish` gate.
async fn publish_held(key: &HeldKey) -> anyhow::Result<()> {
    loop {
        let (discovery, spec) = {
            let held = HELD.lock().await;
            let Some(card) = held.get(key) else {
                return Ok(());
            };
            (Arc::clone(&card.discovery), card.spec.clone())
        };

        // On failure the rollback hook unregisters what earlier hooks wrote.
        let instance = register_with_retry(&discovery, &spec, key).await?;
        PUBLISHED
            .lock()
            .await
            .push((Arc::clone(&discovery), instance.clone()));

        let mut held = HELD.lock().await;
        match held.get(key) {
            Some(card) if card.spec == spec => {
                held.remove(key);
                tracing::info!(card = %key, "Published held model card");
                return Ok(());
            }
            Some(_) => {
                // Edited during the write: write the current card over it.
                tracing::debug!(card = %key, "Held model card changed during its write; rewriting");
            }
            None => {
                // Unregistered during the write: the card must not exist.
                drop(held);
                discovery
                    .unregister(instance)
                    .await
                    .with_context(|| format!("undoing the write of unregistered card {key}"))?;
                return Ok(());
            }
        }
    }
}

async fn register_with_retry(
    discovery: &Arc<dyn Discovery>,
    spec: &DiscoverySpec,
    key: &HeldKey,
) -> anyhow::Result<DiscoveryInstance> {
    let mut attempt = 1;
    loop {
        match discovery.register(spec.clone()).await {
            Ok(instance) => return Ok(instance),
            Err(error) if attempt < PUBLISH_ATTEMPTS => {
                tracing::warn!(card = %key, attempt, %error, "Retrying held model card write");
                attempt += 1;
                tokio::time::sleep(PUBLISH_RETRY_DELAY).await;
            }
            Err(error) => {
                return Err(error).with_context(|| format!("publishing held model card {key}"));
            }
        }
    }
}

/// Unregister every card this release wrote. Best effort: a card that can't
/// be removed is logged, and the worker stays held either way.
async fn rollback_published() {
    let published = std::mem::take(&mut *PUBLISHED.lock().await);
    for (discovery, instance) in published {
        if let Err(error) = discovery.unregister(instance.clone()).await {
            tracing::error!(
                %error,
                instance_id = instance.instance_id(),
                "Could not roll back a model card written by a failed release"
            );
        }
    }
}

/// Apply a taint update to this worker's held base-model card. Returns
/// `Ok(false)` when no card is held, so the caller updates discovery instead.
pub(crate) async fn update_held_taints(
    endpoint: &Endpoint,
    taints: &HashSet<String>,
) -> anyhow::Result<bool> {
    if endpoint.drt().worker_validation().is_none() {
        return Ok(false);
    }
    let key = HeldKey::new(endpoint, None);
    let mut held = HELD.lock().await;
    let Some(card) = held.get_mut(&key) else {
        return Ok(false);
    };
    card.spec = card.spec.clone().with_model_taints(taints.clone())?;
    Ok(true)
}

/// Drop a held card. Returns `true` if one was held, in which case it was
/// never written and there is nothing to unregister from discovery.
pub(crate) async fn forget_held(endpoint: &Endpoint, model_suffix: Option<&str>) -> bool {
    if endpoint.drt().worker_validation().is_none() {
        return false;
    }
    HELD.lock()
        .await
        .remove(&HeldKey::new(endpoint, model_suffix))
        .is_some()
}

#[cfg(all(test, feature = "integration"))]
mod tests {
    use super::*;
    use crate::local_model::{LocalModel, register_model_card, update_model_taints};
    use crate::model_card::ModelDeploymentCard;
    use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery};
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    const GATES_PASS: &str = r#"{"g": {"exec": ["true"], "timeout_secs": 30}}"#;

    /// A runtime whose gates start only when a local engine registers, so each
    /// test decides when release happens.
    async fn gated_drt(gates: &str, hold: Option<&str>) -> DistributedRuntime {
        temp_env::async_with_vars(
            [
                ("DYN_SYSTEM_PORT", Some("0")),
                ("DYN_WORKER_VALIDATION_GATES", Some(gates)),
                ("DYN_WORKER_VALIDATION_HOLD", hold),
            ],
            async {
                let runtime = Runtime::from_current().unwrap();
                DistributedRuntime::new(runtime, DistributedConfig::process_local())
                    .await
                    .unwrap()
            },
        )
        .await
    }

    fn card(name: &str) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only(name);
        card.runtime_config.max_num_seqs = Some(8);
        card
    }

    async fn model_cards(drt: &DistributedRuntime) -> Vec<DiscoveryInstance> {
        drt.discovery()
            .list(DiscoveryQuery::AllModels)
            .await
            .unwrap()
            .into_iter()
            .filter(|instance| instance.instance_id() == drt.connection_id())
            .collect()
    }

    async fn release(drt: &DistributedRuntime) {
        let validation = drt.worker_validation().unwrap();
        validation.start(Default::default(), drt.child_token());
        tokio::time::timeout(Duration::from_secs(10), validation.wait_released())
            .await
            .unwrap_or_else(|_| panic!("not released: {}", validation.status_json()));
    }

    fn taints_of(instance: &DiscoveryInstance) -> Vec<String> {
        let DiscoveryInstance::Model { card_json, .. } = instance else {
            panic!("not a model card")
        };
        serde_json::from_value(card_json["runtime_config"]["taints"].clone()).unwrap_or_default()
    }

    #[tokio::test]
    async fn card_is_held_until_release_then_written() {
        let drt = gated_drt(GATES_PASS, None).await;
        let endpoint = drt
            .namespace("ns")
            .unwrap()
            .component("c")
            .unwrap()
            .endpoint("generate");
        register_model_card(&endpoint, &card("m")).await.unwrap();
        assert!(model_cards(&drt).await.is_empty(), "held card was written");

        release(&drt).await;
        assert_eq!(model_cards(&drt).await.len(), 1);
    }

    #[tokio::test]
    async fn hold_health_writes_the_card_immediately() {
        let drt = gated_drt(GATES_PASS, Some("health")).await;
        let endpoint = drt
            .namespace("ns")
            .unwrap()
            .component("c")
            .unwrap()
            .endpoint("generate");
        register_model_card(&endpoint, &card("m")).await.unwrap();
        assert_eq!(model_cards(&drt).await.len(), 1);
        assert!(!drt.worker_validation().unwrap().is_released());
    }

    #[tokio::test]
    async fn cards_are_written_in_registration_order() {
        let drt = gated_drt(GATES_PASS, None).await;
        let component = drt.namespace("ns").unwrap().component("c").unwrap();
        let base = component.endpoint("generate");
        let mut base_card = card("base");
        base_card.source_path = Some("org/base".into());
        register_model_card(&base, &base_card).await.unwrap();
        let mut lora = card("adapter");
        lora.source_path = Some("org/base".into());
        lora.lora = Some(crate::model_card::LoraInfo {
            name: "adapter".into(),
            max_gpu_lora_count: None,
        });
        register_model_card(&base, &lora).await.unwrap();

        // Observe write order through discovery's watch stream.
        let watch_drt = drt.clone();
        let observed = tokio::spawn(async move {
            use futures::StreamExt;
            let mut stream = watch_drt
                .discovery()
                .list_and_watch(DiscoveryQuery::AllModels, None)
                .await
                .unwrap();
            let mut order = Vec::new();
            while order.len() < 2 {
                if let Some(Ok(dynamo_runtime::discovery::DiscoveryEvent::Added(instance))) =
                    stream.next().await
                {
                    let DiscoveryInstance::Model { model_suffix, .. } = instance else {
                        continue;
                    };
                    order.push(model_suffix);
                }
            }
            order
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        release(&drt).await;
        let order = tokio::time::timeout(Duration::from_secs(5), observed)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(order, vec![None, Some("adapter".to_string())]);
    }

    #[tokio::test]
    async fn taints_set_during_the_hold_are_in_the_written_card() {
        let drt = gated_drt(GATES_PASS, None).await;
        let endpoint = drt
            .namespace("ns")
            .unwrap()
            .component("c")
            .unwrap()
            .endpoint("generate");
        register_model_card(&endpoint, &card("m")).await.unwrap();
        update_model_taints(&endpoint, HashSet::from(["mirror".to_string()]))
            .await
            .unwrap();
        assert!(model_cards(&drt).await.is_empty());

        release(&drt).await;
        let cards = model_cards(&drt).await;
        assert_eq!(cards.len(), 1);
        assert_eq!(taints_of(&cards[0]), vec!["mirror".to_string()]);

        // After release, taint updates go straight to discovery.
        update_model_taints(&endpoint, HashSet::new())
            .await
            .unwrap();
        assert!(taints_of(&model_cards(&drt).await[0]).is_empty());
    }

    #[tokio::test]
    async fn reserved_taints_are_rejected_during_the_hold() {
        let drt = gated_drt(GATES_PASS, None).await;
        let endpoint = drt
            .namespace("ns")
            .unwrap()
            .component("c")
            .unwrap()
            .endpoint("generate");
        register_model_card(&endpoint, &card("m")).await.unwrap();
        let error = update_model_taints(
            &endpoint,
            HashSet::from(["dynamo.topology/zone=west".to_string()]),
        )
        .await
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("reserved prefix"),
            "{error:#}"
        );
    }

    #[tokio::test]
    async fn unregister_during_the_hold_drops_the_card() {
        let drt = gated_drt(GATES_PASS, None).await;
        let endpoint = drt
            .namespace("ns")
            .unwrap()
            .component("c")
            .unwrap()
            .endpoint("generate");
        register_model_card(&endpoint, &card("m")).await.unwrap();
        LocalModel::detach_from_endpoint(&endpoint, None)
            .await
            .unwrap();

        release(&drt).await;
        assert!(
            model_cards(&drt).await.is_empty(),
            "unregistered card was written"
        );
    }

    #[tokio::test]
    async fn after_release_cards_are_written_directly() {
        let drt = gated_drt(GATES_PASS, None).await;
        release(&drt).await;
        let endpoint = drt
            .namespace("ns")
            .unwrap()
            .component("c")
            .unwrap()
            .endpoint("generate");
        register_model_card(&endpoint, &card("m")).await.unwrap();
        assert_eq!(model_cards(&drt).await.len(), 1);
    }

    #[tokio::test]
    async fn a_failing_gate_never_writes_the_card() {
        let drt = gated_drt(r#"{"g": {"exec": ["false"], "timeout_secs": 30}}"#, None).await;
        let endpoint = drt
            .namespace("ns")
            .unwrap()
            .component("c")
            .unwrap()
            .endpoint("generate");
        register_model_card(&endpoint, &card("m")).await.unwrap();
        let validation = drt.worker_validation().unwrap();
        validation.start(Default::default(), drt.child_token());
        for _ in 0..100 {
            if validation.status_json()["gates"]["g"]["state"] == "failed" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(validation.status_json()["gates"]["g"]["state"], "failed");
        assert!(model_cards(&drt).await.is_empty());
    }

    #[tokio::test]
    async fn a_different_model_on_the_same_endpoint_is_rejected_while_held() {
        let drt = gated_drt(GATES_PASS, None).await;
        let endpoint = drt
            .namespace("ns")
            .unwrap()
            .component("c")
            .unwrap()
            .endpoint("generate");
        register_model_card(&endpoint, &card("model-a"))
            .await
            .unwrap();
        let error = register_model_card(&endpoint, &card("model-b"))
            .await
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("a different model 'model-a' is already registered"),
            "{error:#}"
        );
        // Re-registering the same model replaces the held card, as discovery
        // would accept it.
        register_model_card(&endpoint, &card("model-a"))
            .await
            .unwrap();
        release(&drt).await;
        assert_eq!(model_cards(&drt).await.len(), 1);
    }

    #[tokio::test]
    async fn a_failed_release_unregisters_cards_it_already_wrote() {
        let drt = gated_drt(GATES_PASS, None).await;
        let endpoint = drt
            .namespace("ns")
            .unwrap()
            .component("c")
            .unwrap()
            .endpoint("generate");
        register_model_card(&endpoint, &card("m")).await.unwrap();
        // A later release step fails after the card's own hook has written it.
        let validation = drt.worker_validation().unwrap();
        validation.add_release_hook(Box::new(|| {
            Box::pin(async { Err(anyhow::anyhow!("later release step failed")) })
        }));
        validation.start(Default::default(), drt.child_token());
        for _ in 0..100 {
            if validation.status_json()["gates"]["publish"]["state"] == "failed" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(
            validation.status_json()["gates"]["publish"]["state"],
            "failed"
        );
        // Rollback is best effort and runs after the gate is marked failed.
        for _ in 0..100 {
            if model_cards(&drt).await.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            model_cards(&drt).await.is_empty(),
            "a card written by a failed release stayed routable"
        );
        assert!(!validation.is_released());
    }

    #[tokio::test]
    async fn without_gates_cards_are_written_as_today() {
        let drt =
            temp_env::async_with_vars([("DYN_WORKER_VALIDATION_GATES", None::<&str>)], async {
                DistributedRuntime::new(
                    Runtime::from_current().unwrap(),
                    DistributedConfig::process_local(),
                )
                .await
                .unwrap()
            })
            .await;
        assert!(drt.worker_validation().is_none());
        let endpoint = drt
            .namespace("ns")
            .unwrap()
            .component("c")
            .unwrap()
            .endpoint("generate");
        register_model_card(&endpoint, &card("m")).await.unwrap();
        assert_eq!(model_cards(&drt).await.len(), 1);
    }
}
