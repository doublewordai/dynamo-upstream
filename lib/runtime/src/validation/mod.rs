// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker validation gates.
//!
//! A worker configured with `DYN_WORKER_VALIDATION_GATES` runs each gate once,
//! as soon as its engine is serving, and stays held until every gate passes:
//! `/health` reports not ready, and with [`HoldMode::Card`] the model card's
//! discovery write is deferred (see `dynamo_llm::validation_hold`). `/live` is
//! never affected, so a held worker is not restarted.
//!
//! Verdicts live only in this process's memory. A restarted worker container
//! starts held and re-runs every gate. A failed gate keeps the worker held for
//! the life of the process; deleting the pod is the retry.

pub mod config;
mod runner;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::future::BoxFuture;
use serde_json::json;
use tokio::sync::watch;

pub use config::{GateCheck, GateSpec, HoldMode, ValidationConfig};

use crate::metrics::MetricsHierarchy;
use crate::metrics::prometheus_names::worker_validation as metric_names;

/// Name of the synthetic gate that reports a failed release hook, such as a
/// held model card that could not be published.
pub const PUBLISH_GATE: &str = "publish";

/// Work that must finish before a worker whose gates have passed reports
/// ready, for example writing held model cards. An error fails closed: the
/// worker stays held and the error is reported as the [`PUBLISH_GATE`] gate.
pub type ReleaseHook = Box<dyn FnOnce() -> BoxFuture<'static, anyhow::Result<()>> + Send>;

/// Environment passed to every gate.
#[derive(Clone, Debug, Default)]
pub struct GateEnv {
    /// Loopback OpenAI-compatible endpoint serving this worker's model.
    pub openai_base_url: Option<String>,
    /// Model name to request from that endpoint.
    pub model: Option<String>,
}

impl GateEnv {
    pub(crate) fn vars(&self, gate: &str) -> Vec<(&'static str, String)> {
        let mut vars = vec![("DYN_VALIDATION_GATE", gate.to_string())];
        if let Some(url) = &self.openai_base_url {
            vars.push(("OPENAI_BASE_URL", url.clone()));
        }
        if let Some(model) = &self.model {
            vars.push(("DYN_VALIDATION_MODEL", model.clone()));
        }
        vars
    }
}

/// One gate's verdict so far.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateState {
    Pending,
    Running,
    Passed,
    Failed { reason: String },
}

impl GateState {
    fn as_str(&self) -> &'static str {
        match self {
            GateState::Pending => "pending",
            GateState::Running => "running",
            GateState::Passed => "passed",
            GateState::Failed { .. } => "failed",
        }
    }

    /// Gauge value: 0 while undecided, 1 passed, -1 failed.
    fn gauge_value(&self) -> i64 {
        match self {
            GateState::Pending | GateState::Running => 0,
            GateState::Passed => 1,
            GateState::Failed { .. } => -1,
        }
    }
}

#[derive(Debug)]
struct GateStatus {
    state: GateState,
    duration: Option<Duration>,
}

struct Metrics {
    state: prometheus::IntGaugeVec,
    duration: prometheus::HistogramVec,
}

/// Validation state for one worker process, owned by the `DistributedRuntime`.
pub struct WorkerValidation {
    config: ValidationConfig,
    gates: parking_lot::Mutex<BTreeMap<String, GateStatus>>,
    release_hooks: parking_lot::Mutex<Vec<ReleaseHook>>,
    released: watch::Sender<bool>,
    started: AtomicBool,
    metrics: std::sync::OnceLock<Metrics>,
}

impl std::fmt::Debug for WorkerValidation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerValidation")
            .field("config", &self.config)
            .field("released", &self.is_released())
            .finish_non_exhaustive()
    }
}

impl WorkerValidation {
    pub fn new(config: ValidationConfig) -> Arc<Self> {
        let gates = config
            .gates
            .iter()
            .map(|gate| {
                (
                    gate.name.clone(),
                    GateStatus {
                        state: GateState::Pending,
                        duration: None,
                    },
                )
            })
            .collect();
        Arc::new(Self {
            config,
            gates: parking_lot::Mutex::new(gates),
            release_hooks: parking_lot::Mutex::new(Vec::new()),
            released: watch::channel(false).0,
            started: AtomicBool::new(false),
            metrics: std::sync::OnceLock::new(),
        })
    }

    pub fn config(&self) -> &ValidationConfig {
        &self.config
    }

    pub fn hold_mode(&self) -> HoldMode {
        self.config.hold
    }

    /// True once every gate has passed and every release hook has finished.
    /// Latches: a released worker never becomes held again.
    pub fn is_released(&self) -> bool {
        *self.released.borrow()
    }

    /// Resolves when the worker is released. Never resolves if a gate fails.
    pub async fn wait_released(&self) {
        let mut rx = self.released.subscribe();
        let _ = rx.wait_for(|released| *released).await;
    }

    /// Run `hook` after every gate passes and before the worker reports ready.
    /// Hooks run in registration order. If the worker is already released the
    /// hook is not run and this returns `false`: the caller should do the work
    /// itself.
    pub fn add_release_hook(&self, hook: ReleaseHook) -> bool {
        // Checked under the hooks lock so a hook cannot be added after the
        // runner has taken the list but before `released` flips.
        let mut hooks = self.release_hooks.lock();
        if self.is_released() {
            return false;
        }
        hooks.push(hook);
        true
    }

    /// Gate state for the `/health` body.
    pub fn status_json(&self) -> serde_json::Value {
        let gates = self.gates.lock();
        let gates: serde_json::Map<String, serde_json::Value> = gates
            .iter()
            .map(|(name, status)| {
                let mut entry = json!({ "state": status.state.as_str() });
                if let GateState::Failed { reason } = &status.state {
                    entry["reason"] = json!(reason);
                }
                if let Some(duration) = status.duration {
                    entry["duration_secs"] = json!(duration.as_secs_f64());
                }
                (name.clone(), entry)
            })
            .collect();
        json!({
            "released": self.is_released(),
            "hold": self.config.hold.as_str(),
            "gates": gates,
        })
    }

    /// Register `dynamo_component_worker_validation_gate_*` metrics.
    pub fn register_metrics<T: MetricsHierarchy>(&self, registry: &T) -> anyhow::Result<()> {
        let state = registry.metrics().create_intgaugevec(
            metric_names::GATE_STATE,
            "Worker validation gate verdict: 0 undecided, 1 passed, -1 failed",
            &[metric_names::GATE_LABEL],
            &[],
        )?;
        let duration = registry.metrics().create_histogramvec(
            metric_names::GATE_DURATION_SECONDS,
            "Time a worker validation gate took to reach its verdict",
            &[metric_names::GATE_LABEL],
            &[],
            Some(vec![
                1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0,
            ]),
        )?;
        for (name, status) in self.gates.lock().iter() {
            state
                .with_label_values(&[name])
                .set(status.state.gauge_value());
        }
        let _ = self.metrics.set(Metrics { state, duration });
        Ok(())
    }

    /// Start every gate, once. Later calls do nothing.
    ///
    /// Gates run in parallel on the current Tokio runtime. When the last gate
    /// passes, release hooks run and then the worker is released. `cancel`
    /// stops the gates on shutdown, failing them.
    pub fn start(self: &Arc<Self>, env: GateEnv, cancel: tokio_util::sync::CancellationToken) {
        if self.started.swap(true, Ordering::SeqCst) {
            return;
        }
        tracing::info!(
            gates = ?self.config.gates.iter().map(|gate| &gate.name).collect::<Vec<_>>(),
            hold = self.config.hold.as_str(),
            "Starting worker validation gates"
        );
        let this = Arc::clone(self);
        tokio::spawn(async move { this.run(env, cancel).await });
    }

    async fn run(self: Arc<Self>, env: GateEnv, cancel: tokio_util::sync::CancellationToken) {
        let client = reqwest::Client::new();
        let runs = self.config.gates.iter().map(|gate| {
            let this = Arc::clone(&self);
            let env = env.clone();
            let client = client.clone();
            let cancel = cancel.clone();
            async move {
                this.set_state(&gate.name, GateState::Running, None);
                let started = tokio::time::Instant::now();
                let verdict = tokio::select! {
                    verdict = runner::run_gate(gate, &env, &client) => verdict,
                    _ = cancel.cancelled() => Err("worker is shutting down".to_string()),
                };
                let elapsed = started.elapsed();
                match verdict {
                    Ok(()) => {
                        tracing::info!(gate = %gate.name, ?elapsed, "Validation gate passed");
                        this.set_state(&gate.name, GateState::Passed, Some(elapsed));
                        true
                    }
                    Err(reason) => {
                        tracing::error!(gate = %gate.name, ?elapsed, %reason, "Validation gate failed; worker stays held");
                        this.set_state(&gate.name, GateState::Failed { reason }, Some(elapsed));
                        false
                    }
                }
            }
        });
        let all_passed = futures::future::join_all(runs)
            .await
            .into_iter()
            .all(|passed| passed);
        if !all_passed {
            return;
        }

        // Hooks can be added while earlier ones run, so drain until none are
        // left. The release flips under the hooks lock, after which
        // `add_release_hook` refuses new hooks.
        loop {
            let hooks = {
                let mut hooks = self.release_hooks.lock();
                if hooks.is_empty() {
                    self.released.send_replace(true);
                    break;
                }
                std::mem::take(&mut *hooks)
            };
            for hook in hooks {
                if let Err(error) = hook().await {
                    let reason = format!("{error:#}");
                    tracing::error!(%reason, "Validation release failed; worker stays held");
                    self.set_state(PUBLISH_GATE, GateState::Failed { reason }, None);
                    return;
                }
            }
        }
        tracing::info!("Worker validation passed; worker released");
    }

    fn set_state(&self, gate: &str, state: GateState, duration: Option<Duration>) {
        if let Some(metrics) = self.metrics.get() {
            metrics
                .state
                .with_label_values(&[gate])
                .set(state.gauge_value());
            if let Some(duration) = duration {
                metrics
                    .duration
                    .with_label_values(&[gate])
                    .observe(duration.as_secs_f64());
            }
        }
        self.gates
            .lock()
            .insert(gate.to_string(), GateStatus { state, duration });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::sync::CancellationToken;

    fn validation(gates: &str) -> Arc<WorkerValidation> {
        WorkerValidation::new(ValidationConfig::parse(Some(gates), None).unwrap().unwrap())
    }

    async fn settle(validation: &WorkerValidation) {
        for _ in 0..200 {
            let settled = validation
                .gates
                .lock()
                .values()
                .all(|s| !matches!(s.state, GateState::Pending | GateState::Running));
            if settled {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("gates did not settle: {:?}", validation.status_json());
    }

    #[tokio::test]
    async fn releases_when_every_gate_passes() {
        let validation = validation(
            r#"{"a": {"exec": ["true"], "timeout_secs": 5}, "b": {"exec": ["sh", "-c", "exit 0"], "timeout_secs": 5}}"#,
        );
        assert!(!validation.is_released());
        validation.start(GateEnv::default(), CancellationToken::new());
        tokio::time::timeout(Duration::from_secs(5), validation.wait_released())
            .await
            .unwrap();
        let status = validation.status_json();
        assert_eq!(status["released"], true);
        assert_eq!(status["gates"]["a"]["state"], "passed");
        assert_eq!(status["gates"]["b"]["state"], "passed");
    }

    #[tokio::test]
    async fn one_failing_gate_holds_the_worker() {
        let validation = validation(
            r#"{"good": {"exec": ["true"], "timeout_secs": 5}, "bad": {"exec": ["sh", "-c", "echo broken parser >&2; exit 1"], "timeout_secs": 5}}"#,
        );
        validation.start(GateEnv::default(), CancellationToken::new());
        settle(&validation).await;
        assert!(!validation.is_released());
        let status = validation.status_json();
        assert_eq!(status["gates"]["good"]["state"], "passed");
        assert_eq!(status["gates"]["bad"]["state"], "failed");
        assert!(
            status["gates"]["bad"]["reason"]
                .as_str()
                .unwrap()
                .contains("broken parser"),
            "{status}"
        );
    }

    #[tokio::test]
    async fn start_runs_gates_only_once() {
        let dir = tempfile::tempdir().unwrap();
        let counter = dir.path().join("runs");
        let gates = format!(
            r#"{{"count": {{"exec": ["sh", "-c", "echo x >> {}"], "timeout_secs": 5}}}}"#,
            counter.display()
        );
        let validation = validation(&gates);
        validation.start(GateEnv::default(), CancellationToken::new());
        validation.start(GateEnv::default(), CancellationToken::new());
        validation.wait_released().await;
        validation.start(GateEnv::default(), CancellationToken::new());
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(std::fs::read_to_string(&counter).unwrap(), "x\n");
    }

    #[tokio::test]
    async fn release_hooks_run_in_order_before_release() {
        let validation = validation(r#"{"a": {"exec": ["true"], "timeout_secs": 5}}"#);
        let order = Arc::new(parking_lot::Mutex::new(Vec::new()));
        for i in 0..3 {
            let order = Arc::clone(&order);
            let observed = Arc::clone(&validation);
            assert!(validation.add_release_hook(Box::new(move || {
                Box::pin(async move {
                    assert!(!observed.is_released(), "hook ran after release");
                    order.lock().push(i);
                    Ok(())
                })
            })));
        }
        validation.start(GateEnv::default(), CancellationToken::new());
        validation.wait_released().await;
        assert_eq!(*order.lock(), vec![0, 1, 2]);
        // Released workers refuse new hooks so callers do the work directly.
        assert!(!validation.add_release_hook(Box::new(|| Box::pin(async { Ok(()) }))));
    }

    #[tokio::test]
    async fn a_hook_added_while_hooks_run_still_runs_before_release() {
        let validation = validation(r#"{"a": {"exec": ["true"], "timeout_secs": 5}}"#);
        let late_ran = Arc::new(AtomicBool::new(false));
        let outer = Arc::clone(&validation);
        let flag = Arc::clone(&late_ran);
        validation.add_release_hook(Box::new(move || {
            Box::pin(async move {
                let observed = Arc::clone(&outer);
                assert!(outer.add_release_hook(Box::new(move || {
                    Box::pin(async move {
                        assert!(!observed.is_released());
                        flag.store(true, Ordering::SeqCst);
                        Ok(())
                    })
                })));
                Ok(())
            })
        }));
        validation.start(GateEnv::default(), CancellationToken::new());
        validation.wait_released().await;
        assert!(late_ran.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn failed_release_hook_fails_closed() {
        let validation = validation(r#"{"a": {"exec": ["true"], "timeout_secs": 5}}"#);
        validation.add_release_hook(Box::new(|| {
            Box::pin(async { Err(anyhow::anyhow!("discovery write failed")) })
        }));
        validation.start(GateEnv::default(), CancellationToken::new());
        for _ in 0..200 {
            if validation.status_json()["gates"][PUBLISH_GATE]["state"] == "failed" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let status = validation.status_json();
        assert_eq!(status["gates"][PUBLISH_GATE]["state"], "failed", "{status}");
        assert!(!validation.is_released());
    }

    #[tokio::test]
    async fn shutdown_fails_running_gates() {
        let validation = validation(r#"{"slow": {"exec": ["sleep", "30"], "timeout_secs": 60}}"#);
        let cancel = CancellationToken::new();
        validation.start(GateEnv::default(), cancel.clone());
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();
        settle(&validation).await;
        assert_eq!(validation.status_json()["gates"]["slow"]["state"], "failed");
        assert!(!validation.is_released());
    }

    #[tokio::test]
    async fn gates_see_the_validation_environment() {
        let validation = validation(
            r#"{"env": {"exec": ["sh", "-c", "test \"$DYN_VALIDATION_GATE\" = env && test \"$OPENAI_BASE_URL\" = http://127.0.0.1:1/v1 && test \"$DYN_VALIDATION_MODEL\" = m"], "timeout_secs": 5}}"#,
        );
        validation.start(
            GateEnv {
                openai_base_url: Some("http://127.0.0.1:1/v1".into()),
                model: Some("m".into()),
            },
            CancellationToken::new(),
        );
        tokio::time::timeout(Duration::from_secs(5), validation.wait_released())
            .await
            .unwrap_or_else(|_| panic!("{}", validation.status_json()));
    }
}
