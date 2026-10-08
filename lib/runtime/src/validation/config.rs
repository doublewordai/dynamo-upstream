// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsing for `DYN_WORKER_VALIDATION_GATES` and `DYN_WORKER_VALIDATION_HOLD`.
//!
//! Every misconfiguration is an error: a worker that cannot tell what it is
//! supposed to check must not start, rather than start unchecked.

use std::collections::BTreeMap;
use std::net::IpAddr;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::config::environment_names::runtime::validation as env_validation;

/// Longest accepted gate name. Gate names become metric label values and
/// `/health` keys, so keep them short and predictable.
const MAX_GATE_NAME_LEN: usize = 63;

/// What a held worker keeps back until every gate passes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HoldMode {
    /// Hold the model card's discovery write and `/health`. No card means no
    /// routing under any router.
    #[default]
    Card,
    /// Publish the card immediately and hold only `/health`. The deployment's
    /// own routing must keep client traffic off the worker.
    Health,
}

impl HoldMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            HoldMode::Card => "card",
            HoldMode::Health => "health",
        }
    }
}

impl std::str::FromStr for HoldMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "card" => Ok(HoldMode::Card),
            "health" => Ok(HoldMode::Health),
            other => bail!(
                "{} must be \"card\" or \"health\", got {other:?}",
                env_validation::DYN_WORKER_VALIDATION_HOLD
            ),
        }
    }
}

/// How one gate is checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateCheck {
    /// Run this command in the worker container. Exit status 0 is a pass.
    Exec(Vec<String>),
    /// POST to a check runner on loopback, which answers
    /// `{"pass": true}` or `{"pass": false, "reason": "..."}`.
    Url(reqwest::Url),
}

/// One configured gate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateSpec {
    pub name: String,
    pub check: GateCheck,
    /// The gate fails if it has no verdict by then.
    pub timeout: Duration,
}

/// Parsed validation configuration. Only exists when gates are configured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationConfig {
    /// Gates in name order.
    pub gates: Vec<GateSpec>,
    pub hold: HoldMode,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGate {
    exec: Option<Vec<String>>,
    url: Option<String>,
    timeout_secs: Option<u64>,
}

impl ValidationConfig {
    /// Read the configuration from the environment.
    ///
    /// Returns `Ok(None)` when no gates are configured, which keeps today's
    /// behaviour unchanged.
    pub fn from_env() -> Result<Option<Self>> {
        let gates = std::env::var(env_validation::DYN_WORKER_VALIDATION_GATES).ok();
        let hold = std::env::var(env_validation::DYN_WORKER_VALIDATION_HOLD).ok();
        Self::parse(gates.as_deref(), hold.as_deref())
    }

    pub fn parse(gates: Option<&str>, hold: Option<&str>) -> Result<Option<Self>> {
        let gates = gates.map(str::trim).filter(|value| !value.is_empty());
        let hold = hold.map(str::trim).filter(|value| !value.is_empty());

        let Some(gates) = gates else {
            if let Some(hold) = hold {
                bail!(
                    "{}={hold:?} is set but {} is not; set both or neither",
                    env_validation::DYN_WORKER_VALIDATION_HOLD,
                    env_validation::DYN_WORKER_VALIDATION_GATES
                );
            }
            return Ok(None);
        };

        let hold = hold.map(str::parse).transpose()?.unwrap_or_default();

        let raw: BTreeMap<String, RawGate> = serde_json::from_str(gates).with_context(|| {
            format!(
                "{} must be a JSON object of gate name to \
                 {{\"exec\": [...]}} or {{\"url\": \"...\"}} with \"timeout_secs\"",
                env_validation::DYN_WORKER_VALIDATION_GATES
            )
        })?;
        if raw.is_empty() {
            bail!(
                "{} must configure at least one gate",
                env_validation::DYN_WORKER_VALIDATION_GATES
            );
        }

        let gates = raw
            .into_iter()
            .map(|(name, gate)| {
                parse_gate(&name, gate).with_context(|| format!("validation gate {name:?}"))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Some(ValidationConfig { gates, hold }))
    }
}

fn parse_gate(name: &str, gate: RawGate) -> Result<GateSpec> {
    validate_gate_name(name)?;

    let check = match (gate.exec, gate.url) {
        (Some(_), Some(_)) => bail!("set exactly one of \"exec\" and \"url\", not both"),
        (None, None) => bail!("set one of \"exec\" or \"url\""),
        (Some(argv), None) => {
            if argv.first().is_none_or(|program| program.trim().is_empty()) {
                bail!("\"exec\" must be a non-empty command, e.g. [\"python\", \"check.py\"]");
            }
            GateCheck::Exec(argv)
        }
        (None, Some(url)) => GateCheck::Url(parse_loopback_url(&url)?),
    };

    let timeout_secs = gate
        .timeout_secs
        .context("\"timeout_secs\" is required: a gate without a deadline could hold forever")?;
    if timeout_secs == 0 {
        bail!("\"timeout_secs\" must be greater than zero");
    }

    Ok(GateSpec {
        name: name.to_string(),
        check,
        timeout: Duration::from_secs(timeout_secs),
    })
}

/// Lowercase alphanumerics, `-` and `_`, starting and ending alphanumeric.
fn validate_gate_name(name: &str) -> Result<()> {
    let bytes = name.as_bytes();
    let valid_char =
        |c: &u8| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'-' || *c == b'_';
    let alnum = |c: &u8| c.is_ascii_lowercase() || c.is_ascii_digit();
    let ok = !bytes.is_empty()
        && bytes.len() <= MAX_GATE_NAME_LEN
        && bytes.iter().all(valid_char)
        && bytes.first().is_some_and(alnum)
        && bytes.last().is_some_and(alnum);
    if !ok {
        bail!(
            "gate names must be 1-{MAX_GATE_NAME_LEN} characters of a-z, 0-9, '-' or '_', \
             starting and ending with a letter or digit"
        );
    }
    Ok(())
}

/// `url` gates may only reach a check runner in the same pod.
fn parse_loopback_url(url: &str) -> Result<reqwest::Url> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("invalid \"url\" {url:?}"))?;
    if parsed.scheme() != "http" {
        bail!("\"url\" must use http on loopback, got {url:?}");
    }
    let loopback = match parsed.host() {
        Some(url::Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(ip)) => IpAddr::V4(ip).is_loopback(),
        Some(url::Host::Ipv6(ip)) => IpAddr::V6(ip).is_loopback(),
        None => false,
    };
    if !loopback {
        bail!("\"url\" must point at loopback (127.0.0.1, ::1 or localhost), got {url:?}");
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(gates: Option<&str>, hold: Option<&str>) -> Result<Option<ValidationConfig>> {
        ValidationConfig::parse(gates, hold)
    }

    fn parse_err(gates: Option<&str>, hold: Option<&str>) -> String {
        format!("{:#}", parse(gates, hold).unwrap_err())
    }

    #[test]
    fn unset_means_no_validation() {
        assert_eq!(parse(None, None).unwrap(), None);
        assert_eq!(parse(Some("  "), Some("")).unwrap(), None);
    }

    #[test]
    fn parses_exec_and_url_gates_in_name_order() {
        let config = parse(
            Some(
                r#"{
                    "smoke": {"exec": ["python", "/checks/smoke.py"], "timeout_secs": 900},
                    "latency": {"url": "http://127.0.0.1:9100/validate", "timeout_secs": 600}
                }"#,
            ),
            None,
        )
        .unwrap()
        .unwrap();

        assert_eq!(config.hold, HoldMode::Card);
        assert_eq!(config.gates.len(), 2);
        assert_eq!(config.gates[0].name, "latency");
        assert_eq!(
            config.gates[0].check,
            GateCheck::Url("http://127.0.0.1:9100/validate".parse().unwrap())
        );
        assert_eq!(config.gates[0].timeout, Duration::from_secs(600));
        assert_eq!(config.gates[1].name, "smoke");
        assert_eq!(
            config.gates[1].check,
            GateCheck::Exec(vec!["python".into(), "/checks/smoke.py".into()])
        );
    }

    #[test]
    fn parses_hold_mode() {
        let gates = Some(r#"{"a": {"exec": ["true"], "timeout_secs": 1}}"#);
        assert_eq!(
            parse(gates, Some("card")).unwrap().unwrap().hold,
            HoldMode::Card
        );
        assert_eq!(
            parse(gates, Some("health")).unwrap().unwrap().hold,
            HoldMode::Health
        );
    }

    #[test]
    fn accepts_every_loopback_spelling() {
        for url in [
            "http://127.0.0.1:9100/v",
            "http://127.0.0.2:9100/v",
            "http://[::1]:9100/v",
            "http://localhost:9100/v",
            "http://LOCALHOST:9100/v",
        ] {
            let gates = format!(r#"{{"a": {{"url": "{url}", "timeout_secs": 1}}}}"#);
            parse(Some(&gates), None).unwrap_or_else(|err| panic!("{url}: {err:#}"));
        }
    }

    #[test]
    fn rejects_every_misconfiguration() {
        let cases: &[(Option<&str>, Option<&str>, &str)] = &[
            (None, Some("card"), "set both or neither"),
            (Some("not json"), None, "must be a JSON object"),
            (Some("[]"), None, "must be a JSON object"),
            (Some("{}"), None, "at least one gate"),
            (
                Some(r#"{"a": {"exec": ["true"], "timeout_secs": 1}}"#),
                Some("both"),
                "must be \"card\" or \"health\"",
            ),
            (
                Some(r#"{"Bad": {"exec": ["true"], "timeout_secs": 1}}"#),
                None,
                "gate names must be",
            ),
            (
                Some(r#"{"-a": {"exec": ["true"], "timeout_secs": 1}}"#),
                None,
                "gate names must be",
            ),
            (
                Some(r#"{"a.b": {"exec": ["true"], "timeout_secs": 1}}"#),
                None,
                "gate names must be",
            ),
            (Some(r#"{"a": {"timeout_secs": 1}}"#), None, "set one of"),
            (
                Some(r#"{"a": {"exec": ["true"], "url": "http://127.0.0.1/", "timeout_secs": 1}}"#),
                None,
                "not both",
            ),
            (
                Some(r#"{"a": {"exec": [], "timeout_secs": 1}}"#),
                None,
                "non-empty command",
            ),
            (
                Some(r#"{"a": {"exec": [""], "timeout_secs": 1}}"#),
                None,
                "non-empty command",
            ),
            (
                Some(r#"{"a": {"exec": ["true"]}}"#),
                None,
                "\"timeout_secs\" is required",
            ),
            (
                Some(r#"{"a": {"exec": ["true"], "timeout_secs": 0}}"#),
                None,
                "greater than zero",
            ),
            (
                Some(r#"{"a": {"exec": ["true"], "timeout_secs": 1, "retries": 3}}"#),
                None,
                "unknown field",
            ),
            (
                Some(r#"{"a": {"url": "http://10.0.0.1:9100/", "timeout_secs": 1}}"#),
                None,
                "must point at loopback",
            ),
            (
                Some(r#"{"a": {"url": "http://checks.svc:9100/", "timeout_secs": 1}}"#),
                None,
                "must point at loopback",
            ),
            (
                Some(r#"{"a": {"url": "https://127.0.0.1:9100/", "timeout_secs": 1}}"#),
                None,
                "must use http",
            ),
            (
                Some(r#"{"a": {"url": "not a url", "timeout_secs": 1}}"#),
                None,
                "invalid \"url\"",
            ),
        ];
        for (gates, hold, expected) in cases {
            let err = parse_err(*gates, *hold);
            assert!(
                err.contains(expected),
                "gates={gates:?} hold={hold:?}: expected {expected:?} in {err:?}"
            );
        }
    }
}
