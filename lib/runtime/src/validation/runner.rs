// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runs one gate to a verdict. `Err` carries the reason shown on `/health`.

use std::process::Stdio;
use std::time::Duration;

use serde::Deserialize;
use tokio::io::AsyncReadExt;
use tokio::time::Instant;

use super::{GateCheck, GateEnv, GateSpec};

/// How much of a failed `exec` gate's stderr is kept as its reason.
const STDERR_TAIL_BYTES: usize = 2048;

/// Backoff between `url` attempts while the check runner is unreachable.
const URL_RETRY_INITIAL: Duration = Duration::from_millis(500);
const URL_RETRY_MAX: Duration = Duration::from_secs(10);

pub(super) async fn run_gate(
    gate: &GateSpec,
    env: &GateEnv,
    client: &reqwest::Client,
) -> Result<(), String> {
    let deadline = Instant::now() + gate.timeout;
    let run = async {
        match &gate.check {
            GateCheck::Exec(argv) => run_exec(&gate.name, argv, env).await,
            GateCheck::Url(url) => run_url(&gate.name, url, env, client, deadline).await,
        }
    };
    match tokio::time::timeout_at(deadline, run).await {
        Ok(verdict) => verdict,
        Err(_) => Err(format!(
            "no verdict within timeout_secs={}",
            gate.timeout.as_secs()
        )),
    }
}

/// Run the command; exit status 0 is a pass. The child is killed if the
/// deadline drops this future.
async fn run_exec(gate: &str, argv: &[String], env: &GateEnv) -> Result<(), String> {
    let mut child = tokio::process::Command::new(&argv[0])
        .args(&argv[1..])
        .envs(env.vars(gate))
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| format!("could not run {:?}: {error}", argv[0]))?;

    let mut stderr = child.stderr.take().expect("stderr is piped");
    let read_tail = async {
        let mut tail: Vec<u8> = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            match stderr.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    // Forward the check's stderr to the worker log as it runs.
                    tracing::info!(gate, "{}", String::from_utf8_lossy(&buf[..n]).trim_end());
                    tail.extend_from_slice(&buf[..n]);
                    if tail.len() > STDERR_TAIL_BYTES {
                        tail.drain(..tail.len() - STDERR_TAIL_BYTES);
                    }
                }
            }
        }
        tail
    };
    let (status, tail) = tokio::join!(child.wait(), read_tail);
    let status = status.map_err(|error| format!("waiting for {:?}: {error}", argv[0]))?;
    if status.success() {
        return Ok(());
    }
    let tail = String::from_utf8_lossy(&tail).trim().to_string();
    let exit = match status.code() {
        Some(code) => format!("exited with status {code}"),
        None => "was killed by a signal".to_string(),
    };
    if tail.is_empty() {
        Err(exit)
    } else {
        Err(format!("{exit}: {tail}"))
    }
}

#[derive(Deserialize)]
struct UrlVerdict {
    pass: bool,
    #[serde(default)]
    reason: Option<String>,
}

/// POST the gate to its check runner. Connection failures and error statuses
/// are retried until the deadline, which covers a runner that is not up yet or
/// restarts mid-check. A well-formed verdict is final.
async fn run_url(
    gate: &str,
    url: &reqwest::Url,
    env: &GateEnv,
    client: &reqwest::Client,
    deadline: Instant,
) -> Result<(), String> {
    let body: serde_json::Map<String, serde_json::Value> = env
        .vars(gate)
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.into()))
        .collect();
    let mut backoff = URL_RETRY_INITIAL;
    let mut last_error;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let attempt = client
            .post(url.clone())
            .json(&body)
            .timeout(remaining)
            .send()
            .await;
        last_error = match attempt {
            Ok(response) if response.status().is_success() => {
                let verdict: UrlVerdict = response.json().await.map_err(|error| {
                    format!("check runner returned an invalid verdict (want {{\"pass\": bool}}): {error}")
                })?;
                return if verdict.pass {
                    Ok(())
                } else {
                    Err(verdict
                        .reason
                        .unwrap_or_else(|| "check runner reported a failure".to_string()))
                };
            }
            Ok(response) => {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                format!("check runner returned {status}: {}", text.trim())
            }
            Err(error) => format!("check runner unreachable: {error}"),
        };
        tracing::debug!(gate, error = %last_error, "Retrying validation gate");
        if Instant::now() + backoff >= deadline {
            tokio::time::sleep_until(deadline).await;
            return Err(last_error);
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(URL_RETRY_MAX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, http::StatusCode, routing::post};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn exec_gate(argv: &[&str], timeout: Duration) -> GateSpec {
        GateSpec {
            name: "g".into(),
            check: GateCheck::Exec(argv.iter().map(|s| s.to_string()).collect()),
            timeout,
        }
    }

    fn url_gate(addr: std::net::SocketAddr, timeout: Duration) -> GateSpec {
        GateSpec {
            name: "g".into(),
            check: GateCheck::Url(format!("http://{addr}/validate").parse().unwrap()),
            timeout,
        }
    }

    async fn run(gate: &GateSpec) -> Result<(), String> {
        run_gate(gate, &GateEnv::default(), &reqwest::Client::new()).await
    }

    async fn serve(listener: tokio::net::TcpListener, app: Router) {
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    }

    #[tokio::test]
    async fn exec_pass_and_fail() {
        assert_eq!(
            run(&exec_gate(&["true"], Duration::from_secs(5))).await,
            Ok(())
        );
        let err = run(&exec_gate(
            &[
                "sh",
                "-c",
                "echo first >&2; echo tool parser mismatch >&2; exit 3",
            ],
            Duration::from_secs(5),
        ))
        .await
        .unwrap_err();
        assert!(err.starts_with("exited with status 3"), "{err}");
        assert!(err.contains("tool parser mismatch"), "{err}");
    }

    #[tokio::test]
    async fn exec_reason_keeps_only_the_stderr_tail() {
        let err = run(&exec_gate(
            &[
                "sh",
                "-c",
                "head -c 100000 /dev/zero | tr '\\0' a >&2; echo END >&2; exit 1",
            ],
            Duration::from_secs(5),
        ))
        .await
        .unwrap_err();
        assert!(err.len() < STDERR_TAIL_BYTES + 64, "{}", err.len());
        assert!(err.ends_with("END"), "{err}");
    }

    #[tokio::test]
    async fn exec_missing_program_fails() {
        let err = run(&exec_gate(&["/nonexistent/check"], Duration::from_secs(5)))
            .await
            .unwrap_err();
        assert!(err.contains("could not run"), "{err}");
    }

    #[tokio::test]
    async fn exec_hang_fails_at_deadline_and_kills_the_child() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("survived");
        let script = format!("sleep 2; touch {}", marker.display());
        let started = Instant::now();
        let err = run(&exec_gate(
            &["sh", "-c", &script],
            Duration::from_millis(300),
        ))
        .await
        .unwrap_err();
        assert!(err.contains("no verdict within"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(2));
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert!(!marker.exists(), "hung check was not killed");
    }

    #[tokio::test]
    async fn url_pass_and_fail_carry_the_runner_verdict() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/validate",
            post(|Json(body): Json<serde_json::Value>| async move {
                assert_eq!(body["DYN_VALIDATION_GATE"], "g");
                Json(serde_json::json!({"pass": false, "reason": "p99 TTFT 4.1s > 2s"}))
            }),
        );
        serve(listener, app).await;
        let err = run(&url_gate(addr, Duration::from_secs(5)))
            .await
            .unwrap_err();
        assert_eq!(err, "p99 TTFT 4.1s > 2s");

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/validate",
            post(|| async { Json(serde_json::json!({"pass": true})) }),
        );
        serve(listener, app).await;
        assert_eq!(run(&url_gate(addr, Duration::from_secs(5))).await, Ok(()));
    }

    #[tokio::test]
    async fn url_is_retried_until_the_runner_is_up() {
        // Reserve a port, close it, and only start the runner after the gate
        // has seen connection refused.
        let addr = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap()
        };
        let gate = url_gate(addr, Duration::from_secs(10));
        let check = tokio::spawn(async move { run(&gate).await });
        tokio::time::sleep(Duration::from_millis(1200)).await;
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        let app = Router::new().route(
            "/validate",
            post(|| async { Json(serde_json::json!({"pass": true})) }),
        );
        serve(listener, app).await;
        assert_eq!(check.await.unwrap(), Ok(()));
    }

    #[tokio::test]
    async fn url_error_status_and_dropped_connection_are_retried() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let app = Router::new().route(
            "/validate",
            post({
                let calls = Arc::clone(&calls);
                move || {
                    let calls = Arc::clone(&calls);
                    async move {
                        match calls.fetch_add(1, Ordering::SeqCst) {
                            0 => Err(StatusCode::INTERNAL_SERVER_ERROR),
                            1 => Err(StatusCode::SERVICE_UNAVAILABLE),
                            _ => Ok(Json(serde_json::json!({"pass": true}))),
                        }
                    }
                }
            }),
        );
        serve(listener, app).await;
        assert_eq!(run(&url_gate(addr, Duration::from_secs(10))).await, Ok(()));
        assert_eq!(calls.load(Ordering::SeqCst), 3);

        // A runner that accepts and then drops the connection mid-request.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let dropped = Arc::new(AtomicUsize::new(0));
        tokio::spawn({
            let dropped = Arc::clone(&dropped);
            async move {
                let (socket, _) = listener.accept().await.unwrap();
                dropped.fetch_add(1, Ordering::SeqCst);
                drop(socket);
                let app = Router::new().route(
                    "/validate",
                    post(|| async { Json(serde_json::json!({"pass": true})) }),
                );
                axum::serve(listener, app).await.unwrap();
            }
        });
        assert_eq!(run(&url_gate(addr, Duration::from_secs(10))).await, Ok(()));
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn url_never_up_fails_at_deadline() {
        let addr = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap()
        };
        let started = Instant::now();
        let err = run(&url_gate(addr, Duration::from_millis(1500)))
            .await
            .unwrap_err();
        assert!(
            err.contains("unreachable") || err.contains("no verdict"),
            "{err}"
        );
        assert!(started.elapsed() >= Duration::from_millis(1400));
    }

    #[tokio::test]
    async fn url_malformed_verdict_fails_closed() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route("/validate", post(|| async { "ok" }));
        serve(listener, app).await;
        let err = run(&url_gate(addr, Duration::from_secs(5)))
            .await
            .unwrap_err();
        assert!(err.contains("invalid verdict"), "{err}");
    }
}
