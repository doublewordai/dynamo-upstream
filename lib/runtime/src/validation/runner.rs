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

/// Most of a check runner's response body that is read. A verdict is a few
/// bytes; anything near this is a broken runner, not a reason worth keeping.
const URL_RESPONSE_MAX_BYTES: usize = 64 * 1024;

/// Backoff between `url` attempts while the check runner is unreachable.
const URL_RETRY_INITIAL: Duration = Duration::from_millis(500);
const URL_RETRY_MAX: Duration = Duration::from_secs(10);

/// HTTP client for `url` gates. No redirects and no proxy, so a request to
/// loopback can't be steered off the pod and a verdict can't come from
/// anywhere else.
pub(super) fn url_client() -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .expect("reqwest client with static options")
}

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

/// Run the command; exit status 0 is a pass. The check runs in its own process
/// group, and the whole group is killed if the deadline drops this future, so
/// nothing the check started outlives its verdict.
async fn run_exec(gate: &str, argv: &[String], env: &GateEnv) -> Result<(), String> {
    let mut command = tokio::process::Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .envs(env.vars(gate))
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .spawn()
        .map_err(|error| format!("could not run {:?}: {error}", argv[0]))?;
    // Declared after `child`, so it drops after `kill_on_drop` has signalled
    // the check itself and then sweeps up anything the check spawned.
    let mut group = ProcessGroupGuard::new(child.id());

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
    // The check exited on its own. Its group id could be reused by an
    // unrelated process, so don't kill it on drop; descendants it left
    // running are its own doing.
    group.disarm();
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

/// Kills a check's process group when dropped while armed. The check is its
/// own group leader (`process_group(0)`), so its pid is the group id.
struct ProcessGroupGuard {
    pgid: Option<u32>,
}

impl ProcessGroupGuard {
    fn new(pid: Option<u32>) -> Self {
        Self { pgid: pid }
    }

    fn disarm(&mut self) {
        self.pgid = None;
    }
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(pgid) = self.pgid.take() {
            // SAFETY: killpg has no memory-safety preconditions; a stale or
            // unknown group id returns ESRCH, which is ignored.
            unsafe {
                libc::killpg(pgid as libc::pid_t, libc::SIGKILL);
            }
        }
    }
}

#[derive(Deserialize)]
struct UrlVerdict {
    pass: bool,
    #[serde(default)]
    reason: Option<String>,
}

/// Read at most [`URL_RESPONSE_MAX_BYTES`] of a response body.
async fn read_capped(mut response: reqwest::Response) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| format!("reading check runner response: {error}"))?
    {
        body.extend_from_slice(&chunk);
        if body.len() > URL_RESPONSE_MAX_BYTES {
            return Err(format!(
                "check runner response exceeded {URL_RESPONSE_MAX_BYTES} bytes"
            ));
        }
    }
    Ok(body)
}

/// A reason or error text from the runner, cut to the same size as an `exec`
/// gate's stderr tail.
fn truncate_reason(text: &str) -> String {
    let text = text.trim();
    if text.len() <= STDERR_TAIL_BYTES {
        return text.to_string();
    }
    let mut cut = STDERR_TAIL_BYTES;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &text[..cut])
}

/// Whether a non-2xx status may clear up by itself. Server errors and the
/// transient client statuses are retried; any other 4xx is a misconfiguration
/// that won't change, so it fails the gate at once.
fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || matches!(
            status,
            reqwest::StatusCode::REQUEST_TIMEOUT
                | reqwest::StatusCode::TOO_EARLY
                | reqwest::StatusCode::TOO_MANY_REQUESTS
        )
}

/// POST the gate to its check runner. Connection failures, server errors and
/// transient client statuses are retried until the deadline, which covers a
/// runner that is not up yet or restarts mid-check. A well-formed verdict and
/// any other 4xx are final.
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
                let body = read_capped(response).await?;
                let verdict: UrlVerdict = serde_json::from_slice(&body).map_err(|error| {
                    format!("check runner returned an invalid verdict (want {{\"pass\": bool}}): {error}")
                })?;
                return if verdict.pass {
                    Ok(())
                } else {
                    Err(verdict
                        .reason
                        .as_deref()
                        .map(truncate_reason)
                        .unwrap_or_else(|| "check runner reported a failure".to_string()))
                };
            }
            Ok(response) => {
                let status = response.status();
                let text = match read_capped(response).await {
                    Ok(body) => truncate_reason(&String::from_utf8_lossy(&body)),
                    Err(error) => error,
                };
                let error = format!("check runner returned {status}: {text}");
                if !is_retryable_status(status) {
                    return Err(error);
                }
                error
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
        run_gate(gate, &GateEnv::default(), &url_client()).await
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
    async fn exec_hang_fails_at_deadline_and_kills_the_whole_process_group() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("survived");
        // A backgrounded grandchild that would outlive the shell if only the
        // shell were killed.
        let script = format!("(sleep 1; touch {}) & sleep 30", marker.display());
        let started = Instant::now();
        let err = run(&exec_gate(
            &["sh", "-c", &script],
            Duration::from_millis(300),
        ))
        .await
        .unwrap_err();
        assert!(err.contains("no verdict within"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(2));
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(!marker.exists(), "a child of the hung check survived");
    }

    #[tokio::test]
    async fn url_client_error_is_final_and_server_error_is_retried() {
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
                        calls.fetch_add(1, Ordering::SeqCst);
                        (StatusCode::NOT_FOUND, "no such check")
                    }
                }
            }),
        );
        serve(listener, app).await;
        let started = Instant::now();
        let err = run(&url_gate(addr, Duration::from_secs(10)))
            .await
            .unwrap_err();
        assert!(err.contains("404"), "{err}");
        assert!(err.contains("no such check"), "{err}");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "a 404 was retried");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn url_redirects_are_not_followed() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/validate",
            post(|| async {
                (
                    StatusCode::TEMPORARY_REDIRECT,
                    [("location", "http://127.0.0.1:1/elsewhere")],
                )
            }),
        );
        serve(listener, app).await;
        let err = run(&url_gate(addr, Duration::from_secs(10)))
            .await
            .unwrap_err();
        assert!(err.contains("307"), "{err}");
    }

    #[tokio::test]
    async fn url_oversized_response_fails_closed() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/validate",
            post(|| async { "x".repeat(URL_RESPONSE_MAX_BYTES + 1) }),
        );
        serve(listener, app).await;
        let err = run(&url_gate(addr, Duration::from_secs(10)))
            .await
            .unwrap_err();
        assert!(err.contains("exceeded"), "{err}");

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/validate",
            post(|| async {
                Json(serde_json::json!({"pass": false, "reason": "r".repeat(10_000)}))
            }),
        );
        serve(listener, app).await;
        let err = run(&url_gate(addr, Duration::from_secs(10)))
            .await
            .unwrap_err();
        assert!(err.len() < STDERR_TAIL_BYTES + 8, "{}", err.len());
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
