---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Validation Gates
subtitle: Run your own checks against a new worker before it takes client traffic, so a broken worker stalls a rollout instead of serving.
---

A worker can pass its probes and still be broken in ways only a real request shows: a bad tool-call or
reasoning parser, a chat-template regression, or the wrong weight revision. Validation gates let a
worker run checks against itself before it becomes routable. Each check is a plain command or script,
run once, and its exit code is the verdict.

Until every gate passes, the worker gets no client requests and its pod isn't Ready. Liveness is
unaffected, so the worker isn't restarted. During a rolling update, the new pod never becomes Ready, so
the rollout stalls with the old pods serving. There's no operator or CRD change, and nothing changes
when gates aren't configured.

## Configure gates

Set two environment variables on the worker container:

```yaml
env:
- name: DYN_WORKER_VALIDATION_GATES
  value: |
    {
      "completion": {"exec": ["python3", "-m", "dynamo.validation.completion"], "timeout_secs": 600},
      "smoke": {"url": "http://127.0.0.1:9100/validate/smoke", "timeout_secs": 900}
    }
- name: DYN_WORKER_VALIDATION_HOLD   # optional: "card" (default) or "health"
  value: card
```

`DYN_WORKER_VALIDATION_GATES` is a JSON object of gate name to one gate. Gate names are 1-63 characters
of `a-z`, `0-9`, `-` and `_`, starting and ending with a letter or digit. Every gate needs
`timeout_secs`; missing it is a failure.

| Gate | Runs | Verdict |
|---|---|---|
| `exec` | The command in the worker container | Exit status 0 passes. The last 2 KiB of stderr is the failure reason. |
| `url` | A `POST` to a check runner on loopback (`127.0.0.1`, `::1` or `localhost`, http only) | `{"pass": true}` passes; `{"pass": false, "reason": "..."}` fails. |

A `url` gate retries a connection failure or a non-2xx response with backoff until its deadline, which
covers a sidecar that isn't up yet or restarts. A check runner reports a failed check as
`200 {"pass": false, ...}`, not as an error status.

### What a check sees

Gates start once the worker's engine is serving, in parallel, once each. Every gate gets these
variables (an `exec` gate as environment variables, a `url` gate as the JSON request body):

| Variable | Value |
|---|---|
| `OPENAI_BASE_URL` | A loopback-only OpenAI-compatible endpoint on the worker, for example `http://127.0.0.1:PORT/v1`. It runs the same preprocessor, chat template, tool-call and reasoning parsers as a frontend with default settings. |
| `DYN_VALIDATION_MODEL` | The model name to request |
| `DYN_VALIDATION_GATE` | The gate's name |

The endpoint listens on `127.0.0.1` only, on an ephemeral port unless
`DYN_WORKER_VALIDATION_LOCAL_PORT` sets one. It serves the worker's base model (not LoRA adapters),
returns 503 until the engine is registered, and closes once the worker is released. After a failed
gate it stays up, so you can debug the held worker from inside the pod.

Any OpenAI client works as a check:

```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_BASE_URL
reply = client.chat.completions.create(model=os.environ["DYN_VALIDATION_MODEL"], messages=[...])
```

## Hold modes

`DYN_WORKER_VALIDATION_HOLD` describes the deployment's routing, so it's separate from the checks.

| `hold` | Model card written | `/health` | Client traffic kept off the worker by |
|---|---|---|---|
| `card` (default) | When all gates pass | held | The runtime: no card means no routing, under any router |
| `health` | Immediately | held | The deployment's own routing; the runtime doesn't check it |

Under etcd discovery, the frontend routes to any worker with a registered model card, whatever the pod's
readiness, so `card` holds the card itself. Taint updates and LoRA loads and unloads during the hold are
applied to the held card, and the card is written with them on release. `health` is for deployments
that keep the worker off client traffic some other way while still letting the frontend see it.

## Verdicts and retries

- Verdicts live only in the worker process's memory. A restarted worker container starts held and
  re-runs every gate.
- A failed gate keeps the worker held for the life of the process. Deleting the pod is the retry.
- Once every gate passes, the worker stays released. Checking after that is the
  [canary health check](../../reference/observability/health-checks.mdx)'s
  job.
- A gate that crashes, hangs past its deadline, or is misconfigured fails closed: the worker stays held.
- Startup fails, with a clear error, on invalid JSON, a malformed gate name, a gate with both or neither
  of `exec` and `url`, a missing or zero `timeout_secs`, a non-loopback `url`, a `HOLD` set without gates
  or to an unknown value, or a disabled system status server (`DYN_SYSTEM_PORT`).

## Watch progress

`/health` returns 503 while the worker is held, and its body reports each gate:

```json
{
  "status": "notready",
  "validation": {
    "released": false,
    "hold": "card",
    "gates": {
      "completion": {"state": "passed", "duration_secs": 4.2},
      "smoke": {"state": "failed", "reason": "exited with status 1: tool call parser returned no calls", "duration_secs": 31.0}
    }
  }
}
```

States are `pending`, `running`, `passed` and `failed`. A failed release (a held model card that
couldn't be written) shows as a `publish` gate. `/live` never includes validation.

Two metrics are exported on the system port:

- `dynamo_component_worker_validation_gate_state{gate}`: 0 undecided, 1 passed, -1 failed
- `dynamo_component_worker_validation_gate_duration_seconds{gate}`: time to the verdict

## Run checks in a sidecar

Use a `url` gate when a check needs its own image or dependencies. The reference runner is a small
stateless HTTP server that runs one configured command per request:

```yaml
- name: check-runner
  image: my-checks:latest
  command: ["python3", "-m", "dynamo.validation.runner", "--port", "9100",
            "--check", "smoke=python3 /checks/smoke.py",
            "--check", "completion=builtin:completion"]
```

Gate `smoke` then points at `http://127.0.0.1:9100/validate/smoke`. The runner keeps no state; the
worker owns every verdict.

Follow these rules for any check runner, because a sidecar's readiness counts toward the pod's:

- Give it no readiness, liveness or startup probe.
- Never exit on a failed check; answer `{"pass": false}` instead.
- Bind loopback only.

## Roll out with gates

Gates hold the new pod, and the rollout waits on it. Configure the worker component so a held pod never
reduces capacity:

```yaml
podTemplate:
  metadata:
    annotations:
      nvidia.com/deployment-rolling-update-max-surge: "1"
      nvidia.com/deployment-rolling-update-max-unavailable: "0"
```

There's no rollout deadline, so a failing gate stalls the rollout by design. Fix the check or the
worker, then delete the held pod or roll back. A complete example is
[`examples/backends/vllm/deploy/agg_validation_gates.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/agg_validation_gates.yaml).

## Scope and known bypasses

Validation gates cover aggregated workers and decode workers with a chat or completions model, with
etcd or pod-mode Kubernetes discovery. Requests to a decode worker's loopback endpoint don't go through
a prefill worker. They aren't supported for:

- prefill, encode, embeddings-only and text-input workers: they have no chat or completions surface,
  so no loopback endpoint opens and their gates never start. A gated worker of this kind stays held.
- multinode deployments, whose followers have no probes;
- container-mode Kubernetes discovery;
- Dynamo running as a sidecar (`SystemProbePolicy::RuntimeOnly`), which is rejected at startup because
  the engine can restart while the process and its verdicts live on.

Known bypasses and caveats while a worker is held:

- An explicit `worker_id` pin through the standalone router reaches the worker.
- Under `hostNetwork`, the loopback endpoint is reachable from anything on the node while gates run.
- Ready isn't the same as admitted by every frontend: a frontend can see a released card slightly
  after the pod turns Ready.

Set the variables on workers only. A frontend with gates set never registers an engine, so it would
stay held forever.
