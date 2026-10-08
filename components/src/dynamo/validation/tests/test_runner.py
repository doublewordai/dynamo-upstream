# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import sys

import pytest
from aiohttp.test_utils import TestClient, TestServer

from dynamo.validation import runner

pytestmark = [pytest.mark.gpu_0, pytest.mark.pre_merge, pytest.mark.unit]


def test_parse_check_commands_and_builtins():
    assert runner.parse_check("smoke=python3 /checks/smoke.py --x 'a b'") == (
        "smoke",
        ["python3", "/checks/smoke.py", "--x", "a b"],
    )
    name, argv = runner.parse_check("c=builtin:completion")
    assert name == "c"
    assert argv[-2:] == ["-m", "dynamo.validation.completion"]


@pytest.mark.parametrize("spec", ["smoke", "=true", "smoke=", "c=builtin:nope"])
def test_parse_check_rejects_bad_specs(spec):
    with pytest.raises(argparse.ArgumentTypeError):
        runner.parse_check(spec)


@pytest.mark.asyncio
async def test_run_check_pass_fail_and_env():
    assert await runner.run_check(["true"], {}, None) is None
    reason = await runner.run_check(
        ["sh", "-c", "echo parser mismatch >&2; exit 2"], {}, None
    )
    assert reason == "exited with status 2: parser mismatch"
    assert (
        await runner.run_check(
            ["sh", "-c", 'test "$OPENAI_BASE_URL" = http://x/v1'],
            {"OPENAI_BASE_URL": "http://x/v1"},
            None,
        )
        is None
    )


@pytest.mark.asyncio
async def test_run_check_missing_program():
    reason = await runner.run_check(["/nonexistent/check"], {}, None)
    assert reason.startswith("could not run")


@pytest.mark.asyncio
async def test_run_check_timeout_kills_the_process_group(tmp_path):
    marker = tmp_path / "survived"
    reason = await runner.run_check(
        ["sh", "-c", f"(sleep 1; touch {marker}) & sleep 30"], {}, 0.3
    )
    assert reason == "no verdict within 0.3s"
    await asyncio.sleep(1.5)
    assert not marker.exists(), "a child of the timed-out check survived"


@pytest.mark.asyncio
async def test_http_verdicts():
    app = runner.build_app(
        {
            "ok": ["true"],
            "bad": ["sh", "-c", "echo nope >&2; exit 1"],
            "env": ["sh", "-c", 'test "$DYN_VALIDATION_MODEL" = m'],
        },
        None,
    )
    async with TestClient(TestServer(app)) as client:
        await _check_verdicts(client)


async def _check_verdicts(client):
    response = await client.post("/validate/ok", json={})
    assert response.status == 200
    assert await response.json() == {"pass": True}

    response = await client.post("/validate/bad", json={})
    assert await response.json() == {
        "pass": False,
        "reason": "exited with status 1: nope",
    }

    response = await client.post("/validate/env", json={"DYN_VALIDATION_MODEL": "m"})
    assert await response.json() == {"pass": True}

    # Unknown checks are a definite failure, not a retryable error.
    response = await client.post("/validate/missing", json={})
    assert response.status == 200
    assert (await response.json())["pass"] is False


def test_builtin_completion_uses_this_interpreter():
    assert runner.BUILTINS["completion"][0] == sys.executable
