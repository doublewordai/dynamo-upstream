# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless check runner for ``url`` validation gates.

Run it as a sidecar container with its own image. Each ``POST /validate/<name>``
runs the command configured for ``<name>`` once, with the gate's environment
(``OPENAI_BASE_URL``, ``DYN_VALIDATION_MODEL``, ``DYN_VALIDATION_GATE``) taken
from the request body, and answers ``{"pass": true}`` or
``{"pass": false, "reason": "..."}``.

    python3 -m dynamo.validation.runner --port 9100 \\
        --check smoke='python3 /checks/smoke.py' --check completion=builtin:completion

The runner keeps no state between requests: the worker owns every verdict, and
a restarted worker simply asks again. It binds loopback only, has no probes and
never exits on a failed check, so it can't make a released pod NotReady.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shlex
import signal
import sys
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)

STDERR_TAIL_BYTES = 2048
BUILTIN_PREFIX = "builtin:"
BUILTINS = {"completion": [sys.executable, "-m", "dynamo.validation.completion"]}
# Only these request-body keys are passed to the check.
GATE_ENV_KEYS = ("OPENAI_BASE_URL", "DYN_VALIDATION_MODEL", "DYN_VALIDATION_GATE")


def parse_check(spec: str) -> tuple[str, list[str]]:
    name, sep, command = spec.partition("=")
    if not sep or not name or not command.strip():
        raise argparse.ArgumentTypeError(
            f"--check must be NAME=COMMAND or NAME={BUILTIN_PREFIX}<check>, got {spec!r}"
        )
    if command.startswith(BUILTIN_PREFIX):
        builtin = command[len(BUILTIN_PREFIX) :]
        if builtin not in BUILTINS:
            raise argparse.ArgumentTypeError(
                f"unknown built-in check {builtin!r}; known: {', '.join(BUILTINS)}"
            )
        return name, list(BUILTINS[builtin])
    return name, shlex.split(command)


async def run_check(
    argv: list[str], env: dict[str, str], timeout: Optional[float]
) -> Optional[str]:
    """Run one check. Returns None on a pass, or the failure reason."""
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=None,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **env},
            start_new_session=True,
        )
    except OSError as error:
        return f"could not run {argv[0]!r}: {error}"

    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except asyncio.TimeoutError:
        await _kill_group(process)
        return f"no verdict within {timeout}s"
    except asyncio.CancelledError:
        # Shutting down: no check outlives the runner.
        await _kill_group(process)
        raise

    if process.returncode == 0:
        return None
    tail = stderr[-STDERR_TAIL_BYTES:].decode(errors="replace").strip()
    status = (
        f"exited with status {process.returncode}"
        if process.returncode > 0
        else f"was killed by signal {-process.returncode}"
    )
    return f"{status}: {tail}" if tail else status


async def _kill_group(process: asyncio.subprocess.Process) -> None:
    """Kill the check and anything it started."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


def build_app(
    checks: dict[str, list[str]], timeout: Optional[float]
) -> web.Application:
    async def validate(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        argv = checks.get(name)
        if argv is None:
            # A permanent misconfiguration: answer with a verdict so the gate
            # fails now instead of retrying until its deadline.
            return web.json_response(
                {"pass": False, "reason": f"no check named {name!r} on this runner"}
            )
        try:
            body = await request.json() if request.can_read_body else {}
        except ValueError:
            body = {}
        env = {key: str(body[key]) for key in GATE_ENV_KEYS if key in body}
        logger.info("Running check %s: %s", name, shlex.join(argv))
        reason = await run_check(argv, env, timeout)
        if reason is None:
            logger.info("Check %s passed", name)
            return web.json_response({"pass": True})
        logger.warning("Check %s failed: %s", name, reason)
        return web.json_response({"pass": False, "reason": reason})

    app = web.Application()
    app.router.add_post("/validate/{name}", validate)
    return app


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument(
        "--check",
        dest="checks",
        action="append",
        type=parse_check,
        required=True,
        help=f"NAME=COMMAND or NAME={BUILTIN_PREFIX}completion; repeatable",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Kill a check after this many seconds. The gate's own timeout_secs "
        "applies regardless.",
    )
    args = parser.parse_args(argv)
    checks = dict(args.checks)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info(
        "Check runner serving %s on %s:%d", ", ".join(checks), args.host, args.port
    )
    web.run_app(
        build_app(checks, args.timeout), host=args.host, port=args.port, print=None
    )


if __name__ == "__main__":
    main()
