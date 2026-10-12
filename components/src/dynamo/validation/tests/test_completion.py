# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dynamo.validation import completion

pytestmark = [pytest.mark.gpu_0, pytest.mark.pre_merge, pytest.mark.unit]


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    """urllib honours an ambient proxy even for 127.0.0.1; keep the stub local."""
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)


def _serve(status: int, payload: dict):
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1", seen


def _choice(message: dict) -> dict:
    return {"choices": [{"index": 0, "message": message}]}


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "ready"},
        {"role": "assistant", "content": None, "reasoning_content": "thinking"},
        {"role": "assistant", "content": None, "reasoning": "thinking"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1", "type": "function"}],
        },
    ],
)
def test_pass(message):
    server, url, seen = _serve(200, _choice(message))
    try:
        assert completion.check_completion(url, "m") is None
        assert seen["path"] == "/v1/chat/completions"
        assert seen["body"]["model"] == "m"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (200, _choice({"role": "assistant", "content": ""}), "empty message"),
        (200, {"choices": []}, "no choices"),
        (200, [], "not an object"),
        (503, {"error": "engine not registered"}, "HTTP 503"),
    ],
)
def test_fail(status, payload, expected):
    server, url, _ = _serve(status, payload)
    try:
        reason = completion.check_completion(url, "m")
        assert reason is not None and expected in reason, reason
    finally:
        server.shutdown()
        server.server_close()


def test_unreachable():
    reason = completion.check_completion("http://127.0.0.1:1/v1", "m", timeout=2)
    assert reason is not None and "failed" in reason


def test_main_requires_the_gate_environment(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("DYN_VALIDATION_MODEL", raising=False)
    assert completion.main([]) == 1
    assert "OPENAI_BASE_URL" in capsys.readouterr().err
