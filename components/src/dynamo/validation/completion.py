# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in ``completion`` check: one chat completion must return content.

Reads ``OPENAI_BASE_URL`` and ``DYN_VALIDATION_MODEL``, which the worker sets
for every gate. Exits 0 on a pass. On a failure it exits 1 and writes the
reason to stderr, which the worker reports on ``/health``.

    DYN_WORKER_VALIDATION_GATES='{"completion": {"exec":
        ["python3", "-m", "dynamo.validation.completion"], "timeout_secs": 300}}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_PROMPT = "Reply with the single word: ready"


def check_completion(
    base_url: str,
    model: str,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 32,
    timeout: float = 120.0,
) -> Optional[str]:
    """Return None on a pass, or the failure reason."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        return f"chat completion returned HTTP {error.code}: {detail}"
    except (urllib.error.URLError, OSError, ValueError) as error:
        return f"chat completion failed: {error}"

    choices = payload.get("choices") or []
    if not choices:
        return f"chat completion returned no choices: {json.dumps(payload)[:500]}"
    message = choices[0].get("message") or {}
    content = message.get("content") or message.get("reasoning_content")
    if not content and not message.get("tool_calls"):
        return f"chat completion returned an empty message: {json.dumps(message)[:500]}"
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("DYN_VALIDATION_MODEL"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    if not args.base_url or not args.model:
        print(
            "OPENAI_BASE_URL and DYN_VALIDATION_MODEL must be set "
            "(the worker sets both for every gate)",
            file=sys.stderr,
        )
        return 1
    reason = check_completion(
        args.base_url, args.model, args.prompt, args.max_tokens, args.timeout
    )
    if reason is not None:
        print(reason, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
