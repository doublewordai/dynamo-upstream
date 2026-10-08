# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference checks and check runner for worker validation gates.

A worker configured with ``DYN_WORKER_VALIDATION_GATES`` runs each gate once
before it becomes routable. ``exec`` gates run a command in the worker
container; ``url`` gates POST to a check runner on loopback, usually a sidecar
with its own image. This package provides:

* ``python -m dynamo.validation.completion``: a built-in check that sends one
  chat completion through the worker's loopback OpenAI endpoint.
* ``python -m dynamo.validation.runner``: a stateless HTTP check runner for
  ``url`` gates that runs one configured command per request.
"""
