# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.connectors import virtual
from dynamo.planner.monitoring.worker_info import WorkerInfo

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _WorkerInfoProvider:
    def __init__(self) -> None:
        self.infos = {
            SubComponentType.PREFILL: WorkerInfo(
                k8s_name="prefill",
                component_name="prefill",
                endpoint="generate",
            ),
            SubComponentType.DECODE: WorkerInfo(
                k8s_name="decode",
                component_name="backend",
                endpoint="generate",
            ),
        }

    def get_worker_info(
        self, sub_component_type: SubComponentType, backend: str = "vllm"
    ) -> WorkerInfo:
        del backend
        return self.infos[sub_component_type]


class _Client:
    def __init__(self, instance_ids: list[str]) -> None:
        self._instance_ids = instance_ids

    def instance_ids(self) -> list[str]:
        return self._instance_ids


class _Endpoint:
    def __init__(self, client: _Client) -> None:
        self._client = client
        self.client_calls = 0

    async def client(self) -> _Client:
        self.client_calls += 1
        return self._client


class _Runtime:
    def __init__(self, instances_by_endpoint: dict[str, list[str]]) -> None:
        self.endpoints = {
            name: _Endpoint(_Client(instance_ids))
            for name, instance_ids in instances_by_endpoint.items()
        }
        self.endpoint_calls: list[str] = []

    def endpoint(self, name: str) -> _Endpoint:
        self.endpoint_calls.append(name)
        return self.endpoints[name]


def _make_connector(monkeypatch, *, instances_by_endpoint, scaling_ready):
    coordinator = MagicMock()
    coordinator.read_state.return_value = SimpleNamespace(
        num_prefill_workers=-1,
        num_decode_workers=-1,
        decision_id=-1,
    )
    coordinator.is_scaling_ready = AsyncMock(return_value=scaling_ready)
    coordinator.update_scaling_decision = AsyncMock()
    coordinator.wait_for_scaling_completion = AsyncMock()
    coordinator.async_init = AsyncMock()
    monkeypatch.setattr(
        virtual,
        "VirtualConnectorCoordinator",
        MagicMock(return_value=coordinator),
    )
    monkeypatch.setattr(virtual.asyncio, "sleep", AsyncMock())
    # Without bindings, model cards can't be tracked; tests that need them
    # install a fake tracker.
    monkeypatch.setattr(
        virtual.VirtualConnector, "_new_card_tracker", lambda self, name: None
    )
    runtime = _Runtime(instances_by_endpoint)
    connector = virtual.VirtualConnector(
        runtime=runtime,
        dynamo_namespace="test-ns",
        worker_info_provider=_WorkerInfoProvider(),
        model_name="Test-Model",
    )
    return connector, coordinator, runtime


def test_virtual_connector_constructs_without_protocol_parent_init(monkeypatch):
    connector, _, _ = _make_connector(
        monkeypatch,
        instances_by_endpoint={},
        scaling_ready=True,
    )

    assert connector.model_name == "test-model"


@pytest.mark.asyncio
async def test_actual_counts_come_from_runtime_not_desired_state(monkeypatch):
    connector, coordinator, runtime = _make_connector(
        monkeypatch,
        instances_by_endpoint={
            "test-ns.prefill.generate": ["p-1", "p-2"],
            "test-ns.backend.generate": ["d-1", "d-2", "d-3"],
        },
        scaling_ready=False,
    )
    coordinator.read_state.return_value = SimpleNamespace(
        num_prefill_workers=10,
        num_decode_workers=20,
        decision_id=7,
    )

    counts = await connector.get_actual_worker_counts(
        prefill_component_name="prefill",
        decode_component_name="decode",
    )

    assert counts == (2, 3, False)
    assert runtime.endpoint_calls == [
        "test-ns.prefill.generate",
        "test-ns.backend.generate",
    ]
    coordinator.read_state.assert_called_once_with()
    coordinator.is_scaling_ready.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_unset_decision_reports_discovered_counts_and_stable(monkeypatch):
    connector, _, _ = _make_connector(
        monkeypatch,
        instances_by_endpoint={
            "test-ns.prefill.generate": [],
            "test-ns.backend.generate": [],
        },
        scaling_ready=True,
    )

    counts = await connector.get_actual_worker_counts(
        prefill_component_name="prefill",
        decode_component_name="decode",
    )

    assert counts == (0, 0, True)


@pytest.mark.asyncio
async def test_acknowledgement_waits_for_discovery_to_reach_desired_count(monkeypatch):
    connector, coordinator, _ = _make_connector(
        monkeypatch,
        instances_by_endpoint={"test-ns.prefill.generate": ["p-1", "p-2"]},
        scaling_ready=True,
    )
    coordinator.read_state.return_value = SimpleNamespace(
        num_prefill_workers=3,
        num_decode_workers=-1,
        decision_id=4,
    )

    counts = await connector.get_actual_worker_counts(
        prefill_component_name="prefill",
    )

    assert counts == (2, 0, False)


@pytest.mark.asyncio
async def test_new_runtime_client_waits_for_initial_discovery_snapshot(monkeypatch):
    connector, _, runtime = _make_connector(
        monkeypatch,
        instances_by_endpoint={"test-ns.prefill.generate": []},
        scaling_ready=True,
    )
    client = runtime.endpoints["test-ns.prefill.generate"]._client

    async def populate_snapshot(_delay):
        client._instance_ids.append("p-1")

    sleep = AsyncMock(side_effect=populate_snapshot)
    monkeypatch.setattr(virtual.asyncio, "sleep", sleep)

    counts = await connector.get_actual_worker_counts(
        prefill_component_name="prefill",
    )

    assert counts == (1, 0, True)
    sleep.assert_awaited_once_with(0.1)


@pytest.mark.asyncio
async def test_runtime_clients_are_reused(monkeypatch):
    connector, _, runtime = _make_connector(
        monkeypatch,
        instances_by_endpoint={
            "test-ns.prefill.generate": ["p-1"],
            "test-ns.backend.generate": ["d-1"],
        },
        scaling_ready=True,
    )

    for _ in range(2):
        await connector.get_actual_worker_counts(
            prefill_component_name="prefill",
            decode_component_name="decode",
        )

    assert runtime.endpoints["test-ns.prefill.generate"].client_calls == 1
    assert runtime.endpoints["test-ns.backend.generate"].client_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sub_component_type", "instances", "expected_prefill", "expected_decode"),
    [
        (SubComponentType.PREFILL, ["p-1", "p-2"], 3, None),
        (SubComponentType.DECODE, [], None, 1),
    ],
)
async def test_add_component_uses_actual_count_when_decision_is_unset(
    monkeypatch,
    sub_component_type,
    instances,
    expected_prefill,
    expected_decode,
):
    endpoint = (
        "test-ns.prefill.generate"
        if sub_component_type == SubComponentType.PREFILL
        else "test-ns.backend.generate"
    )
    connector, coordinator, _ = _make_connector(
        monkeypatch,
        instances_by_endpoint={endpoint: instances},
        scaling_ready=True,
    )

    await connector.add_component(sub_component_type, blocking=False)

    coordinator.update_scaling_decision.assert_awaited_once_with(
        expected_prefill, expected_decode
    )


@pytest.mark.asyncio
async def test_remove_component_uses_actual_count_when_decision_is_unset(monkeypatch):
    connector, coordinator, _ = _make_connector(
        monkeypatch,
        instances_by_endpoint={"test-ns.backend.generate": ["d-1", "d-2"]},
        scaling_ready=True,
    )

    await connector.remove_component(SubComponentType.DECODE, blocking=False)

    coordinator.update_scaling_decision.assert_awaited_once_with(None, 1)


@pytest.mark.asyncio
async def test_add_component_preserves_existing_desired_count(monkeypatch):
    connector, coordinator, runtime = _make_connector(
        monkeypatch,
        instances_by_endpoint={},
        scaling_ready=True,
    )
    coordinator.read_state.return_value = SimpleNamespace(
        num_prefill_workers=4,
        num_decode_workers=6,
        decision_id=2,
    )

    await connector.add_component(SubComponentType.PREFILL, blocking=False)

    coordinator.update_scaling_decision.assert_awaited_once_with(5, None)
    assert runtime.endpoint_calls == []


class _CardTracker:
    def __init__(self, worker_ids: list[str]) -> None:
        self.worker_ids = worker_ids
        self.shutdown_calls = 0

    def get_model_cards(self) -> dict[str, str]:
        return {worker_id: "{}" for worker_id in self.worker_ids}

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.mark.asyncio
async def test_workers_without_a_model_card_are_not_counted(monkeypatch):
    # A worker held by validation gates has an endpoint but no card yet.
    connector, _, _ = _make_connector(
        monkeypatch,
        instances_by_endpoint={"test-ns.backend.generate": [1, 2, 3]},
        scaling_ready=True,
    )
    tracker = _CardTracker(["1", "3"])
    monkeypatch.setattr(
        virtual.VirtualConnector, "_new_card_tracker", lambda self, name: tracker
    )

    counts = await connector.get_actual_worker_counts(decode_component_name="decode")
    assert counts == (0, 2, True)

    # The card is published on release, and the worker is counted.
    tracker.worker_ids.append("2")
    counts = await connector.get_actual_worker_counts(decode_component_name="decode")
    assert counts == (0, 3, True)
    assert tracker.shutdown_calls == 0


@pytest.mark.asyncio
async def test_untracked_cards_fall_back_to_endpoint_instances(monkeypatch):
    connector, _, _ = _make_connector(
        monkeypatch,
        instances_by_endpoint={"test-ns.backend.generate": [1, 2]},
        scaling_ready=True,
    )

    class _NotStarted(_CardTracker):
        def get_model_cards(self) -> dict[str, str]:
            raise RuntimeError("start_tracking() has not been called")

    monkeypatch.setattr(
        virtual.VirtualConnector,
        "_new_card_tracker",
        lambda self, name: _NotStarted([]),
    )

    counts = await connector.get_actual_worker_counts(decode_component_name="decode")
    assert counts == (0, 2, True)
