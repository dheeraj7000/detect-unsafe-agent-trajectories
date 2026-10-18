"""Unit tests for the harness schema and telemetry."""

from __future__ import annotations

import os
import tempfile

from harness.schema import (EdgeType, EpisodeManifest, EventType, GraphEdge,
                            GraphNode, NodeType, TraceEvent)
from harness.telemetry import Telemetry


def test_graph_node_creation():
    n = GraphNode(
        node_type=NodeType.TOOL_CALL,
        step=1,
        attributes={"tool": "file_read"},
        episode_id="ep1",
    )
    assert n.node_type == NodeType.TOOL_CALL
    assert n.step == 1
    assert n.episode_id == "ep1"
    assert n.node_id


def test_graph_edge_creation():
    e = GraphEdge(
        source_id="n1",
        target_id="n2",
        edge_type=EdgeType.INVOKES,
        step=2,
        episode_id="ep1",
    )
    assert e.edge_type == EdgeType.INVOKES
    assert e.step == 2


def test_trace_event_serialization():
    ev = TraceEvent(
        episode_id="ep1",
        step=3,
        event_type=EventType.TOOL_CALL,
        actor="agent",
        payload={"tool": "file_write"},
    )
    json_str = ev.model_dump_json()
    ev2 = TraceEvent.model_validate_json(json_str)
    assert ev2.event_type == EventType.TOOL_CALL
    assert ev2.step == 3


def test_telemetry_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tel = Telemetry(base_dir=tmp)
        tel.begin_episode("ep_x")
        tel.next_step()
        tel.record(
            event_type="agent_action",
            actor="agent",
            payload={"tool": "noop"},
        )
        tel.end_episode({
            "episode_id": "ep_x",
            "task_success": True,
            "policy_violation": False,
            "unsafe_event_step": None,
            "total_steps": 1,
            "intervention_step": None,
            "alerts": [],
        })

        events = tel.load_trace("ep_x")
        assert len(events) >= 2  # episode_start + agent_action (+ episode_end)
        assert events[0].event_type == EventType.EPISODE_START


def test_manifest_defaults():
    m = EpisodeManifest(
        episode_id="ep_m",
        task_family="demo",
        scenario_name="x",
        seed=7,
    )
    assert m.max_steps == 500
    assert m.unsafe_event_step is None
    assert m.policy_violation is False
