"""Tests for Phase 5 evaluator module."""

from __future__ import annotations

import json

import pytest

from harness.graph import BehaviorGraph, EdgeType, NodeType
from evaluator.statistics import (bootstrap_ci, ci_for_difference,
                                  paired_ttest, wilcoxon_signed_rank,
                                  detect_rate)
from evaluator.ablations import ablate_graph, DEFAULT_ABLATIONS, AblationConfig
from evaluator.counterfactual import (remove_edge_type_transform,
                                      remove_sends_to_external_transform,
                                      sanitize_tool_output_transform,
                                      counterfactual_replay)
from evaluator.protocol import freeze_experiment, verify_frozen_experiment


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------

def test_bootstrap_ci_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    point, lower, upper = bootstrap_ci(values, n_bootstrap=1000, seed=42)
    assert abs(point - 3.0) < 1e-6
    assert lower < point < upper


def test_bootstrap_ci_single_value():
    point, lower, upper = bootstrap_ci([42.0], n_bootstrap=100, seed=42)
    assert point == 42.0
    assert lower == 42.0
    assert upper == 42.0


def test_ci_for_difference():
    g = [3.0, 4.0, 5.0]
    b = [1.0, 2.0, 3.0]
    point, lower, upper = ci_for_difference(g, b, n_bootstrap=1000, seed=42)
    assert abs(point - 2.0) < 1e-6
    assert lower > 0  # all differences positive


def test_detect_rate():
    assert detect_rate([True, True, False]) == 2 / 3
    assert detect_rate([]) == 0.0


def test_paired_ttest():
    g = [3.1, 4.2, 5.1, 6.2]
    b = [1.0, 2.0, 3.0, 4.0]
    stat, p = paired_ttest(g, b)
    assert stat is not None
    assert p is not None
    assert p < 0.01  # clear difference


def test_wilcoxon():
    g = [3.0, 4.0, 5.0, 6.0]
    b = [1.0, 2.0, 3.0, 4.0]
    stat, p = wilcoxon_signed_rank(g, b)
    assert stat is not None
    assert p is not None


def test_paired_tests_handle_constant_differences():
    assert paired_ttest([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]) == (None, None)
    assert wilcoxon_signed_rank([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Ablation tests
# ---------------------------------------------------------------------------

def test_ablate_graph_removes_edges():
    g = BehaviorGraph(episode_id="ep1")
    n1 = g.node(NodeType.TOOL_CALL, 1, {})
    n2 = g.node(NodeType.TOOL_RESULT, 2, {})
    g.edge(n1, n2, EdgeType.DERIVED_FROM, 2, {})
    g.edge(n1, n2, EdgeType.FOLLOWS, 2, {})

    assert len(g.edges) == 2

    ablated = ablate_graph(g, AblationConfig(
        name="test", description="test",
        remove_edge_types=[EdgeType.DERIVED_FROM]
    ))
    assert len(ablated.edges) == 1
    assert list(ablated.edges.values())[0].edge_type == EdgeType.FOLLOWS


def test_no_temporal_ablation_removes_step_features():
    graph = BehaviorGraph(episode_id="temporal")
    source = graph.node(NodeType.TOOL_CALL, 3, {})
    target = graph.node(NodeType.TOOL_RESULT, 5, {})
    graph.edge(source, target, EdgeType.FOLLOWS, 5)

    ablated = ablate_graph(
        graph, next(a for a in DEFAULT_ABLATIONS if a.name == "no_temporal")
    )
    assert all(node.step == 0 for node in ablated.nodes.values())
    assert all(edge.step == 0 for edge in ablated.edges.values())
    assert all(edge.edge_type != EdgeType.FOLLOWS for edge in ablated.edges.values())


def test_default_ablations():
    assert len(DEFAULT_ABLATIONS) == 6
    names = [a.name for a in DEFAULT_ABLATIONS]
    assert "full_graph" in names
    assert "no_provenance" in names
    assert "no_authorization" in names
    assert "no_temporal" in names
    assert "untyped" in names
    assert "bounded_window_5" in names


# ---------------------------------------------------------------------------
# Counterfactual tests
# ---------------------------------------------------------------------------

def test_counterfactual_transforms():
    g = BehaviorGraph(episode_id="ep1")
    n1 = g.node(NodeType.TOOL_CALL, 1, {})
    n2 = g.node(NodeType.EXTERNAL_EFFECT, 2, {})
    n3 = g.node(NodeType.TOOL_RESULT, 1, {})
    g.edge(n1, n2, EdgeType.SENDS_TO, 2, {})
    g.edge(n1, n3, EdgeType.DERIVED_FROM, 1, {})

    # remove_sends_to_external should remove the edge
    cf = remove_sends_to_external_transform()(g)
    assert len(cf.edges) == 1
    assert all(edge.edge_type != EdgeType.SENDS_TO for edge in cf.edges.values())

    # remove_authorizes should keep the sends_to edge (not an authorizes edge)
    cf2 = remove_edge_type_transform(EdgeType.AUTHORIZES)(g)
    assert len(cf2.edges) == 2

    sanitized = sanitize_tool_output_transform()(g)
    assert all(node.node_type != NodeType.TOOL_RESULT
               for node in sanitized.nodes.values())


def test_counterfactual_replay_runs():
    # Minimal test that counterfactual_replay runs without error
    from monitors import GraphMonitor
    gm = GraphMonitor(name="graph", epochs=3)

    # Build a tiny graph with one node
    trace = []
    # We need at least one trace event for score_prefix to work
    from harness.schema import TraceEvent, EventType
    ev = TraceEvent(episode_id="ep1", step=1, event_type=EventType.AGENT_ACTION,
                    actor="agent", payload={"tool": "file_read", "arguments": {}})
    trace = [ev]

    g = BehaviorGraph(episode_id="ep1")
    g.ingest_event(ev, 1)

    cf_results = counterfactual_replay(
        gm, trace, 1, 1, "ep1", [
            {
                "name": "test_cf",
                "description": "test",
                "transform": remove_edge_type_transform(EdgeType.FOLLOWS),
                "expected_direction": "decrease",
            },
        ],
    )
    assert len(cf_results) == 1
    assert cf_results[0].episode_id == "ep1"
    assert 0.0 <= cf_results[0].original_risk <= 1.0
    assert 0.0 <= cf_results[0].counterfactual_risk <= 1.0


def test_calibration_forwards_variant_graph_getter(monkeypatch):
    from harness.schema import EventType, TraceEvent
    from monitors import GraphMonitor, calibrate_monitor

    monitor = GraphMonitor(name="graph", epochs=1)
    event = TraceEvent(
        episode_id="calibration",
        step=1,
        event_type=EventType.AGENT_ACTION,
        actor="agent",
        payload={"tool": "file_read", "arguments": {}},
    )
    expected_graph = BehaviorGraph(episode_id="calibration")
    seen_graphs = []
    monkeypatch.setattr(
        monitor,
        "score_prefix",
        lambda trace, step, graph=None: (
            seen_graphs.append(graph) or 0.5
        ),
    )

    calibrate_monitor(
        monitor,
        [([event], 1)],
        target_fprs=(0.05,),
        graph_getter=lambda trace, step: expected_graph,
    )
    assert seen_graphs == [expected_graph]


def test_frozen_experiment_manifest_is_immutable(tmp_path):
    protocol = {"fpr": 0.05, "seed": 42, "model": "graph"}
    splits = {
        "train": ["train-1"],
        "validation": ["validation-1"],
        "test": ["test-1"],
    }
    manifest = tmp_path / "experiment.json"
    digest = freeze_experiment(manifest, protocol, splits)
    assert digest
    verify_frozen_experiment(manifest, protocol, splits)

    with pytest.raises(ValueError, match="already exists and differs"):
        freeze_experiment(manifest, {"fpr": 0.1}, splits)


def test_freeze_rejects_overlapping_episode_ids(tmp_path):
    splits = {
        "train": ["same"],
        "validation": ["validation"],
        "test": ["same"],
    }
    with pytest.raises(ValueError, match="disjoint"):
        freeze_experiment(tmp_path / "experiment.json", {}, splits)


def test_frozen_experiment_detects_manifest_tampering(tmp_path):
    protocol = {"fpr": 0.05}
    splits = {
        "train": ["train"],
        "validation": ["validation"],
        "test": ["test"],
    }
    manifest = tmp_path / "experiment.json"
    freeze_experiment(manifest, protocol, splits)
    document = json.loads(manifest.read_text())
    document["protocol"]["fpr"] = 0.1
    manifest.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="differs from frozen manifest"):
        verify_frozen_experiment(manifest, protocol, splits)
