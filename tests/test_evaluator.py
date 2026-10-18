"""Tests for Phase 5 evaluator module."""

from __future__ import annotations

from harness.graph import BehaviorGraph, EdgeType, NodeType
from evaluator.statistics import (bootstrap_ci, ci_for_difference,
                                  paired_ttest, wilcoxon_signed_rank,
                                  detect_rate)
from evaluator.ablations import ablate_graph, DEFAULT_ABLATIONS, AblationConfig
from evaluator.counterfactual import (remove_edge_type_transform,
                                      remove_sends_to_external_transform,
                                      sanitize_tool_output_transform,
                                      counterfactual_replay)


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
    g = [3.0, 4.0, 5.0, 6.0]
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
    g.edge(n1, n2, EdgeType.SENDS_TO, 2, {})

    # remove_sends_to_external should remove the edge
    cf = remove_sends_to_external_transform()(g)
    assert len(cf.edges) == 0

    # remove_authorizes should keep the sends_to edge (not an authorizes edge)
    cf2 = remove_edge_type_transform(EdgeType.AUTHORIZES)(g)
    assert len(cf2.edges) == 1


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
