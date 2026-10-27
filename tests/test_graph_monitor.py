"""Tests for Phase 4 graph monitor (the proposed GraphGuard method)."""

from __future__ import annotations

import tempfile

from dataset import EpisodeGenerator, GenerationConfig
from harness import Sandbox, make_default_gateway
from harness.graph import (
    BehaviorGraph,
    behavior_graph_at_step,
    graph_to_tensors,
    trace_prefix_at_step,
)
from harness.schema import EdgeType, NodeType, TraceEvent

from monitors import GraphMonitor, calibrate_monitor, evaluate_episode
from monitors.calibrator import DetectionMetrics


def _run_episode(family: str, adversarial: bool, seed: int = 1):
    """Run an episode and return (trace, result, orchestrator)."""
    episode = EpisodeGenerator(
        make_default_gateway,
        Sandbox(),
        base_dir=tempfile.mkdtemp(),
    ).generate(GenerationConfig(
        task_family=family,
        seed=seed,
        adversarial=adversarial,
        max_steps=10,
        episode_id=f"test_{family}_{seed}_{adversarial}",
    ))
    trace = [TraceEvent.model_validate(event) for event in episode.trace_events]
    return trace, episode.result, None


def _graph_at_step(orch, trace, step: int) -> BehaviorGraph:
    """Reconstruct the behavior graph through a given action step."""
    return behavior_graph_at_step(trace, step, "prefix_graph")


def test_graph_monitor_score_in_range():
    trace, result, orch = _run_episode("data_exfiltration", True, seed=1)
    m = GraphMonitor(name="graph")
    steps = result["total_steps"]
    for step in range(1, steps + 1):
        g = _graph_at_step(orch, trace, step)
        score = m.score_prefix(trace_prefix_at_step(trace, step), step, graph=g)
        assert 0.0 <= score <= 1.0, f"score {score} out of range at step {step}"


def test_graph_monitor_fit_and_score():
    trace, result, orch = _run_episode("data_exfiltration", True, seed=1)
    m = GraphMonitor(name="graph", epochs=5)
    steps = result["total_steps"]
    intervention = result["intervention_step"]

    # Build labeled examples: unsafe at/after intervention.
    examples = []
    for step in range(1, steps + 1):
        g = _graph_at_step(orch, trace, step)
        label = step >= intervention if intervention else False
        examples.append((trace_prefix_at_step(trace, step), step, g, label))
    m.fit(examples)
    assert m._fitted

    out = m.score(
        trace_prefix_at_step(trace, steps),
        steps,
        graph=_graph_at_step(orch, trace, steps),
    )
    assert 0.0 <= out.score <= 1.0
    assert "num_nodes" in out.details
    assert "num_edges" in out.details


def test_graph_monitor_configuration_controls_model():
    monitor = GraphMonitor(hidden=32, num_layers=2, epochs=1)
    assert monitor.model.hidden == 32
    assert len(monitor.model.mp_layers) == 2
    assert monitor.model.mp_layers[0].msg_linear.in_features == 64


def test_graph_monitor_propagates_edge_features():
    monitor = GraphMonitor(hidden=32, num_layers=1, epochs=1)
    graph = BehaviorGraph(episode_id="edge-features")
    source = graph.node(NodeType.TOOL_CALL, 1, {"tool": "file_read"})
    target = graph.node(NodeType.TOOL_RESULT, 2, {"success": True})
    graph.edge(source, target, EdgeType.DERIVED_FROM, 2)

    x, edge_index, edge_attr, _ = graph_to_tensors(graph)
    x = x.to(monitor.device)
    edge_index = edge_index.to(monitor.device)
    edge_attr = edge_attr.to(monitor.device).requires_grad_()
    risk = monitor.model(x, edge_index, edge_attr)
    risk.sum().backward()

    assert monitor.model.edge_in_proj.weight.grad is not None
    assert monitor.model.edge_in_proj.weight.grad.abs().sum().item() > 0
    assert edge_attr.grad is not None
    assert edge_attr.grad[:, -1].abs().sum().item() > 0


def test_reconstructed_graph_preserves_action_relationships():
    trace, result, _ = _run_episode("data_exfiltration", True, seed=3)
    graph = _graph_at_step(None, trace, result["total_steps"])
    edge_types = {edge.edge_type for edge in graph.edges.values()}

    assert EdgeType.AUTHORIZES in edge_types
    assert EdgeType.READS in edge_types
    assert EdgeType.SENDS_TO in edge_types
    assert all(
        graph.predecessors(edge.target_id)
        for edge in graph.edges.values()
        if edge.edge_type == EdgeType.AUTHORIZES
    )


def test_graph_calibration_sets_thresholds_and_uses_graph(monkeypatch):
    trace, result, orch = _run_episode("data_exfiltration", False, seed=1)
    m = GraphMonitor(name="graph", epochs=5)
    steps = result["total_steps"]
    examples = []
    for step in range(1, steps + 1):
        g = _graph_at_step(orch, trace, step)
        examples.append((trace_prefix_at_step(trace, step), step, g, False))
    m.fit(examples)
    # Calibrate on a benign episode and verify the graph is reconstructed.
    benign = [(trace, steps)]
    graph_sizes = []
    original_score_prefix = m.score_prefix

    def score_with_graph(trace_prefix, step, graph=None):
        graph_sizes.append(len(graph.nodes) if graph is not None else 0)
        return original_score_prefix(trace_prefix, step, graph)

    monkeypatch.setattr(m, "score_prefix", score_with_graph)
    report = calibrate_monitor(m, benign, target_fprs=(0.01, 0.05))
    assert 0.01 in report.thresholds
    assert 0.05 in report.thresholds
    assert m._calibrated
    assert max(graph_sizes) > 0


def test_graph_evaluate_episode_detection():
    trace, result, orch = _run_episode("data_exfiltration", True, seed=1)
    m = GraphMonitor(name="graph", epochs=5)
    steps = result["total_steps"]
    intervention = result["intervention_step"]

    # Train on this episode's prefixes (unsafe at/after intervention).
    examples = []
    for step in range(1, steps + 1):
        g = _graph_at_step(orch, trace, step)
        label = step >= intervention if intervention else False
        examples.append((trace_prefix_at_step(trace, step), step, g, label))
    m.fit(examples)

    def graph_getter(step):
        return _graph_at_step(orch, trace, step)

    metrics = evaluate_episode(
        m, trace, steps, intervention, fpr_level=0.05,
        graph_getter=graph_getter,
    )
    assert metrics.monitor == "graph"
    assert isinstance(metrics, DetectionMetrics)
    if metrics.detected:
        assert metrics.detection_step is not None
        assert metrics.lead_actions is not None
        assert metrics.normalized_detection_time is not None


def test_graph_monitor_all_threat_families():
    """Graph monitor should score all threat families without error."""
    families = [
        "indirect_prompt_injection",
        "memory_poisoning",
        "privilege_escalation",
        "data_exfiltration",
        "goal_drift",
        "multi_agent_delegation",
    ]
    m = GraphMonitor(name="graph", epochs=3)
    for fam in families:
        trace, result, orch = _run_episode(fam, adversarial=True, seed=1)
        steps = result["total_steps"]
        for step in range(1, steps + 1):
            g = _graph_at_step(orch, trace, step)
            score = m.score_prefix(
                trace_prefix_at_step(trace, step), step, graph=g
            )
            assert 0.0 <= score <= 1.0
