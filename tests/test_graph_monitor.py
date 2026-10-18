"""Tests for Phase 4 graph monitor (the proposed GraphGuard method)."""

from __future__ import annotations

from harness import (Agent, HarnessConfig, Orchestrator, Sandbox, Telemetry,
                     make_default_gateway)
from harness.graph import BehaviorGraph
from harness.scenarios import get_scenario

from monitors import GraphMonitor, calibrate_monitor, evaluate_episode
from monitors.calibrator import DetectionMetrics


def _run_episode(family: str, adversarial: bool, seed: int = 1):
    """Run an episode and return (trace, result, orchestrator)."""
    manifest, policy = get_scenario(family)
    telemetry = Telemetry(base_dir="data/traces")
    sandbox = Sandbox()
    gateway = make_default_gateway(telemetry)
    agent = Agent(name=f"gm_{family}")
    orch = Orchestrator(
        agent=agent,
        gateway=gateway,
        sandbox=sandbox,
        telemetry=telemetry,
        config=HarnessConfig(max_steps=manifest.max_steps,
                             auto_intervene_on_deny=True),
    )
    result = orch.run_episode(manifest, policy)
    trace = telemetry.load_trace(manifest.episode_id)
    return trace, result, orch


def _graph_at_step(orch: Orchestrator, trace, step: int) -> BehaviorGraph:
    """Reconstruct the behavior graph at a given prefix step by ingesting
    the first `step` telemetry events into a fresh graph."""
    g = BehaviorGraph(episode_id="prefix_graph")
    for ev in trace[:step]:
        # ev.step should equal the prefix position; use it for consistency.
        g.ingest_event(ev, ev.step if ev.step else step)
    return g


def test_graph_monitor_score_in_range():
    trace, result, orch = _run_episode("data_exfiltration", True, seed=1)
    m = GraphMonitor(name="graph")
    steps = result["total_steps"]
    for step in range(1, steps + 1):
        g = _graph_at_step(orch, trace, step)
        score = m.score_prefix(trace[:step], step, graph=g)
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
        examples.append((trace[:step], step, g, label))
    m.fit(examples)
    assert m._fitted

    out = m.score(trace[:steps], steps, graph=_graph_at_step(orch, trace, steps))
    assert 0.0 <= out.score <= 1.0
    assert "num_nodes" in out.details
    assert "num_edges" in out.details


def test_graph_calibration_sets_thresholds():
    trace, result, orch = _run_episode("data_exfiltration", True, seed=1)
    m = GraphMonitor(name="graph", epochs=5)
    steps = result["total_steps"]
    examples = []
    for step in range(1, steps + 1):
        g = _graph_at_step(orch, trace, step)
        examples.append((trace[:step], step, g, False))  # all benign for calibration
    m.fit(examples)
    # Calibrate on benign traces (pretend all benign).
    benign = [(trace, steps)]
    report = calibrate_monitor(m, benign, target_fprs=(0.01, 0.05))
    assert 0.01 in report.thresholds
    assert 0.05 in report.thresholds
    assert m._calibrated


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
        examples.append((trace[:step], step, g, label))
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
            score = m.score_prefix(trace[:step], step, graph=g)
            assert 0.0 <= score <= 1.0
