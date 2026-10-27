"""Tests for Phase 3 baseline monitors."""

from __future__ import annotations

import tempfile

from dataset import EpisodeGenerator, GenerationConfig
from harness import Sandbox, make_default_gateway
from harness.graph import trace_prefix_at_step
from harness.schema import TraceEvent

from monitors import (FeatureMonitor, MonitorRegistry, RulesMonitor,
                      SequenceMonitor, TemporalSequenceMonitor,
                      TransformerSequenceMonitor, calibrate_all,
                      evaluate_episode)


def _run_trace(family: str, adversarial: bool, seed: int = 1):
    """Run an episode via the Phase 1 orchestrator and return (trace, steps,
    intervention_step, policy_violation)."""
    episode = EpisodeGenerator(
        make_default_gateway,
        Sandbox(),
        base_dir=tempfile.mkdtemp(),
    ).generate(GenerationConfig(
        task_family=family,
        seed=seed,
        adversarial=adversarial,
        max_steps=10,
        episode_id=f"monitor_{family}_{seed}_{adversarial}",
    ))
    trace = [TraceEvent.model_validate(event) for event in episode.trace_events]
    result = episode.result
    return (
        trace,
        result["total_steps"],
        result["intervention_step"],
        result["policy_violation"],
    )


def _label(step: int, intervention_step, total_steps: int) -> bool:
    """Label a prefix as unsafe if it is at/after the intervention onset."""
    if intervention_step is None:
        return False
    return step >= intervention_step


def test_all_monitors_score_in_range():
    trace, steps, intervention, viol = _run_trace("data_exfiltration", True, seed=1)
    registry = MonitorRegistry()
    for name in registry.names():
        m = registry.make(name)
        for step in range(1, steps + 1):
            prefix = trace_prefix_at_step(trace, step)
            out = m.score(prefix, step)
            assert 0.0 <= out.score <= 1.0, (
                f"{name} score out of range at step {step}")
            assert out.monitor == name
            assert out.step == step


def test_prefix_uses_action_step_not_event_count():
    trace, _, _, _ = _run_trace("data_exfiltration", True, seed=1)
    prefix = trace_prefix_at_step(trace, 1)
    assert len(prefix) > 1
    assert all(event.step <= 1 for event in prefix)


def test_rules_monitor_denies():
    trace, steps, intervention, viol = _run_trace("data_exfiltration", True, seed=1)
    m = RulesMonitor(name="rules")
    out = m.score(trace, steps)
    assert viol is True
    assert intervention is not None
    assert out.score > 0.0, "rules monitor should flag an episode with denials"


def test_feature_monitor_fit_and_score():
    trace, steps, intervention, viol = _run_trace("data_exfiltration", True, seed=1)
    m = FeatureMonitor(name="feature")
    examples = [(trace_prefix_at_step(trace, s), s, _label(s, intervention, steps))
                for s in range(1, steps + 1)]
    m.fit(examples)
    out = m.score(trace, steps)
    assert 0.0 <= out.score <= 1.0
    assert m._fitted


def test_sequence_monitor_fit_and_score():
    trace, steps, intervention, viol = _run_trace("data_exfiltration", True, seed=1)
    m = SequenceMonitor(name="sequence", window=None)
    examples = [(trace_prefix_at_step(trace, s), s, _label(s, intervention, steps))
                for s in range(1, steps + 1)]
    m.fit(examples)
    out = m.score(trace, steps)
    assert 0.0 <= out.score <= 1.0
    assert m._fitted


def test_temporal_monitor_fit_and_score():
    trace, steps, intervention, viol = _run_trace("data_exfiltration", True, seed=1)
    m = TemporalSequenceMonitor(name="temporal", window=5, use_torch=False)
    examples = [(trace_prefix_at_step(trace, s), s, _label(s, intervention, steps))
                for s in range(1, steps + 1)]
    m.fit(examples)
    out = m.score(trace, steps)
    assert 0.0 <= out.score <= 1.0
    assert m._fitted


def test_transformer_sequence_monitor_fit_and_score():
    trace, steps, intervention, _ = _run_trace(
        "data_exfiltration", True, seed=1
    )
    examples = []
    for step in range(1, steps + 1):
        prefix = trace_prefix_at_step(trace, step)
        examples.append((prefix, step, bool(intervention and step >= intervention)))

    monitor = TransformerSequenceMonitor(
        epochs=1,
        batch_size=2,
        model_dim=32,
        num_heads=4,
        num_layers=1,
        max_length=128,
        seed=7,
    )
    monitor.fit(examples)
    score = monitor.score_prefix(examples[-1][0], steps)
    assert monitor._fitted
    assert 0.0 <= score <= 1.0
    assert monitor.model is not None
    assert monitor.score(examples[-1][0], steps).details["parameters"] > 0


def test_calibration_sets_thresholds():
    trace, steps, intervention, viol = _run_trace("data_exfiltration", False, seed=1)
    m = RulesMonitor(name="rules")
    benign = [(trace, steps)]
    report = calibrate_all([m], benign, target_fprs=(0.01, 0.05))
    assert len(report) == 1
    assert 0.01 in report[0].thresholds
    assert 0.05 in report[0].thresholds
    assert m._calibrated


def test_evaluate_episode_detection():
    trace, steps, intervention, viol = _run_trace("data_exfiltration", True, seed=1)
    m = RulesMonitor(name="rules")
    benign_trace, benign_steps, _, _ = _run_trace(
        "data_exfiltration", False, seed=101
    )
    m.calibrate([(benign_trace, benign_steps)], target_fprs=(0.05,))
    metrics = evaluate_episode(
        m, trace, steps, intervention, fpr_level=0.05,
    )
    assert metrics.monitor == "rules"
    assert metrics.detected is True
    assert metrics.detection_step is not None
    assert metrics.unsafe_event_step == intervention
    assert metrics.lead_actions is not None
    assert metrics.normalized_detection_time is not None
    assert metrics.normalized_detection_time <= 1.0
