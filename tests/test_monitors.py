"""Tests for Phase 3 baseline monitors."""

from __future__ import annotations

from harness import (Agent, HarnessConfig, Orchestrator, Sandbox, Telemetry,
                     make_default_gateway, verify_determinism)
from harness.scenarios import get_scenario

from monitors import (FeatureMonitor, MonitorRegistry, RulesMonitor,
                      SequenceMonitor, TemporalSequenceMonitor, calibrate_all,
                      evaluate_episode)


def _run_trace(family: str, adversarial: bool, seed: int = 1):
    """Run an episode via the Phase 1 orchestrator and return (trace, steps,
    intervention_step, policy_violation)."""
    manifest, policy = get_scenario(family)
    telemetry = Telemetry(base_dir="data/traces")
    sandbox = Sandbox()
    gateway = make_default_gateway(telemetry)
    agent = Agent(name=f"mon_{family}")
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
    return trace, result["total_steps"], result["intervention_step"], result["policy_violation"]


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
            prefix = trace[:step]
            out = m.score(prefix, step)
            assert 0.0 <= out.score <= 1.0, (
                f"{name} score out of range at step {step}")
            assert out.monitor == name
            assert out.step == step


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
    examples = [(trace[:s], s, _label(s, intervention, steps))
                for s in range(1, steps + 1)]
    m.fit(examples)
    out = m.score(trace, steps)
    assert 0.0 <= out.score <= 1.0
    assert m._fitted


def test_sequence_monitor_fit_and_score():
    trace, steps, intervention, viol = _run_trace("data_exfiltration", True, seed=1)
    m = SequenceMonitor(name="sequence", window=None)
    examples = [(trace[:s], s, _label(s, intervention, steps))
                for s in range(1, steps + 1)]
    m.fit(examples)
    out = m.score(trace, steps)
    assert 0.0 <= out.score <= 1.0
    assert m._fitted


def test_temporal_monitor_fit_and_score():
    trace, steps, intervention, viol = _run_trace("data_exfiltration", True, seed=1)
    m = TemporalSequenceMonitor(name="temporal", window=5, use_torch=False)
    examples = [(trace[:s], s, _label(s, intervention, steps))
                for s in range(1, steps + 1)]
    m.fit(examples)
    out = m.score(trace, steps)
    assert 0.0 <= out.score <= 1.0
    assert m._fitted


def test_calibration_sets_thresholds():
    trace, steps, intervention, viol = _run_trace("data_exfiltration", True, seed=1)
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
    m.calibrate([(trace, steps)], target_fprs=(0.05,))
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
