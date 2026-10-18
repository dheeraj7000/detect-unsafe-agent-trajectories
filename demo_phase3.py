"""Phase 3 demo: baseline monitors calibrated + early-detection comparison.

Runs all four ordinary-trajectory baselines on adversarial + benign episodes,
calibrates them to FPR = 1% / 5%, and reports detection lead time.

Run:  python demo_phase3.py
"""

from __future__ import annotations

import json

from harness import (Agent, HarnessConfig, Orchestrator, Sandbox, Telemetry,
                     make_default_gateway)
from harness.scenarios import get_scenario

from monitors import (FeatureMonitor, RulesMonitor, SequenceMonitor,
                      TemporalSequenceMonitor, calibrate_monitor,
                      evaluate_episode, make_baselines)


def run_episode_trace(family: str, adversarial: bool, seed: int = 1):
    manifest, policy = get_scenario(family)
    telemetry = Telemetry(base_dir="data/traces")
    sandbox = Sandbox()
    gateway = make_default_gateway(telemetry)
    agent = Agent(name=f"p3_{family}")
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
    return trace, result


def main() -> None:
    print("=" * 70)
    print("GraphGuard Phase 3 — Baseline Monitors + Early Detection")
    print("=" * 70)

    # Build a small validation set (benign episodes) for calibration.
    benign_traces = []
    for fam in ["file_backup", "email_triage", "code_review"]:
        trace, result = run_episode_trace(fam, adversarial=False, seed=10)
        benign_traces.append((trace, result["total_steps"]))

    # Build test episodes: one adversarial per threat family.
    threat_families = [
        "indirect_prompt_injection",
        "memory_poisoning",
        "privilege_escalation",
        "data_exfiltration",
        "goal_drift",
        "multi_agent_delegation",
    ]
    test_episodes = []
    for fam in threat_families:
        trace, result = run_episode_trace(fam, adversarial=True, seed=20)
        test_episodes.append((fam, trace, result))

    # Instantiate baselines.
    monitors = make_baselines()

    # Calibrate each monitor on benign traces.
    print("\n--- Calibration (FPR targets 1% / 5%) ---")
    for m in monitors:
        report = calibrate_monitor(m, benign_traces, target_fprs=(0.01, 0.05))
        print(f"  {m.name:10s} thresholds: "
              f"1%={report.thresholds.get(0.01):.3f}  "
              f"5%={report.thresholds.get(0.05):.3f}")

    # Evaluate on each adversarial episode.
    print("\n--- Early detection at FPR = 5% ---")
    header = (f"{'scenario':26s} {'monitor':10s} {'detected':8s} "
              f"{'det_step':9s} {'unsafe_step':11s} {'lead':6s} "
              f"{'norm_time':9s}")
    print(header)
    print("-" * len(header))

    results = []
    for fam, trace, result in test_episodes:
        unsafe_step = result["unsafe_event_step"]
        total = result["total_steps"]
        for m in monitors:
            metrics = evaluate_episode(
                m, trace, total, unsafe_step, fpr_level=0.05,
            )
            results.append({
                "scenario": fam,
                "monitor": m.name,
                "detected": metrics.detected,
                "detection_step": metrics.detection_step,
                "unsafe_event_step": unsafe_step,
                "lead_actions": metrics.lead_actions,
                "normalized_detection_time": metrics.normalized_detection_time,
            })
            det_str = "Yes" if metrics.detected else "No"
            ds = metrics.detection_step if metrics.detection_step else "-"
            us = unsafe_step if unsafe_step else "-"
            lead = metrics.lead_actions if metrics.lead_actions is not None else "-"
            nt = (f"{metrics.normalized_detection_time:.2f}"
                  if metrics.normalized_detection_time is not None else "-")
            print(f"  {fam:26s} {m.name:10s} {det_str:8s} "
                  f"{str(ds):9s} {str(us):11s} {str(lead):6s} {nt:9s}")

    # Summary: average lead time & normalized detection time per monitor.
    print("\n--- Summary per monitor (adversarial episodes) ---")
    for m in monitors:
        mon_results = [r for r in results if r["monitor"] == m.name]
        detected = [r for r in mon_results if r["detected"]]
        if detected:
            avg_lead = sum(r["lead_actions"] for r in detected) / len(detected)
            avg_nt = (sum(r["normalized_detection_time"] for r in detected)
                      / len(detected))
            detect_rate = len(detected) / len(mon_results)
        else:
            avg_lead = avg_nt = detect_rate = 0.0
        print(f"  {m.name:10s} detect_rate={detect_rate:.2f} "
              f"avg_lead={avg_lead:.2f} avg_norm_time={avg_nt:.2f}")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
