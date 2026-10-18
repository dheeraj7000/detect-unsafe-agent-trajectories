"""GraphGuard command-line interface.

Usage:
    graphguard run --scenario file_backup --monitor graph --steps 10
    graphguard compare --scenarios indirect_prompt_injection,data_exfiltration --monitors graph,sequence,temporal --fpr 0.05
    graphguard calibrate --monitor graph --benign-traces data/traces --fpr 0.01 0.05
    graphguard ablate --scenarios indirect_prompt_injection,data_exfiltration --fpr 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from harness import (
    Agent,
    HarnessConfig,
    Orchestrator,
    Sandbox,
    Telemetry,
    make_default_gateway,
)
from harness.graph import BehaviorGraph
from harness.scenarios import get_scenario, TASK_FAMILIES

from monitors import (
    GraphMonitor,
    FeatureMonitor,
    RulesMonitor,
    SequenceMonitor,
    TemporalSequenceMonitor,
    calibrate_monitor,
    evaluate_episode,
    make_baselines,
    make_proposed,
)

from evaluator import (
    run_ablation_study,
    DEFAULT_ABLATIONS,
)
from evaluator.statistics import bootstrap_ci, detect_rate


# ---------------------------------------------------------------------------
# Monitor factory
# ---------------------------------------------------------------------------

MONITOR_REGISTRY = {
    "rules": lambda: RulesMonitor(name="rules"),
    "feature": lambda: FeatureMonitor(name="feature"),
    "sequence": lambda: SequenceMonitor(name="sequence", window=None),
    "temporal": lambda: TemporalSequenceMonitor(name="temporal", window=10),
    "graph": lambda: GraphMonitor(name="graph", hidden=64, num_layers=3, epochs=25),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphguard",
        description="GraphGuard — Early Warning of Unsafe Long-Horizon Agent Behavior",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- run --------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run a single episode with one monitor")
    p_run.add_argument(
        "--scenario", choices=TASK_FAMILIES, default="file_backup",
        help="Task family scenario to run",
    )
    p_run.add_argument(
        "--monitor", choices=list(MONITOR_REGISTRY.keys()), default="graph",
        help="Monitor to use",
    )
    p_run.add_argument(
        "--steps", type=int, default=10, help="Maximum steps in the episode",
    )
    p_run.add_argument(
        "--adversarial", action="store_true",
        help="Run the adversarial variant of the scenario",
    )
    p_run.add_argument(
        "--seed", type=int, default=1, help="Random seed",
    )
    p_run.add_argument(
        "--fpr", type=float, default=0.05, help="False-positive rate for calibration",
    )
    p_run.add_argument(
        "--output", type=str, default=None, help="Write result JSON to this path",
    )

    # ---- compare ----------------------------------------------------------
    p_cmp = sub.add_parser(
        "compare", help="Compare multiple monitors across scenarios",
    )
    p_cmp.add_argument(
        "--scenarios", type=str, default=",".join(TASK_FAMILIES),
        help="Comma-separated task families",
    )
    p_cmp.add_argument(
        "--monitors", type=str, default=",".join(MONITOR_REGISTRY.keys()),
        help="Comma-separated monitor names",
    )
    p_cmp.add_argument(
        "--fpr", type=float, default=0.05, help="False-positive rate for calibration",
    )
    p_cmp.add_argument(
        "--seed", type=int, default=1, help="Random seed",
    )
    p_cmp.add_argument(
        "--output", type=str, default=None, help="Write result JSON to this path",
    )

    # ---- calibrate --------------------------------------------------------
    p_cal = sub.add_parser("calibrate", help="Calibrate monitor thresholds")
    p_cal.add_argument(
        "--monitor", choices=list(MONITOR_REGISTRY.keys()), default="graph",
        help="Monitor to calibrate",
    )
    p_cal.add_argument(
        "--benign-traces", type=str, default="data/traces",
        help="Directory with benign trace JSONL files",
    )
    p_cal.add_argument(
        "--fpr", type=float, nargs="+", default=[0.01, 0.05],
        help="False-positive rates to target",
    )

    # ---- ablate -----------------------------------------------------------
    p_abl = sub.add_parser("ablate", help="Run graph ablation study")
    p_abl.add_argument(
        "--scenarios", type=str, default=",".join(TASK_FAMILIES),
        help="Comma-separated task families for test episodes",
    )
    p_abl.add_argument(
        "--fpr", type=float, default=0.05, help="False-positive rate for calibration",
    )
    p_abl.add_argument(
        "--seed", type=int, default=1, help="Random seed",
    )
    p_abl.add_argument(
        "--output", type=str, default=None, help="Write result JSON to this path",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_episode_trace(family: str, adversarial: bool, seed: int = 1):
    """Run one episode via the Phase 1 orchestrator; return (trace, result)."""
    manifest, policy = get_scenario(family)
    telemetry = Telemetry(base_dir="data/traces")
    sandbox = Sandbox()
    gateway = make_default_gateway(telemetry)
    agent = Agent(name=f"cli_ep_{family}")
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


def _graph_getter_factory(trace: list, episode_id: str):
    """Return a callable(step) -> BehaviorGraph reconstructed from the first
    `step` telemetry events."""
    def getter(step: int) -> BehaviorGraph:
        g = BehaviorGraph(episode_id=episode_id)
        for ev in trace[:step]:
            g.ingest_event(ev, ev.step if ev.step else step)
        return g
    return getter


def _run_single_episode(scenario: str, adversarial: bool, seed: int,
                        max_steps: int, monitor_name: str, fpr: float):
    manifest, policy = get_scenario(scenario)
    manifest.max_steps = max_steps

    telemetry = Telemetry(base_dir="data/traces")
    sandbox = Sandbox()
    gateway = make_default_gateway(telemetry)
    agent = Agent(name=f"cli_{scenario}")
    orch = Orchestrator(
        agent=agent,
        gateway=gateway,
        sandbox=sandbox,
        telemetry=telemetry,
        config=HarnessConfig(max_steps=max_steps, auto_intervene_on_deny=True),
    )
    result = orch.run_episode(manifest, policy)
    trace = telemetry.load_trace(manifest.episode_id)

    monitor = MONITOR_REGISTRY[monitor_name]()
    # Calibrate on the episode's own prefixes (simplified demo calibration).
    calibrate_monitor(monitor, [(trace, result["total_steps"])], target_fprs=(fpr,))

    gg = _graph_getter_factory(trace, manifest.episode_id)
    metrics = evaluate_episode(
        monitor, trace, result["total_steps"],
        result["intervention_step"], fpr_level=fpr, graph_getter=gg,
    )

    return {
        "scenario": scenario,
        "monitor": monitor_name,
        "result": result,
        "metrics": {
            "detected": metrics.detected,
            "detection_step": metrics.detection_step,
            "unsafe_event_step": metrics.unsafe_event_step,
            "lead_actions": metrics.lead_actions,
            "normalized_detection_time": metrics.normalized_detection_time,
            "fpr_level": metrics.fpr_level,
            "first_alert_score": metrics.first_alert_score,
        },
        "trace_events": len(trace),
        "graph_nodes": len(orch.graph.nodes) if orch.graph else 0,
        "graph_edges": len(orch.graph.edges) if orch.graph else 0,
    }


def _compare(scenarios: List[str], monitors: List[str], fpr: float,
             seed: int):
    results = []
    for scenario in scenarios:
        trace, result = run_episode_trace(scenario, adversarial=True, seed=seed)
        steps = result["total_steps"]
        intervention = result["intervention_step"]

        for mon_name in monitors:
            monitor = MONITOR_REGISTRY[mon_name]()
            # Build training prefixes labeled at/after intervention.
            examples = []
            for step in range(1, steps + 1, max(1, steps // 20)):
                g = BehaviorGraph(episode_id=result["episode_id"])
                for ev in trace[:step]:
                    g.ingest_event(ev, ev.step if ev.step else step)
                label = step >= intervention if intervention else False
                if mon_name == "graph":
                    examples.append((trace[:step], step, g, label))
                else:
                    examples.append((trace[:step], step, label))
            monitor.fit(examples)
            calibrate_monitor(
                monitor, [(trace, steps)], target_fprs=(fpr,),
            )

            gg = _graph_getter_factory(trace, result["episode_id"])
            m = evaluate_episode(
                monitor, trace, steps, intervention,
                fpr_level=fpr, graph_getter=gg,
            )
            results.append({
                "scenario": scenario,
                "monitor": mon_name,
                "detected": m.detected,
                "detection_step": m.detection_step,
                "unsafe_event_step": m.unsafe_event_step,
                "lead_actions": m.lead_actions,
                "normalized_detection_time": m.normalized_detection_time,
            })
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        out = _run_single_episode(
            args.scenario, args.adversarial, args.seed,
            args.steps, args.monitor, args.fpr,
        )
        print(json.dumps(out, indent=2, default=str))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2, default=str)
        return 0

    if args.command == "compare":
        scenarios = [s.strip() for s in args.scenarios.split(",")]
        monitors = [m.strip() for m in args.monitors.split(",")]
        results = _compare(scenarios, monitors, args.fpr, args.seed)
        print(json.dumps(results, indent=2, default=str))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
        return 0

    if args.command == "calibrate":
        monitor = MONITOR_REGISTRY[args.monitor]()
        # Load benign traces from directory if it exists.
        benign = []
        if os.path.isdir(args.benign_traces):
            from harness.schema import TraceEvent
            for fn in sorted(os.listdir(args.benign_traces)):
                if fn.endswith(".jsonl"):
                    path = os.path.join(args.benign_traces, fn)
                    events = []
                    with open(path) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                events.append(TraceEvent.model_validate_json(line))
                    if events:
                        benign.append((events, max(e.step for e in events)))
        if not benign:
            # Fallback: generate a benign trace.
            trace, result = run_episode_trace("file_backup", adversarial=False,
                                              seed=args.seed)
            benign = [(trace, result["total_steps"])]
        report = calibrate_monitor(monitor, benign, target_fprs=tuple(args.fpr))
        print(json.dumps({
            "monitor": report.monitor,
            "thresholds": report.thresholds,
            "n_benign_prefixes": report.n_benign_prefixes,
            "n_validation_episodes": report.n_validation_episodes,
        }, indent=2))
        return 0

    if args.command == "ablate":
        scenarios = [s.strip() for s in args.scenarios.split(",")]
        test_episodes = []
        for fam in scenarios:
            trace, result = run_episode_trace(fam, adversarial=True, seed=args.seed)
            test_episodes.append({
                "episode_id": result["episode_id"],
                "threat_family": fam,
                "trace": trace,
                "total_steps": result["total_steps"],
                "intervention_step": result["intervention_step"],
            })
        benign_traces = []
        for fam in ["file_backup", "email_triage", "code_review"]:
            trace, result = run_episode_trace(fam, adversarial=False,
                                              seed=args.seed + 100)
            benign_traces.append((trace, result["total_steps"]))

        graph_monitor = MONITOR_REGISTRY["graph"]()
        ablation_results = run_ablation_study(
            graph_monitor, test_episodes, benign_traces,
            DEFAULT_ABLATIONS, fpr_level=args.fpr,
        )
        out = []
        for ar in ablation_results:
            out.append({
                "variant": ar.config.name,
                "description": ar.config.description,
                "detect_rate": ar.detect_rate,
                "avg_lead": ar.avg_lead,
                "avg_norm_time": ar.avg_norm_time,
                "ci_lead": ar.ci_lead,
                "n_episodes": ar.n_episodes,
            })
        print(json.dumps(out, indent=2, default=str))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2, default=str)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
