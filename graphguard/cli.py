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
    Sandbox,
    make_default_gateway,
)
from harness.graph import BehaviorGraph, behavior_graph_at_step, trace_prefix_at_step
from harness.scenarios import TASK_FAMILIES
from harness.schema import TraceEvent
from dataset import EpisodeGenerator, GenerationConfig

from monitors import (
    GraphMonitor,
    FeatureMonitor,
    RulesMonitor,
    SequenceMonitor,
    TemporalSequenceMonitor,
    TransformerSequenceMonitor,
    calibrate_monitor,
    evaluate_episode,
)

from evaluator import (
    run_ablation_study,
    DEFAULT_ABLATIONS,
)
# ---------------------------------------------------------------------------
# Monitor factory
# ---------------------------------------------------------------------------

MONITOR_REGISTRY = {
    "rules": lambda: RulesMonitor(name="rules"),
    "feature": lambda: FeatureMonitor(name="feature"),
    "sequence": lambda: SequenceMonitor(name="sequence", window=None),
    "temporal": lambda: TemporalSequenceMonitor(name="temporal", window=10),
    "transformer": lambda: TransformerSequenceMonitor(name="transformer"),
    "graph": lambda: GraphMonitor(name="graph", hidden=64, num_layers=3, epochs=25),
}


def _make_monitor(name: str, seed: int = 42):
    if name == "graph":
        return GraphMonitor(
            name="graph", hidden=64, num_layers=3, epochs=25, seed=seed
        )
    if name == "temporal":
        return TemporalSequenceMonitor(name="temporal", window=10, seed=seed)
    if name == "transformer":
        return TransformerSequenceMonitor(name="transformer", seed=seed)
    return MONITOR_REGISTRY[name]()
DEFAULT_TRAIN_SCENARIOS = (
    "indirect_prompt_injection,memory_poisoning,privilege_escalation"
)
DEFAULT_TEST_SCENARIOS = (
    "data_exfiltration,goal_drift,multi_agent_delegation"
)


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
        "--scenarios", type=str, default=DEFAULT_TEST_SCENARIOS,
        help="Comma-separated task families",
    )
    p_cmp.add_argument(
        "--monitors", type=str, default=",".join(MONITOR_REGISTRY.keys()),
        help="Comma-separated monitor names",
    )
    p_cmp.add_argument(
        "--train-scenarios", type=str, default=DEFAULT_TRAIN_SCENARIOS,
        help="Comma-separated training attack families, disjoint from test",
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
    p_cal.add_argument("--seed", type=int, default=1, help="Random seed")

    # ---- ablate -----------------------------------------------------------
    p_abl = sub.add_parser("ablate", help="Run graph ablation study")
    p_abl.add_argument(
        "--scenarios", type=str, default=DEFAULT_TEST_SCENARIOS,
        help="Comma-separated task families for test episodes",
    )
    p_abl.add_argument(
        "--train-scenarios", type=str, default=DEFAULT_TRAIN_SCENARIOS,
        help="Comma-separated attack families for training (disjoint from test)",
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

def run_episode_trace(
    family: str,
    adversarial: bool,
    seed: int = 1,
    max_steps: Optional[int] = None,
    wording_variant: str = "default",
):
    """Run one episode via the Phase 1 orchestrator; return (trace, result)."""
    episode_id = (
        f"cli_{family}_{seed}_{'adversarial' if adversarial else 'benign'}"
    )
    episode = EpisodeGenerator(
        make_default_gateway, Sandbox(), base_dir="data/traces"
    ).generate(GenerationConfig(
        task_family=family,
        seed=seed,
        adversarial=adversarial,
        max_steps=max_steps or 10,
        episode_id=episode_id,
        template_id=family,
        wording_variant=wording_variant,
    ))
    trace = [
        TraceEvent.model_validate(event) for event in episode.trace_events
    ]
    return trace, episode.result


def _graph_getter_factory(trace: list, episode_id: str):
    """Return a callable(step) -> graph reconstructed through an action step."""
    def getter(step: int) -> BehaviorGraph:
        return behavior_graph_at_step(trace, step, episode_id)
    return getter


def _build_training_examples(families: List[str], seed: int):
    examples = []
    episodes = [
        (family, False, seed + index)
        for index, family in enumerate(
            ("file_backup", "email_triage")
        )
    ]
    episodes.extend(
        (family, True, seed + 10 + index)
        for index, family in enumerate(families)
    )
    for family, adversarial, episode_seed in episodes:
        trace, result = run_episode_trace(
            family, adversarial, seed=episode_seed,
            wording_variant="neutral-0",
        )
        for step in range(1, result["total_steps"] + 1):
            prefix = trace_prefix_at_step(trace, step)
            graph = behavior_graph_at_step(trace, step, result["episode_id"])
            label = bool(
                result["intervention_step"]
                and step >= result["intervention_step"]
            )
            examples.append((prefix, step, graph, label))
    return examples


def _build_episode_records(families: List[str], seed: int):
    records = []
    specs = [
        (family, False, seed + index)
        for index, family in enumerate(
            ("file_backup", "email_triage")
        )
    ]
    specs.extend(
        (family, True, seed + 10 + index)
        for index, family in enumerate(families)
    )
    for family, adversarial, episode_seed in specs:
        trace, result = run_episode_trace(
            family, adversarial, seed=episode_seed,
            wording_variant="neutral-0",
        )
        records.append({
            "episode_id": result["episode_id"],
            "template_id": family,
            "trace": trace,
            "total_steps": result["total_steps"],
            "intervention_step": result["intervention_step"],
        })
    return records


def _run_single_episode(
    scenario: str,
    adversarial: bool,
    seed: int,
    max_steps: int,
    monitor_name: str,
    fpr: float,
):
    trace, result = run_episode_trace(
        scenario, adversarial, seed=seed, max_steps=max_steps
    )
    monitor = _make_monitor(monitor_name, seed)
    if hasattr(monitor, "fit"):
        training = _build_training_examples(
            [family for family in (
                "indirect_prompt_injection",
                "memory_poisoning",
                "privilege_escalation",
            ) if family != scenario],
            seed + 10,
        )
        if training:
            monitor.fit(training if monitor_name == "graph" else [
                (prefix, step, label)
                for prefix, step, _graph, label in training
            ])
    benign = [
        run_episode_trace(
            family,
            False,
            seed + 100 + index,
            wording_variant="neutral-1",
        )
        for index, family in enumerate(
            ("code_review",)
        )
    ]
    calibrate_monitor(
        monitor,
        [(events, result["total_steps"]) for events, result in benign],
        target_fprs=(fpr,),
    )

    gg = _graph_getter_factory(trace, result["episode_id"])
    graph = gg(result["total_steps"])
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
        "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges),
    }


def _compare(
    scenarios: List[str],
    monitors: List[str],
    fpr: float,
    seed: int,
    train_scenarios: List[str],
):
    overlap = set(scenarios) & set(train_scenarios)
    if overlap:
        raise ValueError(
            f"Training and test scenarios must be disjoint: {sorted(overlap)}"
        )
    train_templates = set(train_scenarios) | {"file_backup", "email_triage"}
    validation_templates = {"code_review"}
    test_templates = set(scenarios)
    if (
        train_templates & validation_templates
        or train_templates & test_templates
        or validation_templates & test_templates
    ):
        raise ValueError(
            "Training, calibration, and test templates must be disjoint."
        )
    train_examples = _build_training_examples(train_scenarios, seed + 100)
    benign_traces = [
        run_episode_trace(
            family,
            False,
            seed + 200 + index,
            wording_variant="neutral-1",
        )
        for index, family in enumerate(
            ("code_review",)
        )
    ]
    results = []
    for scenario_index, scenario in enumerate(scenarios):
        trace, result = run_episode_trace(
            scenario,
            adversarial=True,
            seed=seed + 300 + scenario_index,
            wording_variant="neutral-2",
        )
        steps = result["total_steps"]
        intervention = result["intervention_step"]

        for mon_name in monitors:
            monitor = _make_monitor(mon_name, seed)
            if hasattr(monitor, "fit"):
                examples = (
                    train_examples
                    if mon_name == "graph"
                    else [
                        (prefix, step, label)
                        for prefix, step, _graph, label in train_examples
                    ]
                )
                monitor.fit(examples)
            calibrate_monitor(
                monitor,
                [(events, benign_result["total_steps"])
                 for events, benign_result in benign_traces],
                target_fprs=(fpr,),
            )

            m = evaluate_episode(
                monitor, trace, steps, intervention,
                fpr_level=fpr,
                graph_getter=_graph_getter_factory(trace, result["episode_id"]),
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
        train_scenarios = [s.strip() for s in args.train_scenarios.split(",")]
        results = _compare(
            scenarios, monitors, args.fpr, args.seed, train_scenarios
        )
        print(json.dumps(results, indent=2, default=str))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
        return 0

    if args.command == "calibrate":
        monitor = _make_monitor(args.monitor, args.seed)
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
                    is_benign = not any(
                        event.event_type.value == "episode_end"
                        and event.payload.get("policy_violation")
                        for event in events
                    )
                    if events and is_benign:
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
        train_scenarios = [
            s.strip() for s in args.train_scenarios.split(",")
        ]
        overlap = set(scenarios) & set(train_scenarios)
        if overlap:
            raise ValueError(
                f"Training and test scenarios must be disjoint: {sorted(overlap)}"
            )
        train_templates = set(train_scenarios) | {
            "file_backup",
            "email_triage",
        }
        validation_templates = {"code_review"}
        test_templates = set(scenarios)
        if (
            train_templates & validation_templates
            or train_templates & test_templates
            or validation_templates & test_templates
        ):
            raise ValueError(
                "Training, calibration, and test templates must be disjoint."
            )
        train_episodes = _build_episode_records(train_scenarios, args.seed)
        test_episodes = []
        for fam in scenarios:
            trace, result = run_episode_trace(
                fam,
                adversarial=True,
                seed=args.seed,
                wording_variant="neutral-2",
            )
            test_episodes.append({
                "episode_id": result["episode_id"],
                "threat_family": fam,
                "template_id": fam,
                "trace": trace,
                "total_steps": result["total_steps"],
                "intervention_step": result["intervention_step"],
            })
        benign_traces = []
        for fam in ["code_review"]:
            trace, result = run_episode_trace(fam, adversarial=False,
                                              seed=args.seed + 100,
                                              wording_variant="neutral-1")
            benign_traces.append((trace, result["total_steps"]))

        graph_monitor = _make_monitor("graph", args.seed)
        ablation_results = run_ablation_study(
            graph_monitor, train_episodes, test_episodes, benign_traces,
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
