"""Ablation study for Phase 5.

Tests which graph structure adds signal beyond sequence models:
  - Remove provenance edges (derived_from, reads, writes, persists_to_memory)
  - Remove authorization edges (authorizes, escalates_privilege)
  - Remove temporal edges (follows)
  - Untype nodes/edges (collapse all to generic types)
  - Full graph vs bounded recent subgraph (last N steps)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from harness.graph import (
    BehaviorGraph,
    EdgeType,
    NodeType,
    behavior_graph_at_step,
    trace_prefix_at_step,
)
from monitors import GraphMonitor
from monitors.calibrator import calibrate_monitor, evaluate_episode
from evaluator.statistics import bootstrap_ci, detect_rate


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant."""

    name: str
    description: str
    remove_edge_types: List[EdgeType] = field(default_factory=list)
    untyped: bool = False  # collapse all node/edge types to generic
    bounded_window: Optional[int] = None  # keep only last N steps
    remove_temporal_features: bool = False


@dataclass
class AblationResult:
    """Results for one ablation variant."""

    config: AblationConfig
    monitor_name: str
    n_episodes: int
    detect_rate: float
    avg_lead: float
    avg_norm_time: float
    ci_lead: Optional[Tuple[float, float, float]] = None
    per_episode: List[dict] = field(default_factory=list)


def ablate_graph(graph: BehaviorGraph, config: AblationConfig) -> BehaviorGraph:
    """Return an ablated copy of ``graph`` per ``config``.

    Nodes are preserved; edges matching ``remove_edge_types`` are dropped.
    If ``untyped`` is True, all node and edge types are collapsed to a single
    generic type. If ``bounded_window`` is set, only edges/nodes from the last
    ``bounded_window`` steps are kept.
    """
    # Determine step cutoff for bounded window.
    step_cutoff = None
    if config.bounded_window is not None and graph.nodes:
        max_step = max(n.step for n in graph.nodes.values())
        step_cutoff = max_step - config.bounded_window

    # Copy nodes (optionally filtered by step cutoff).
    ablated = BehaviorGraph(episode_id=graph.episode_id)
    old_to_new: Dict[str, str] = {}

    for node in graph.nodes.values():
        if step_cutoff is not None and node.step < step_cutoff:
            continue
        node_type = NodeType.OBSERVATION if config.untyped else node.node_type
        new_id = ablated.node(
            node_type=node_type,
            step=0 if config.remove_temporal_features else node.step,
            attributes=dict(node.attributes),
        )
        old_to_new[node.node_id] = new_id

    # Copy edges.
    for edge in graph.edges.values():
        if edge.source_id not in old_to_new or edge.target_id not in old_to_new:
            continue
        if edge.edge_type in config.remove_edge_types:
            continue
        if step_cutoff is not None and edge.step < step_cutoff:
            continue
        edge_type = EdgeType.FOLLOWS if config.untyped else edge.edge_type
        ablated.edge(
            source_id=old_to_new[edge.source_id],
            target_id=old_to_new[edge.target_id],
            edge_type=edge_type,
            step=0 if config.remove_temporal_features else edge.step,
            attributes=dict(edge.attributes),
        )

    return ablated


def run_ablation_study(
    graph_monitor: GraphMonitor,
    train_episodes: List[dict],
    test_episodes: List[dict],
    benign_traces: List[Tuple[List[Any], int]],
    ablations: List[AblationConfig],
    fpr_level: float = 0.05,
) -> List[AblationResult]:
    """Run the ablation study.

    For each ablation variant:
      1. Train graph_monitor on training prefixes (using ablated graphs).
      2. Calibrate on benign traces.
      3. Evaluate on test adversarial episodes (using ablated graphs).
      4. Compute detection metrics + bootstrap CI for lead time.

    Args:
        graph_monitor: GraphMonitor to ablate (refit per variant).
        train_episodes: labeled training episodes, each with keys: trace,
            total_steps, intervention_step, episode_id.
        test_episodes: held-out evaluation episodes with keys: trace, total_steps,
            intervention_step, episode_id.
        benign_traces: list of (trace_events, total_steps) for calibration.
        ablations: list of AblationConfig variants.
        fpr_level: FPR level for calibration/evaluation.

    Returns:
        List of AblationResult, one per variant.
    """
    results = []

    train_ids = {ep["episode_id"] for ep in train_episodes}
    test_ids = {ep["episode_id"] for ep in test_episodes}
    calibration_ids = {
        trace[0].episode_id
        for trace, _ in benign_traces
        if trace
    }
    if train_ids & test_ids or calibration_ids & (train_ids | test_ids):
        raise ValueError("Training, calibration, and test episode IDs must be disjoint.")
    if not train_episodes or not test_episodes or not benign_traces:
        raise ValueError("Training, calibration, and test data must be non-empty.")

    train_templates = {
        ep["template_id"] for ep in train_episodes if ep.get("template_id")
    }
    test_templates = {
        ep["template_id"] for ep in test_episodes if ep.get("template_id")
    }
    if train_templates & test_templates:
        raise ValueError("Attack templates must be disjoint between train and test.")

    for config in ablations:
        # Build labeled training prefixes using ablated graphs.
        train_examples = []
        for ep in train_episodes:
            trace = ep["trace"]
            steps = ep["total_steps"]
            intervention = ep["intervention_step"]
            for step in range(1, steps + 1, max(1, steps // 20)):
                g_full = behavior_graph_at_step(trace, step, ep["episode_id"])
                g_abl = ablate_graph(g_full, config)
                label = step >= intervention if intervention else False
                train_examples.append(
                    (trace_prefix_at_step(trace, step), step, g_abl, label)
                )

        graph_monitor.fit(train_examples)

        def calibration_graph_getter(trace, step):
            episode_id = trace[0].episode_id if trace else "calibration"
            graph = behavior_graph_at_step(trace, step, episode_id)
            return ablate_graph(graph, config)

        calibrate_monitor(
            graph_monitor,
            benign_traces,
            target_fprs=(fpr_level,),
            graph_getter=calibration_graph_getter,
        )

        # Evaluate on test episodes with ablated graphs.
        per_episode = []
        lead_values = []
        detected_flags = []
        for ep in test_episodes:
            trace = ep["trace"]
            steps = ep["total_steps"]
            intervention = ep["intervention_step"]
            episode_id = ep["episode_id"]

            def graph_getter(step, tr=trace, cfg=config, eid=episode_id):
                g_full = behavior_graph_at_step(tr, step, eid)
                return ablate_graph(g_full, cfg)

            metrics = evaluate_episode(
                graph_monitor, trace, steps, intervention,
                fpr_level=fpr_level, graph_getter=graph_getter,
            )
            per_episode.append({
                "episode_id": episode_id,
                "detected": metrics.detected,
                "detection_step": metrics.detection_step,
                "lead_actions": metrics.lead_actions,
                "normalized_detection_time": metrics.normalized_detection_time,
            })
            detected_flags.append(metrics.detected)
            lead_values.append(metrics.lead_actions or 0)

        detect = detect_rate(detected_flags)
        avg_lead = float(np.mean(lead_values))
        nt_vals = [p["normalized_detection_time"] for p in per_episode
                   if p["normalized_detection_time"] is not None]
        avg_nt = float(np.mean(nt_vals)) if nt_vals else 0.0

        ci = None
        if lead_values:
            ci = bootstrap_ci(lead_values, n_bootstrap=2000, seed=42)

        results.append(AblationResult(
            config=config,
            monitor_name="graph",
            n_episodes=len(test_episodes),
            detect_rate=detect,
            avg_lead=avg_lead,
            avg_norm_time=avg_nt,
            ci_lead=ci,
            per_episode=per_episode,
        ))

    return results


# Preset ablation variants used in the study.
DEFAULT_ABLATIONS: List[AblationConfig] = [
    AblationConfig(
        name="full_graph",
        description="Complete typed behavior graph (no ablation)",
    ),
    AblationConfig(
        name="no_provenance",
        description="Remove provenance edges (derived_from, reads, writes)",
        remove_edge_types=[
            EdgeType.DERIVED_FROM, EdgeType.READS, EdgeType.WRITES,
            EdgeType.PERSISTS_TO_MEMORY,
        ],
    ),
    AblationConfig(
        name="no_authorization",
        description="Remove authorization edges (authorizes, escalates_privilege)",
        remove_edge_types=[
            EdgeType.AUTHORIZES, EdgeType.ESCALATES_PRIVILEGE,
        ],
    ),
    AblationConfig(
        name="no_temporal",
        description="Remove follows edges and node/edge step features",
        remove_edge_types=[EdgeType.FOLLOWS],
        remove_temporal_features=True,
    ),
    AblationConfig(
        name="untyped",
        description="Collapse all node/edge types to generic",
        untyped=True,
    ),
    AblationConfig(
        name="bounded_window_5",
        description="Bounded recent subgraph (last 5 steps)",
        bounded_window=5,
    ),
]
