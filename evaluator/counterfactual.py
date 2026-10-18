"""Counterfactual replay for Phase 5.

Alters one dependency in an episode's behavior graph and tests whether the
graph monitor's risk changes appropriately. Used for H4 (interpretability)
and structural validation: removing a dangerous dependency should lower risk;
adding one should raise it.

Example counterfactuals:
  - Remove the `authorizes` edge to a denied policy decision -> risk drops.
  - Remove the `sends_to` edge to an external destination -> risk drops.
  - Sanitize tool output (remove derived_from from untrusted output) -> risk drops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from harness.graph import BehaviorGraph, EdgeType, NodeType
from monitors import GraphMonitor


@dataclass
class CounterfactualResult:
    """Result of a single counterfactual test."""

    episode_id: str
    counterfactual_name: str
    description: str
    original_risk: float
    counterfactual_risk: float
    risk_delta: float  # counterfactual - original
    direction_correct: bool  # True if risk moved in expected direction
    expected_direction: str  # "decrease" or "increase"
    step: int


def counterfactual_replay(
    graph_monitor: GraphMonitor,
    trace: List[Any],
    total_steps: int,
    step: int,
    episode_id: str,
    counterfactuals: List[Dict[str, Any]],
) -> List[CounterfactualResult]:
    """Run counterfactual replay at a given prefix step.

    For each counterfactual spec, build the graph at ``step``, apply the
    counterfactual transformation, score both graphs, and check whether the
    risk moved in the expected direction.

    Args:
        graph_monitor: fitted GraphMonitor.
        trace: full telemetry trace for the episode.
        total_steps: total steps in the episode.
        step: prefix step at which to evaluate.
        episode_id: episode identifier.
        counterfactuals: list of dicts, each with keys:
            - name: str
            - description: str
            - transform: callable(BehaviorGraph) -> BehaviorGraph
            - expected_direction: "decrease" or "increase"

    Returns:
        List of CounterfactualResult.
    """
    prefix = trace[:step]

    # Build the original graph at this step.
    g_orig = BehaviorGraph(episode_id=episode_id)
    for ev in prefix:
        g_orig.ingest_event(ev, ev.step if ev.step else step)

    original_risk = float(graph_monitor.score_prefix(prefix, step, graph=g_orig))

    results = []
    for cf in counterfactuals:
        name = cf["name"]
        description = cf["description"]
        transform = cf["transform"]
        expected = cf.get("expected_direction", "decrease")

        g_cf = transform(g_orig)
        cf_risk = float(graph_monitor.score_prefix(prefix, step, graph=g_cf))

        delta = cf_risk - original_risk
        if expected == "decrease":
            direction_correct = delta < 0
        elif expected == "increase":
            direction_correct = delta > 0
        else:
            direction_correct = False

        results.append(CounterfactualResult(
            episode_id=episode_id,
            counterfactual_name=name,
            description=description,
            original_risk=original_risk,
            counterfactual_risk=cf_risk,
            risk_delta=delta,
            direction_correct=direction_correct,
            expected_direction=expected,
            step=step,
        ))

    return results


# ---------------------------------------------------------------------------
# Predefined counterfactual transforms
# ---------------------------------------------------------------------------

def remove_edge_type_transform(edge_type: EdgeType):
    """Return a transform that removes all edges of ``edge_type``."""

    def transform(graph: BehaviorGraph) -> BehaviorGraph:
        new_g = BehaviorGraph(episode_id=graph.episode_id)
        # Copy nodes.
        for node in graph.nodes.values():
            new_g.node(node_type=node.node_type, step=node.step,
                       attributes=dict(node.attributes))
        # Build old->new id mapping by (step, type) match.
        old_to_new: Dict[str, str] = {}
        for new_node in new_g.nodes.values():
            for old_node in graph.nodes.values():
                if (old_node.step == new_node.step
                        and old_node.node_type == new_node.node_type
                        and old_node.node_id not in old_to_new):
                    old_to_new[old_node.node_id] = new_node.node_id
                    break
        # Copy edges except the removed type.
        for edge in graph.edges.values():
            if edge.edge_type == edge_type:
                continue
            if edge.source_id not in old_to_new or edge.target_id not in old_to_new:
                continue
            new_g.edge(
                source_id=old_to_new[edge.source_id],
                target_id=old_to_new[edge.target_id],
                edge_type=edge.edge_type,
                step=edge.step,
                attributes=dict(edge.attributes),
            )
        return new_g

    return transform


def remove_sends_to_external_transform():
    """Remove sends_to edges whose target is an external effect node."""

    def transform(graph: BehaviorGraph) -> BehaviorGraph:
        new_g = BehaviorGraph(episode_id=graph.episode_id)
        for node in graph.nodes.values():
            new_g.node(node_type=node.node_type, step=node.step,
                       attributes=dict(node.attributes))
        old_to_new: Dict[str, str] = {}
        for new_node in new_g.nodes.values():
            for old_node in graph.nodes.values():
                if (old_node.step == new_node.step
                        and old_node.node_type == new_node.node_type
                        and old_node.node_id not in old_to_new):
                    old_to_new[old_node.node_id] = new_node.node_id
                    break
        for edge in graph.edges.values():
            if edge.edge_type == EdgeType.SENDS_TO:
                target = graph.nodes.get(edge.target_id)
                if target is not None and target.node_type == NodeType.EXTERNAL_EFFECT:
                    continue  # remove external sends
            if edge.source_id not in old_to_new or edge.target_id not in old_to_new:
                continue
            new_g.edge(
                source_id=old_to_new[edge.source_id],
                target_id=old_to_new[edge.target_id],
                edge_type=edge.edge_type,
                step=edge.step,
                attributes=dict(edge.attributes),
            )
        return new_g

    return transform


def sanitize_tool_output_transform():
    """Sanitize untrusted tool output: remove derived_from edges from
    tool_result nodes (simulating output sanitization)."""

    def transform(graph: BehaviorGraph) -> BehaviorGraph:
        new_g = BehaviorGraph(episode_id=graph.episode_id)
        for node in graph.nodes.values():
            new_g.node(node_type=node.node_type, step=node.step,
                       attributes=dict(node.attributes))
        old_to_new: Dict[str, str] = {}
        for new_node in new_g.nodes.values():
            for old_node in graph.nodes.values():
                if (old_node.step == new_node.step
                        and old_node.node_type == new_node.node_type
                        and old_node.node_id not in old_to_new):
                    old_to_new[old_node.node_id] = new_node.node_id
                    break
        for edge in graph.edges.values():
            if edge.edge_type == EdgeType.DERIVED_FROM:
                source = graph.nodes.get(edge.source_id)
                if source is not None and source.node_type == NodeType.TOOL_RESULT:
                    continue  # remove derived_from from tool results
            if edge.source_id not in old_to_new or edge.target_id not in old_to_new:
                continue
            new_g.edge(
                source_id=old_to_new[edge.source_id],
                target_id=old_to_new[edge.target_id],
                edge_type=edge.edge_type,
                step=edge.step,
                attributes=dict(edge.attributes),
            )
        return new_g

    return transform
