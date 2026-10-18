"""Deterministic replay of recorded episodes.

Replays a trace from the JSONL store and verifies that re-running the same
policy produces identical steps, decisions, and outcomes (fixed seed +
immutable manifest).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .orchestrator import Orchestrator
from .schema import EpisodeManifest
from .telemetry import Telemetry


def replay_episode(orchestrator: Orchestrator, manifest: EpisodeManifest,
                   policy, trace_store=None) -> Dict[str, Any]:
    """Re-run an episode and return the result.

    Determinism is guaranteed by:
      - fixed manifest.seed (policy can use it for randomness),
      - immutable scenario manifest,
      - same agent/gateway/sandbox instances with no external state.
    """
    return orchestrator.run_episode(manifest, policy)


def verify_determinism(orchestrator: Orchestrator, manifest: EpisodeManifest,
                       policy, runs: int = 3) -> Dict[str, Any]:
    """Run the same episode N times and confirm identical results.

    Returns a dict with 'deterministic' (bool) and 'results' (list).
    """
    results = []
    for _ in range(runs):
        result = replay_episode(orchestrator, manifest, policy)
        results.append(result)

    first = results[0]
    deterministic = all(
        r["task_success"] == first["task_success"]
        and r["policy_violation"] == first["policy_violation"]
        and r["unsafe_event_step"] == first["unsafe_event_step"]
        and r["total_steps"] == first["total_steps"]
        and r["intervention_step"] == first["intervention_step"]
        for r in results
    )
    return {"deterministic": deterministic, "results": results}
