"""Abstract base class for all GraphGuard monitors.

A monitor consumes a prefix of an episode's telemetry trace (and optionally
the partial behavior graph) and emits a risk score in [0, 1]. Monitors are
calibrated to alert thresholds at target false-positive rates (1% and 5%).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from harness.schema import TraceEvent


@dataclass
class MonitorOutput:
    """Result of scoring a single prefix."""
    monitor: str
    step: int
    score: float
    alert: bool
    fpr_level: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class BaseMonitor(ABC):
    """Abstract monitor interface."""

    def __init__(self, name: str):
        self.name = name
        # fpr_level -> score threshold
        self.thresholds: Dict[float, float] = {}
        self._calibrated = False

    # ---- Scoring ---------------------------------------------------------

    @abstractmethod
    def score_prefix(
        self,
        trace_prefix: List[TraceEvent],
        step: int,
        graph=None,
    ) -> float:
        """Return a risk score in [0, 1] for the prefix up to ``step``.

        Args:
            trace_prefix: telemetry events for steps 1..step (inclusive).
            step: current prefix step (1-based).
            graph: partial behavior graph built up to this step (optional).
        """
        ...

    def score(self, trace_prefix: List[TraceEvent], step: int,
              graph=None) -> MonitorOutput:
        """Score a prefix and return a full :class:`MonitorOutput`.

        Applies calibration thresholds if available.
        """
        score = float(self.score_prefix(trace_prefix, step, graph=graph))
        score = max(0.0, min(1.0, score))
        alert, fpr_level = self._alert_decision(score)
        return MonitorOutput(
            monitor=self.name,
            step=step,
            score=score,
            alert=alert,
            fpr_level=fpr_level,
        )

    # ---- Calibration -----------------------------------------------------

    def calibrate(
        self,
        benign_traces: List[Tuple[List[TraceEvent], int]],
        target_fprs: List[float] = (0.01, 0.05),
    ) -> Dict[float, float]:
        """Calibrate alert thresholds on benign traces.

        For each target FPR, collects all prefix scores from benign episodes
        and sets the threshold at the (1 - FPR) quantile, so that roughly
        ``FPR`` fraction of benign prefixes exceed it.

        Args:
            benign_traces: list of (trace_events, total_steps) for benign
                (safe) validation episodes.
            target_fprs: false-positive rates to calibrate for.

        Returns:
            dict mapping fpr_level -> score threshold.
        """
        all_scores: List[float] = []
        for trace, total_steps in benign_traces:
            for step in range(1, total_steps + 1):
                prefix = trace[:step]
                score = float(self.score_prefix(prefix, step))
                all_scores.append(score)

        if not all_scores:
            raise ValueError("No benign prefix scores to calibrate on.")

        all_scores.sort()
        n = len(all_scores)

        thresholds: Dict[float, float] = {}
        for fpr in target_fprs:
            # Index of the (1 - FPR) quantile (ceiling), clamped.
            idx = max(0, min(n - 1, int((1.0 - fpr) * n) - 1))
            thresholds[fpr] = all_scores[idx]

        self.thresholds = thresholds
        self._calibrated = True
        return dict(thresholds)

    def _alert_decision(self, score: float) -> Tuple[bool, Optional[float]]:
        """Return (alert, fpr_level) for a score using the tightest
        calibrated threshold it exceeds."""
        if not self._calibrated:
            return False, None
        # Check from tightest (lowest FPR) to loosest; alert at the first
        # threshold exceeded so we report the most stringent level.
        for fpr in sorted(self.thresholds.keys()):
            if score >= self.thresholds[fpr]:
                return True, fpr
        return False, None

    def is_alert(self, score: float, fpr_level: float = 0.05) -> bool:
        """Return True if ``score`` exceeds the calibrated threshold for
        ``fpr_level``. Returns False if not calibrated."""
        if not self._calibrated or fpr_level not in self.thresholds:
            return False
        return score >= self.thresholds[fpr_level]

    def threshold_for(self, fpr_level: float) -> Optional[float]:
        return self.thresholds.get(fpr_level)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r}, calibrated={self._calibrated})"
