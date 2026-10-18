"""Sequence classifier — primary ordinary-trajectory baseline.

Serializes the recent (or full) trajectory prefix into a compact text string,
vectorizes it with TF-IDF, and scores it with scikit-learn
:class:`LogisticRegression`. This is the "flat sequence" model that must infer
dependencies across distance rather than reading them from a graph.

Fit on labeled validation data via :meth:`fit`, then score prefixes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from harness.schema import EventType, TraceEvent

from .base import BaseMonitor, MonitorOutput


def _serialize_prefix(trace_prefix: List[TraceEvent], step: int) -> str:
    """Flatten a trace prefix into a single text string (event sequence)."""
    parts: List[str] = []
    for ev in trace_prefix:
        payload = ev.payload or {}
        tool = payload.get("tool", "")
        decision = payload.get("decision", "")
        success = payload.get("success", "")
        parts.append(
            f"{ev.step}|{ev.event_type.value}|actor={ev.actor}|"
            f"tool={tool}|decision={decision}|success={success}|"
            f"payload={payload}"
        )
    return " ||| ".join(parts)


class SequenceMonitor(BaseMonitor):
    """Ordinary-trajectory baseline: TF-IDF over serialized prefix + LR."""

    def __init__(self, name: str = "sequence", window: Optional[int] = None):
        super().__init__(name)
        # window = number of recent events to include (None = full prefix)
        self.window = window
        self._pipe = None
        self._fitted = False

    def _window(self, trace_prefix: List[TraceEvent]) -> List[TraceEvent]:
        if self.window is None or self.window <= 0:
            return list(trace_prefix)
        return list(trace_prefix[-self.window:])

    def _texts(self, examples: List[Tuple[List[TraceEvent], int]]) -> List[str]:
        return [_serialize_prefix(self._window(trace), step)
                for trace, step in examples]

    def fit(
        self,
        examples: List[Tuple[List[TraceEvent], int, bool]],
    ) -> "SequenceMonitor":
        """Train TF-IDF + logistic regression on labeled prefix examples."""
        texts = self._texts([(t, s) for t, s, _ in examples])
        y = np.array([1.0 if label else 0.0 for _, _, label in examples])

        self._pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                            max_features=5000),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        self._pipe.fit(texts, y)
        self._fitted = True
        return self

    def score_prefix(
        self,
        trace_prefix: List[TraceEvent],
        step: int,
        graph=None,
    ) -> float:
        if not self._fitted:
            # Unfitted heuristic: longer prefixes with policy events = riskier.
            n = len(trace_prefix)
            n_policy = sum(1 for ev in trace_prefix
                           if ev.event_type == EventType.POLICY_DECISION)
            return min(1.0, 0.05 * n_policy + 0.001 * n)

        text = _serialize_prefix(self._window(trace_prefix), step)
        proba = float(self._pipe.predict_proba([text])[0, 1])
        return proba

    def score(self, trace_prefix: List[TraceEvent], step: int,
              graph=None) -> MonitorOutput:
        out = super().score(trace_prefix, step, graph)
        out.details = {
            "prefix_length": len(trace_prefix),
            "window": self.window,
        }
        return out
