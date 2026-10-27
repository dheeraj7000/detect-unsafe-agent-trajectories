"""Temporal sequence model — strong long-context baseline.

Embeds each telemetry event into a fixed vector and processes the sequence
over time. Falls back to a scikit-learn MLP over windowed event embeddings
when PyTorch is unavailable; uses a small 1D-CNN/MLP via PyTorch when
available.

This is a *sequence* model: it sees events in temporal order but has no
explicit graph structure (no provenance / authorization edges), so it must
learn relational dependencies implicitly from the serialized sequence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from harness.schema import EventType, TraceEvent

from .base import BaseMonitor, MonitorOutput

try:  # pragma: no cover - torch optional
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False

# Fixed embedding dimension for event types + key attributes.
EMBED_DIM = 32

# Hash-based stable event embedding (offline, no LLM API needed).
_EVENT_TYPE_VECTORS: Dict[str, np.ndarray] = {}


def _event_vector(ev: TraceEvent) -> np.ndarray:
    """Deterministic embedding for a single event (offline, no external API)."""
    et = ev.event_type.value
    if et not in _EVENT_TYPE_VECTORS:
        # Stable pseudo-random vector per event type.
        rng = np.random.RandomState(abs(hash(et)) % (2 ** 31))
        _EVENT_TYPE_VECTORS[et] = rng.normal(0, 1, EMBED_DIM)
    base = _EVENT_TYPE_VECTORS[et].copy()
    payload = ev.payload or {}
    v = base.astype(float)

    # Modulate by a few salient payload features so the embedding carries
    # some content signal (tool name, decision, success).
    tool = str(payload.get("tool", ""))
    decision = str(payload.get("decision", ""))
    success = str(payload.get("success", ""))

    for i, ch in enumerate((tool + "|" + decision + "|" + success)[:8]):
        v[i % EMBED_DIM] += 0.1 * (ord(ch) % 7 - 3)

    # L2 normalize for stability.
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


class _TemporalMLP(nn.Module):  # type: ignore[misc]
    """Small PyTorch MLP over a fixed-length window of event embeddings."""

    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


class TemporalSequenceMonitor(BaseMonitor):
    """Long-context baseline: event embeddings processed over time."""

    def __init__(
        self,
        name: str = "temporal",
        window: int = 10,
        use_torch: bool = False,
        seed: int = 42,
    ):
        super().__init__(name)
        self.window = window
        self.use_torch = use_torch and _HAS_TORCH
        self.seed = seed
        self._sklearn: Optional[MLPClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._torch_model: Optional[_TemporalMLP] = None
        self._fitted = False

    def _window_vectors(self, trace_prefix: List[TraceEvent]) -> np.ndarray:
        """Return a fixed-length (window) matrix of event vectors.

        Shorter prefixes are left-padded with zeros; longer prefixes keep
        the most recent ``window`` events.
        """
        win = trace_prefix[-self.window:] if self.window > 0 else trace_prefix
        vecs = [_event_vector(ev) for ev in win]
        if len(vecs) < self.window:
            pad = [np.zeros(EMBED_DIM)] * (self.window - len(vecs))
            vecs = pad + vecs
        return np.stack(vecs, axis=0).reshape(-1)  # flattened window

    def _to_features(self, examples: List[Tuple[List[TraceEvent], int]]) -> np.ndarray:
        return np.vstack([self._window_vectors(trace) for trace, _ in examples])

    def fit(
        self,
        examples: List[Tuple[List[TraceEvent], int, bool]],
    ) -> "TemporalSequenceMonitor":
        """Train on labeled prefix examples.

        Examples: (trace_prefix, step, is_unsafe_label).
        """
        X = self._to_features([(t, s) for t, s, _ in examples])
        y = np.array([1.0 if label else 0.0 for _, _, label in examples])

        if self.use_torch:
            self._fit_torch(X, y)
        else:
            self._scaler = StandardScaler()
            Xs = self._scaler.fit_transform(X)
            self._sklearn = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=2000,
                random_state=self.seed,
            )
            self._sklearn.fit(Xs, y)

        self._fitted = True
        return self

    def _fit_torch(self, X: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        model = _TemporalMLP(input_dim=X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        model.train()
        for _ in range(300):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        self._torch_model = model

    def score_prefix(
        self,
        trace_prefix: List[TraceEvent],
        step: int,
        graph=None,
    ) -> float:
        if not self._fitted:
            # Unfitted heuristic: risk grows with policy events in window.
            win = trace_prefix[-self.window:] if self.window > 0 else trace_prefix
            n_deny = sum(1 for ev in win
                         if ev.event_type == EventType.POLICY_DECISION
                         and (ev.payload or {}).get("decision") == "deny")
            return min(1.0, 0.5 * n_deny + 0.02 * len(win))

        x = self._window_vectors(trace_prefix).reshape(1, -1)

        if self.use_torch and self._torch_model is not None:
            with torch.no_grad():
                xt = torch.tensor(x, dtype=torch.float32)
                return float(self._torch_model(xt).item())

        assert self._sklearn is not None and self._scaler is not None
        xs = self._scaler.transform(x)
        return float(self._sklearn.predict_proba(xs)[0, 1])

    def score(self, trace_prefix: List[TraceEvent], step: int,
              graph=None) -> MonitorOutput:
        out = super().score(trace_prefix, step, graph)
        out.details = {
            "window": self.window,
            "use_torch": self.use_torch,
            "prefix_length": len(trace_prefix),
        }
        return out
