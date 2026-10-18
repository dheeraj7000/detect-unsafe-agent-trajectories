"""Graph monitor — the proposed GraphGuard method.

Scores risk on the partial typed behavior graph using a temporal GNN
(monitors.graph_model.TemporalGraphNN). Unlike the sequence baselines, the
graph monitor sees explicit provenance / authorization / dependency edges,
so it can detect relational risk patterns (e.g. untrusted tool output ->
plan update -> credential read -> external send) earlier than a flat
sequence model.

Implements the BaseMonitor interface so it can be calibrated and evaluated
apples-to-apples against the baselines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.graph import BehaviorGraph, graph_to_tensors, NODE_TYPE_LIST, EDGE_TYPE_LIST
from .base import BaseMonitor, MonitorOutput
from .graph_model import TemporalGraphNN, build_model


class GraphMonitor(BaseMonitor):
    """Graph-based monitor: temporal GNN over the partial behavior graph.

    Args:
        name: monitor name.
        hidden: GNN hidden dimension.
        num_layers: number of message-passing layers.
        lr: learning rate for fitting.
        epochs: training epochs for fitting.
        batch_size: training batch size.
        max_steps: max episode length (for step normalization).
        device: torch device.
    """

    def __init__(
        self,
        name: str = "graph",
        hidden: int = 64,
        num_layers: int = 3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        max_steps: int = 500,
        device: Optional[str] = None,
    ):
        super().__init__(name)
        self.hidden = hidden
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Determine feature dims from schema.
        node_feat_dim = len(NODE_TYPE_LIST) + 1  # onehot + step_norm
        edge_feat_dim = len(EDGE_TYPE_LIST) + 1  # onehot + step_norm

        self.model: TemporalGraphNN = build_model(node_feat_dim, edge_feat_dim)
        self.model.to(self.device)
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit / train
    # ------------------------------------------------------------------

    def fit(
        self,
        examples: List[Tuple[List[Any], int, Optional[BehaviorGraph], bool]],
    ) -> "GraphMonitor":
        """Train the GNN on labeled prefix examples.

        Each example: (trace_prefix, step, graph_at_step, is_unsafe_label).
        The graph may be None; in that case an empty graph is used.
        """
        xs, edge_indices, edge_attrs, ys = [], [], [], []

        for trace_prefix, step, graph, label in examples:
            g = graph if graph is not None else BehaviorGraph(episode_id="fit")
            x, edge_index, edge_attr, _ = graph_to_tensors(
                g, max_steps=self.max_steps
            )
            xs.append(x)
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
            ys.append(1.0 if label else 0.0)

        if not xs:
            raise ValueError("No examples provided for fitting.")

        y_tensor = torch.tensor(ys, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.model.train()
        for _ in range(self.epochs):
            total_loss = 0.0
            # Graphs vary in size, so train one example at a time.
            for i in range(len(xs)):
                xi = xs[i].to(self.device)
                eii = edge_indices[i].to(self.device)
                eai = edge_attrs[i].to(self.device)
                yi = y_tensor[i:i + 1].to(self.device)

                optimizer.zero_grad()
                risk = self.model(xi, eii, eai)  # shape (1,)
                loss = criterion(risk, yi)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score_prefix(
        self,
        trace_prefix: List[Any],
        step: int,
        graph: Optional[BehaviorGraph] = None,
    ) -> float:
        """Score the prefix using the GNN on the partial behavior graph."""
        g = graph if graph is not None else BehaviorGraph(episode_id="score")
        x, edge_index, edge_attr, _ = graph_to_tensors(
            g, max_steps=self.max_steps
        )

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        self.model.eval()
        with torch.no_grad():
            risk = self.model(x, edge_index, edge_attr)
        return float(risk.item())

    def score(self, trace_prefix: List[Any], step: int,
              graph: Optional[BehaviorGraph] = None) -> MonitorOutput:
        out = super().score(trace_prefix, step, graph)
        # Record graph size as a detail (useful for overhead analysis).
        g = graph if graph is not None else BehaviorGraph(episode_id="score")
        out.details = {
            "num_nodes": len(g.nodes),
            "num_edges": len(g.edges),
        }
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return (f"{self.__class__.__name__}(name={self.name!r}, "
                f"fitted={self._fitted}, device={self.device})")
