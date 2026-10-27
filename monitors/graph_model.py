"""Temporal graph neural network for GraphGuard.

A pure-PyTorch message-passing GNN that scores risk on a partial behavior
graph. Designed to run without PyTorch Geometric / DGL (an optional PyG path
can be added later).

Architecture:
  1. Node embedding: node_type embedding + step positional encoding
  2. Edge embedding: edge_type embedding + projected edge type/time features
  3. K rounds of typed message passing (aggregate neighbor messages,
     weighted by edge type)
  4. Global mean-pooling readout over all nodes
  5. MLP -> sigmoid risk score in [0, 1]

The model is "temporal" because node/edge step information is folded into
the embeddings, so the GNN is aware of when each element appeared in the
episode.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.graph import (
    EDGE_TYPE_LIST,
    EDGE_TYPE_TO_IDX,
    NODE_TYPE_LIST,
    NODE_TYPE_TO_IDX,
    graph_to_tensors,
)


class TypedMessagePassing(nn.Module):
    """One round of typed message passing.

    For each edge (s -> t) with type r, the message is:
        m_{s->t} = Linear(concat(h_s, edge_embedding))
    Node t aggregates incoming messages by sum, then updates:
        h_t' = GRU(h_t, sum_incoming)  or  MLP(concat(h_t, agg))
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64):
        super().__init__()
        self.node_dim = node_dim
        self.hidden = hidden

        self.msg_linear = nn.Linear(node_dim + edge_dim, hidden)

        # Update: combine current node state with aggregated messages.
        self.update_gru = nn.GRUCell(hidden, node_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            h: (num_nodes, node_dim)
            edge_index: (2, num_edges)
            edge_emb: (num_edges, edge_dim), including edge type and time.
        Returns:
            h_updated: (num_nodes, node_dim)
        """
        num_nodes = h.size(0)
        if edge_index.size(1) == 0:
            return h

        src = edge_index[0]
        dst = edge_index[1]

        msg = self.msg_linear(torch.cat([h[src], edge_emb], dim=1))

        # Aggregate by sum into destination nodes.
        agg = torch.zeros(num_nodes, self.hidden, device=h.device,
                          dtype=h.dtype)
        agg.index_add_(0, dst, msg)

        # GRU update: treat aggregated message as input, h as hidden state.
        h_updated = self.update_gru(agg, h)
        return h_updated


class TemporalGraphNN(nn.Module):
    """Temporal graph neural network for behavior-graph risk scoring."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        if hidden < 2 or hidden % 2:
            raise ValueError("hidden must be an even integer greater than or equal to 2")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        # Node-type and edge-type embeddings.
        self.node_type_embed = nn.Embedding(len(NODE_TYPE_LIST), hidden // 2)
        self.edge_type_embed = nn.Embedding(len(EDGE_TYPE_LIST), hidden // 2)

        # Project raw node features (onehot + step) into hidden space.
        self.node_in_proj = nn.Linear(node_feat_dim, hidden // 2)
        self.edge_in_proj = nn.Linear(edge_feat_dim, hidden // 2)

        # Message passing layers.
        self.mp_layers = nn.ModuleList([
            TypedMessagePassing(hidden, hidden, hidden=hidden)
            for _ in range(num_layers)
        ])

        # Readout + risk head.
        self.readout = nn.Linear(hidden, hidden)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            x: (num_nodes, node_feat_dim)
            edge_index: (2, num_edges)
            edge_attr: (num_edges, edge_feat_dim)
        Returns:
            risk: (1,) scalar risk score in [0, 1]
        """
        num_nodes = x.size(0)
        if num_nodes == 0:
            return torch.tensor([0.5], device=x.device)

        # ---- Node embeddings -------------------------------------------
        # x columns: [node_type_onehot (13), step_norm (1)]
        node_type_onehot = x[:, : len(NODE_TYPE_LIST)]

        node_type_idx = node_type_onehot.argmax(dim=1)
        h = torch.cat([
            self.node_type_embed(node_type_idx),
            self.node_in_proj(x),
        ], dim=1)  # (num_nodes, hidden)

        # ---- Edge embeddings -------------------------------------------
        if edge_index.size(1) > 0:
            edge_type_onehot = edge_attr[:, : len(EDGE_TYPE_LIST)]
            edge_type_idx = edge_type_onehot.argmax(dim=1)
            edge_emb = torch.cat([
                self.edge_type_embed(edge_type_idx),
                self.edge_in_proj(edge_attr),
            ], dim=1)  # (num_edges, hidden)

        # ---- Message passing -------------------------------------------
        for mp in self.mp_layers:
            if edge_index.size(1) > 0:
                h = mp(h, edge_index, edge_emb)

        # ---- Readout: global mean pooling -------------------------------
        graph_emb = h.mean(dim=0, keepdim=True)  # (1, hidden)
        graph_emb = F.relu(self.readout(graph_emb))

        # ---- Risk head --------------------------------------------------
        risk = self.risk_head(graph_emb)  # (1, 1)
        return risk.squeeze(0)  # (1,)


def build_model(
    node_feat_dim: int,
    edge_feat_dim: int,
    hidden: int = 64,
    num_layers: int = 3,
) -> TemporalGraphNN:
    """Factory: build a TemporalGraphNN with configurable architecture."""
    return TemporalGraphNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden=hidden,
        num_layers=num_layers,
    )
