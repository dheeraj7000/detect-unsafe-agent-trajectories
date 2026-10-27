"""Transformer encoder baseline over serialized telemetry events."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from harness.schema import TraceEvent

from .base import BaseMonitor, MonitorOutput

_TOKEN_PATTERN = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)
_PAD = "<pad>"
_UNK = "<unk>"


class _SequenceTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.position_embed = nn.Embedding(max_length, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = token_ids.eq(0)
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embed(token_ids) + self.position_embed(positions)[None]
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.classifier(pooled).squeeze(-1)


class TransformerSequenceMonitor(BaseMonitor):
    """Small Transformer sequence classifier for full trajectory prefixes."""

    def __init__(
        self,
        name: str = "transformer",
        max_length: int = 512,
        max_vocab: int = 4096,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        super().__init__(name)
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if max_vocab < 2:
            raise ValueError("max_vocab must be at least 2")
        if model_dim < 2 or model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")

        self.max_length = max_length
        self.max_vocab = max_vocab
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocabulary = {_PAD: 0, _UNK: 1}
        self.model: Optional[_SequenceTransformer] = None
        self._fitted = False

    @staticmethod
    def _tokens(trace: List[TraceEvent]) -> List[str]:
        tokens: List[str] = []
        for event in trace:
            payload = json.dumps(event.payload or {}, sort_keys=True, default=str)
            serialized = (
                f"event {event.event_type.value} actor {event.actor} "
                f"step {event.step} payload {payload}"
            ).lower()
            tokens.extend(_TOKEN_PATTERN.findall(serialized))
        return tokens

    def _encode(self, trace: List[TraceEvent]) -> List[int]:
        token_ids = [
            self.vocabulary.get(token, self.vocabulary[_UNK])
            for token in self._tokens(trace)
        ][:self.max_length]
        return token_ids or [self.vocabulary[_UNK]]

    def _batch(self, traces: List[List[TraceEvent]]) -> torch.Tensor:
        rows = [self._encode(trace) for trace in traces]
        length = min(self.max_length, max(map(len, rows)))
        result = torch.zeros((len(rows), length), dtype=torch.long)
        for index, row in enumerate(rows):
            truncated = row[:length]
            result[index, :len(truncated)] = torch.tensor(truncated)
        return result

    def fit(
        self,
        examples: List[Tuple[List[TraceEvent], int, bool]],
    ) -> "TransformerSequenceMonitor":
        if not examples:
            raise ValueError("No examples provided for fitting.")

        token_counts: Counter[str] = Counter()
        for trace, _, _ in examples:
            token_counts.update(self._tokens(trace))
        most_common = sorted(
            token_counts.items(), key=lambda item: (-item[1], item[0])
        )[:self.max_vocab - 2]
        self.vocabulary = {_PAD: 0, _UNK: 1}
        self.vocabulary.update(
            {token: index + 2 for index, (token, _) in enumerate(most_common)}
        )

        torch.manual_seed(self.seed)
        self.model = _SequenceTransformer(
            vocab_size=len(self.vocabulary),
            max_length=self.max_length,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
        ).to(self.device)

        labels = torch.tensor(
            [1.0 if label else 0.0 for _, _, label in examples],
            dtype=torch.float32,
        )
        encoded = [self._encode(trace) for trace, _, _ in examples]
        positives = float(labels.sum())
        negatives = float(len(labels) - positives)
        pos_weight = torch.tensor(
            [negatives / positives if positives else 1.0],
            dtype=torch.float32,
            device=self.device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        rng = np.random.RandomState(self.seed)

        self.model.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(examples))
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start + self.batch_size]
                batch_traces = [examples[index][0] for index in indices]
                token_ids = self._batch(batch_traces).to(self.device)
                batch_labels = labels[indices].to(self.device)
                optimizer.zero_grad()
                logits = self.model(token_ids)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

        self.model.eval()
        self._fitted = True
        return self

    def score_prefix(
        self,
        trace_prefix: List[TraceEvent],
        step: int,
        graph=None,
    ) -> float:
        if not self._fitted or self.model is None:
            return min(1.0, 0.02 * len(trace_prefix))
        token_ids = self._batch([trace_prefix]).to(self.device)
        with torch.no_grad():
            return float(torch.sigmoid(self.model(token_ids)).item())

    def score(
        self,
        trace_prefix: List[TraceEvent],
        step: int,
        graph=None,
    ) -> MonitorOutput:
        output = super().score(trace_prefix, step, graph)
        output.details = {
            "prefix_events": len(trace_prefix),
            "tokens": len(self._tokens(trace_prefix)),
            "parameters": sum(p.numel() for p in self.model.parameters())
            if self.model is not None else 0,
        }
        return output
