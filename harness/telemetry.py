"""Telemetry layer: records trace events and persists them to JSONL/Parquet.

Implements an OpenTelemetry-style event store. Each episode appends events to
a JSONL file; the full trace can be replayed deterministically from the file.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PARQUET = True
except ImportError:  # pragma: no cover - parquet optional
    _HAS_PARQUET = False

from .schema import EpisodeManifest, TraceEvent


class TraceStore:
    """Append-only trace store backed by JSONL (with optional Parquet export)."""

    def __init__(self, base_dir: str | Path = "data/traces"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._current_file: Optional[Path] = None
        self._episode_id: Optional[str] = None

    def _path_for(self, episode_id: str) -> Path:
        return self.base_dir / f"{episode_id}.jsonl"

    def start_episode(self, episode_id: str) -> None:
        self._episode_id = episode_id
        self._current_file = self._path_for(episode_id)
        # Truncate any prior trace for deterministic replay idempotency.
        self._current_file.write_text("")

    def append(self, event: TraceEvent) -> None:
        if self._current_file is None:
            raise RuntimeError("No episode started. Call start_episode() first.")
        line = event.model_dump_json()
        with self._current_file.open("a") as f:
            f.write(line + "\n")

    def iter_events(self, episode_id: str) -> Iterator[TraceEvent]:
        path = self._path_for(episode_id)
        if not path.exists():
            return iter([])
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield TraceEvent.model_validate_json(line)

    def load_episode(self, episode_id: str) -> List[TraceEvent]:
        return list(self.iter_events(episode_id))

    def export_parquet(self, episode_id: str, out_path: str | Path) -> None:
        if not _HAS_PARQUET:
            raise ImportError("pyarrow is required for Parquet export")
        events = [e.model_dump() for e in self.iter_events(episode_id)]
        if not events:
            # Write an empty but typed table.
            table = pa.table({})
        else:
            table = pa.Table.from_pylist(events)
        pq.write_table(table, out_path)


class Telemetry:
    """High-level telemetry facade used by the harness orchestrator."""

    def __init__(self, base_dir: str | Path = "data/traces"):
        self.store = TraceStore(base_dir)
        self._step: int = 0
        self._episode_id: Optional[str] = None

    def begin_episode(self, episode_id: str) -> None:
        self._episode_id = episode_id
        self._step = 0
        self.store.start_episode(episode_id)
        self.record(
            event_type="episode_start",
            actor="orchestrator",
            payload={"episode_id": episode_id},
        )

    def next_step(self) -> int:
        self._step += 1
        return self._step

    @property
    def step(self) -> int:
        return self._step

    def record(
        self,
        event_type: str,
        actor: str,
        payload: dict,
        parent_event_id: Optional[str] = None,
        attributes: Optional[dict] = None,
    ) -> TraceEvent:
        if self._episode_id is None:
            raise RuntimeError("No episode started. Call begin_episode() first.")
        event = TraceEvent(
            episode_id=self._episode_id,
            step=self._step,
            event_type=event_type,  # type: ignore[arg-type]
            actor=actor,
            payload=payload,
            parent_event_id=parent_event_id,
            attributes=attributes or {},
        )
        self.store.append(event)
        return event

    def end_episode(self, result: dict) -> None:
        if self._episode_id is None:
            raise RuntimeError("No episode started.")
        self.record(
            event_type="episode_end",
            actor="orchestrator",
            payload=result,
        )

    def load_trace(self, episode_id: str) -> List[TraceEvent]:
        return self.store.load_episode(episode_id)
