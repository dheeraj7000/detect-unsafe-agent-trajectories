"""Dataset store: persist and load generated episodes (JSONL + Parquet)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, List, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PARQUET = True
except ImportError:  # pragma: no cover
    _HAS_PARQUET = False

from .generator import GeneratedEpisode


class DatasetStore:
    """Append-only store for generated episodes."""

    def __init__(self, base_dir: str = "data/episodes"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.jsonl"

    def _episode_path(self, episode_id: str) -> Path:
        return self.base_dir / f"{episode_id}.json"

    def save(self, episode: GeneratedEpisode) -> None:
        path = self._episode_path(episode.manifest.episode_id)
        payload = {
            "manifest": episode.manifest.model_dump(),
            "result": episode.result,
            "label": episode.label,
            "trace_events": episode.trace_events,
            "graph_nodes": episode.graph_nodes,
            "graph_edges": episode.graph_edges,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))

        # Append to index.
        with self.index_path.open("a") as f:
            f.write(json.dumps({
                "episode_id": episode.manifest.episode_id,
                "task_family": episode.manifest.task_family,
                "attack_family": episode.manifest.attack_family,
                "policy_violation": episode.result.get("policy_violation"),
                "unsafe_event_step": episode.result.get("unsafe_event_step"),
                "total_steps": episode.result.get("total_steps"),
                "path": str(path),
            }, default=str) + "\n")

    def load(self, episode_id: str) -> Optional[dict]:
        path = self._episode_path(episode_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def iter_index(self) -> Iterator[dict]:
        if not self.index_path.exists():
            return iter([])
        with self.index_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def list_episodes(self) -> List[dict]:
        return list(self.iter_index())

    def export_parquet(self, out_path: str) -> None:
        if not _HAS_PARQUET:
            raise ImportError("pyarrow required for Parquet export")
        rows = []
        for ep in self.list_episodes():
            detail = self.load(ep["episode_id"])
            if detail:
                rows.append({
                    "episode_id": ep["episode_id"],
                    "task_family": ep["task_family"],
                    "attack_family": ep["attack_family"],
                    "policy_violation": ep["policy_violation"],
                    "unsafe_event_step": ep["unsafe_event_step"],
                    "total_steps": ep["total_steps"],
                    "task_success": detail["result"]["task_success"],
                    "label": detail["label"],
                })
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_path)


def save_episode(episode: GeneratedEpisode, base_dir: str = "data/episodes") -> None:
    store = DatasetStore(base_dir)
    store.save(episode)


def load_episode(episode_id: str, base_dir: str = "data/episodes") -> Optional[dict]:
    store = DatasetStore(base_dir)
    return store.load(episode_id)
