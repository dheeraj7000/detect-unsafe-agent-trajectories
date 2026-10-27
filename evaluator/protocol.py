"""Immutable experiment protocol and split-manifest helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _freeze_payload(
    protocol: Mapping[str, Any],
    split_episode_ids: Mapping[str, Sequence[str]],
) -> dict:
    required = {"train", "validation", "test"}
    if set(split_episode_ids) != required:
        raise ValueError("Splits must contain train, validation, and test IDs.")
    normalized_splits = {
        split: sorted(str(episode_id) for episode_id in split_episode_ids[split])
        for split in sorted(required)
    }
    if any(not ids for ids in normalized_splits.values()):
        raise ValueError("Every split must contain at least one episode ID.")
    all_ids = [episode_id for ids in normalized_splits.values() for episode_id in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Episode IDs must be disjoint across all splits.")
    return {
        "protocol": json.loads(json.dumps(protocol, sort_keys=True, default=str)),
        "splits": normalized_splits,
    }


def freeze_experiment(
    path: str | Path,
    protocol: Mapping[str, Any],
    split_episode_ids: Mapping[str, Sequence[str]],
) -> str:
    """Write a pre-evaluation protocol manifest that cannot be silently changed."""
    payload = _freeze_payload(protocol, split_episode_ids)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    document = {**payload, "sha256": digest}
    destination = Path(path)

    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing != document:
            raise ValueError(
                f"Frozen experiment manifest already exists and differs: {destination}"
            )
        return digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return digest


def verify_frozen_experiment(
    path: str | Path,
    protocol: Mapping[str, Any],
    split_episode_ids: Mapping[str, Sequence[str]],
) -> None:
    """Raise if the supplied protocol or split differs from its frozen manifest."""
    expected = _freeze_payload(protocol, split_episode_ids)
    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    document = json.loads(Path(path).read_text())
    stored_payload = {
        "protocol": document.get("protocol"),
        "splits": document.get("splits"),
    }
    stored_canonical = json.dumps(
        stored_payload, sort_keys=True, separators=(",", ":")
    )
    stored_digest = hashlib.sha256(stored_canonical.encode("utf-8")).hexdigest()
    if document.get("sha256") != stored_digest or stored_digest != digest:
        raise ValueError("Experiment protocol or episode split differs from frozen manifest.")
