"""Strict episode splitting by attack template and wording variant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class EpisodeSplits:
    train: List[Any]
    validation: List[Any]
    test: List[Any]


def _episode_metadata(episode: Any) -> Tuple[str, str, str]:
    if isinstance(episode, Mapping):
        labels = episode.get("labels", {})
        manifest = episode.get("manifest", {})
        if hasattr(manifest, "labels"):
            labels = manifest.labels
        elif isinstance(manifest, Mapping):
            labels = manifest.get("labels", labels)
        episode_id = episode.get("episode_id")
        if episode_id is None and hasattr(manifest, "episode_id"):
            episode_id = manifest.episode_id
        elif episode_id is None and isinstance(manifest, Mapping):
            episode_id = manifest.get("episode_id")
        template_id = episode.get("template_id") or labels.get("attack_template_id")
        wording = episode.get("wording_variant") or labels.get("wording_variant")
    else:
        manifest = getattr(episode, "manifest", None)
        labels = getattr(manifest, "labels", {})
        episode_id = getattr(manifest, "episode_id", None)
        template_id = labels.get("attack_template_id")
        wording = labels.get("wording_variant")

    if not episode_id or not template_id or not wording:
        raise ValueError(
            "Each episode needs episode_id, attack_template_id, and wording_variant."
        )
    return str(episode_id), str(template_id), str(wording)


def split_by_template(
    episodes: Sequence[Any],
    train_templates: Iterable[str],
    validation_templates: Iterable[str],
    test_templates: Iterable[str],
) -> EpisodeSplits:
    """Split episodes using explicit, disjoint template assignments.

    A wording variant is assigned to exactly one split as well, preventing
    paraphrase variants from crossing split boundaries.
    """
    template_splits: Dict[str, str] = {}
    for split_name, templates in (
        ("train", train_templates),
        ("validation", validation_templates),
        ("test", test_templates),
    ):
        for template in templates:
            if template in template_splits:
                raise ValueError(f"Template {template!r} occurs in multiple splits.")
            template_splits[template] = split_name

    if set(template_splits.values()) != {"train", "validation", "test"}:
        raise ValueError("Train, validation, and test each need at least one template.")

    result: Dict[str, List[Any]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    ids_by_split: Dict[str, set[str]] = {name: set() for name in result}
    wordings_by_split: Dict[str, set[str]] = {name: set() for name in result}
    for episode in episodes:
        episode_id, template_id, wording = _episode_metadata(episode)
        split_name = template_splits.get(template_id)
        if split_name is None:
            raise ValueError(f"Template {template_id!r} has no declared split.")
        if episode_id in ids_by_split[split_name]:
            raise ValueError(f"Duplicate episode id {episode_id!r}.")
        ids_by_split[split_name].add(episode_id)
        wordings_by_split[split_name].add(wording)
        result[split_name].append(episode)

    if any(not result[name] for name in result):
        raise ValueError("Every split must contain at least one episode.")
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        if wordings_by_split[left] & wordings_by_split[right]:
            raise ValueError(
                f"Wording variants overlap between {left} and {right}."
            )
    return EpisodeSplits(**result)
