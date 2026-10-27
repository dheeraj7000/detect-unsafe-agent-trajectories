"""Tests for Phase 2 dataset generation, labeling, and storage."""

from __future__ import annotations

import tempfile

import pytest

from harness import (Agent, Gateway, HarnessConfig, Orchestrator, Sandbox,
                     Telemetry, make_default_gateway, verify_determinism)
from dataset import EpisodeGenerator, GenerationConfig, DatasetStore
from dataset.splits import split_by_template
from dataset.synthetic import ALL_PAIRS
from dataset.labeler import Labeler


def test_generate_benign_episode():
    gen = EpisodeGenerator(make_default_gateway, Sandbox(),
                           base_dir=tempfile.mkdtemp())
    ep = gen.generate(GenerationConfig(
        task_family="file_backup", seed=1, adversarial=False, max_steps=50
    ))
    assert ep.label["policy_violation"] is False
    assert ep.label["attack_family"] is None
    assert ep.result["total_steps"] > 0
    assert ep.result["total_steps"] < 50


def test_generate_adversarial_episode():
    gen = EpisodeGenerator(make_default_gateway, Sandbox(),
                           base_dir=tempfile.mkdtemp())
    ep = gen.generate(GenerationConfig(
        task_family="data_exfiltration", seed=2, adversarial=True, max_steps=50
    ))
    assert ep.label["policy_violation"] is True
    assert ep.label["attack_family"] == "data_exfiltration"
    assert ep.label["unsafe_event_step"] is not None


def test_exfiltration_pair_uses_neutral_shared_wording():
    with tempfile.TemporaryDirectory() as tmp:
        generator = EpisodeGenerator(make_default_gateway, Sandbox(), base_dir=tmp)
        benign = generator.generate(GenerationConfig(
            task_family="data_exfiltration",
            seed=1,
            adversarial=False,
            episode_id="paired-benign",
        ))
        adversarial = generator.generate(GenerationConfig(
            task_family="data_exfiltration",
            seed=1,
            adversarial=True,
            episode_id="paired-adversarial",
        ))

    def actions(episode):
        return [
            event["payload"]
            for event in episode.trace_events
            if event["event_type"] == "agent_action"
        ]

    benign_actions = actions(benign)
    adversarial_actions = actions(adversarial)
    assert len(benign_actions) >= len(adversarial_actions)
    for benign_action, adversarial_action in zip(
        benign_actions[:len(adversarial_actions)], adversarial_actions
    ):
        benign_payload = dict(benign_action)
        adversarial_payload = dict(adversarial_action)
        benign_args = dict(benign_payload["arguments"])
        adversarial_args = dict(adversarial_payload["arguments"])
        benign_args.pop("to", None)
        adversarial_args.pop("to", None)
        assert benign_args == adversarial_args
        assert benign_payload["reasoning"] == adversarial_payload["reasoning"]
    assert benign.result["policy_violation"] is False
    assert adversarial.result["policy_violation"] is True
    assert benign.manifest.labels["wording_variant"] == "neutral-1"


def test_privilege_pair_uses_neutral_action_wording():
    with tempfile.TemporaryDirectory() as tmp:
        generator = EpisodeGenerator(make_default_gateway, Sandbox(), base_dir=tmp)
        episodes = [
            generator.generate(GenerationConfig(
                task_family="privilege_escalation",
                seed=2,
                adversarial=adversarial,
                episode_id=f"privilege-{adversarial}",
            ))
            for adversarial in (False, True)
        ]

    actions = [
        [
            event["payload"]
            for event in episode.trace_events
            if event["event_type"] == "agent_action"
        ]
        for episode in episodes
    ]
    assert len(actions[0]) >= 2
    assert len(actions[1]) == 2
    for benign_action, adversarial_action in zip(actions[0][:2], actions[1]):
        assert benign_action["tool"] == adversarial_action["tool"]
        assert benign_action["reasoning"] == adversarial_action["reasoning"]
        benign_args = dict(benign_action["arguments"])
        adversarial_args = dict(adversarial_action["arguments"])
        benign_args.pop("path", None)
        adversarial_args.pop("path", None)
        assert benign_args == adversarial_args


def test_memory_pair_shares_neutral_fact_and_action_wording():
    with tempfile.TemporaryDirectory() as tmp:
        generator = EpisodeGenerator(make_default_gateway, Sandbox(), base_dir=tmp)
        episodes = [
            generator.generate(GenerationConfig(
                task_family="memory_poisoning",
                seed=2,
                adversarial=adversarial,
                episode_id=f"memory-{adversarial}",
            ))
            for adversarial in (False, True)
        ]

    memory_actions = []
    for episode in episodes:
        memory_actions.append(next(
            event["payload"]
            for event in episode.trace_events
            if event["event_type"] == "agent_action"
            and event["payload"]["tool"] == "memory_write"
        ))

    benign, adversarial = memory_actions
    assert benign["reasoning"] == adversarial["reasoning"]
    assert benign["arguments"]["fact"] == adversarial["arguments"]["fact"]
    assert benign["arguments"]["provenance"] == "trusted"
    assert adversarial["arguments"]["provenance"] == "untrusted"


def test_all_threat_families_generate():
    gen = EpisodeGenerator(make_default_gateway, Sandbox(),
                           base_dir=tempfile.mkdtemp())
    families = [
        "indirect_prompt_injection",
        "memory_poisoning",
        "privilege_escalation",
        "data_exfiltration",
        "goal_drift",
        "multi_agent_delegation",
    ]
    for fam in families:
        ep = gen.generate(GenerationConfig(
            task_family=fam, seed=3, adversarial=True, max_steps=50
        ))
        assert ep.label["policy_violation"] is True, f"{fam} not flagged"
        assert ep.label["attack_family"] == fam
        assert ep.label["unsafe_event_step"] is not None


def test_store_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = DatasetStore(tmp)
        gen = EpisodeGenerator(make_default_gateway, Sandbox(), base_dir=tmp)
        ep = gen.generate(GenerationConfig(
            task_family="privilege_escalation", seed=7, adversarial=True
        ))
        store.save(ep)
        loaded = store.load(ep.manifest.episode_id)
        assert loaded is not None
        assert loaded["manifest"]["episode_id"] == ep.manifest.episode_id
        assert loaded["label"]["policy_violation"] is True
        index = store.list_episodes()
        assert len(index) == 1


def test_synthetic_pairs_diverge_on_dependency():
    for pair in ALL_PAIRS:
        assert pair.benign_manifest.attack_family is None
        assert pair.adversarial_manifest.attack_family is not None
        assert pair.divergence_point
        assert pair.unsafe_dependency
        assert len(pair.shared_prefix) >= 2


def test_labeler_extracts_signals():
    gen = EpisodeGenerator(make_default_gateway, Sandbox(),
                           base_dir=tempfile.mkdtemp())
    ep = gen.generate(GenerationConfig(
        task_family="data_exfiltration", seed=9, adversarial=True, max_steps=50
    ))
    labeler = Labeler()
    label = labeler.label(ep.result, ep.manifest, None)
    assert "detection_signals" in label.to_dict()


def test_template_split_holds_out_templates_and_wording():
    episodes = []
    for template, wording, episode_id in (
        ("template-a", "wording-a", "train-1"),
        ("template-a", "wording-a", "train-2"),
        ("template-b", "wording-b", "validation-1"),
        ("template-c", "wording-c", "test-1"),
    ):
        episodes.append({
            "episode_id": episode_id,
            "template_id": template,
            "wording_variant": wording,
        })

    splits = split_by_template(
        episodes,
        train_templates=["template-a"],
        validation_templates=["template-b"],
        test_templates=["template-c"],
    )
    assert [ep["episode_id"] for ep in splits.train] == ["train-1", "train-2"]
    assert splits.validation[0]["episode_id"] == "validation-1"
    assert splits.test[0]["episode_id"] == "test-1"


def test_template_split_rejects_wording_leakage():
    episodes = [
        {"episode_id": "train", "template_id": "a", "wording_variant": "same"},
        {"episode_id": "validation", "template_id": "b", "wording_variant": "val"},
        {"episode_id": "test", "template_id": "c", "wording_variant": "same"},
    ]

    with pytest.raises(ValueError, match="Wording variants overlap"):
        split_by_template(
            episodes,
            train_templates=["a"],
            validation_templates=["b"],
            test_templates=["c"],
        )
