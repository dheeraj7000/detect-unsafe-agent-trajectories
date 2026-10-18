"""Tests for Phase 2 dataset generation, labeling, and storage."""

from __future__ import annotations

import tempfile

from harness import (Agent, Gateway, HarnessConfig, Orchestrator, Sandbox,
                     Telemetry, make_default_gateway, verify_determinism)
from dataset import EpisodeGenerator, GenerationConfig, DatasetStore
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


def test_generate_adversarial_episode():
    gen = EpisodeGenerator(make_default_gateway, Sandbox(),
                           base_dir=tempfile.mkdtemp())
    ep = gen.generate(GenerationConfig(
        task_family="data_exfiltration", seed=2, adversarial=True, max_steps=50
    ))
    assert ep.label["policy_violation"] is True
    assert ep.label["attack_family"] == "data_exfiltration"
    assert ep.label["unsafe_event_step"] is not None


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
