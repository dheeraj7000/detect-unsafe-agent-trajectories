"""End-to-end demo of Phase 1 + Phase 2: run all 3 task families through the
harness, generate labeled adversarial episodes, build behavior graphs online,
and verify deterministic replay for each.

Run:  python demo.py
"""

from __future__ import annotations

import json

from harness import (
    Agent,
    HarnessConfig,
    Orchestrator,
    Sandbox,
    Telemetry,
    make_default_gateway,
    verify_determinism,
)
from harness.scenarios import get_scenario, TASK_FAMILIES
from dataset import EpisodeGenerator, GenerationConfig


def run_harness_family(family: str) -> dict:
    """Run a single Phase 1 scenario family through the orchestrator."""
    manifest, policy = get_scenario(family)

    telemetry = Telemetry(base_dir="data/traces")
    sandbox = Sandbox()
    gateway = make_default_gateway(telemetry)
    agent = Agent(name=f"{family}_agent")

    orch = Orchestrator(
        agent=agent,
        gateway=gateway,
        sandbox=sandbox,
        telemetry=telemetry,
        config=HarnessConfig(
            max_steps=manifest.max_steps, auto_intervene_on_deny=True
        ),
    )

    result = orch.run_episode(manifest, policy)
    check = verify_determinism(orch, manifest, policy, runs=3)

    return {
        "family": family,
        "result": result,
        "deterministic": check["deterministic"],
        "graph_nodes": len(orch.graph.nodes) if orch.graph else 0,
        "graph_edges": len(orch.graph.edges) if orch.graph else 0,
    }


def generate_dataset_episode(family: str, adversarial: bool, seed: int) -> dict:
    """Generate one Phase 2 labeled episode."""
    gen = EpisodeGenerator(make_default_gateway, Sandbox(),
                           base_dir="data/episodes")
    ep = gen.generate(GenerationConfig(
        task_family=family, seed=seed, adversarial=adversarial, max_steps=100
    ))
    return {
        "family": family,
        "adversarial": adversarial,
        "policy_violation": ep.label["policy_violation"],
        "unsafe_event_step": ep.label["unsafe_event_step"],
        "attack_family": ep.label["attack_family"],
        "severity": ep.label["severity"],
        "trace_events": len(ep.trace_events),
        "graph_nodes": len(ep.graph_nodes),
        "graph_edges": len(ep.graph_edges),
    }


def main() -> None:
    print("=" * 60)
    print("GraphGuard Demo — Phase 1 harness + Phase 2 dataset")
    print("=" * 60)

    print("\n--- Phase 1: 3 task families (deterministic replay) ---")
    all_det = True
    for family in TASK_FAMILIES:
        s = run_harness_family(family)
        all_det = all_det and s["deterministic"]
        print(
            f"  {s['family']:14s} steps={s['result']['total_steps']:3d} "
            f"violation={str(s['result']['policy_violation']):5s} "
            f"intervention={s['result']['intervention_step']} "
            f"graph={s['graph_nodes']}n/{s['graph_edges']}e "
            f"deterministic={'OK' if s['deterministic'] else 'FAIL'}"
        )
    print(f"  All deterministic: {all_det}")

    print("\n--- Phase 2: labeled dataset (benign + adversarial) ---")
    dataset_families = [
        "indirect_prompt_injection",
        "memory_poisoning",
        "privilege_escalation",
        "data_exfiltration",
        "goal_drift",
        "multi_agent_delegation",
    ]
    for fam in dataset_families:
        # Benign
        b = generate_dataset_episode(fam, adversarial=False, seed=100)
        # Adversarial
        a = generate_dataset_episode(fam, adversarial=True, seed=100)
        print(
            f"  {fam:26s} benign_violation={str(b['policy_violation']):5s} "
            f"adv_violation={str(a['policy_violation']):5s} "
            f"adv_onset={a['unsafe_event_step']} "
            f"attack={a['attack_family']} severity={a['severity']}"
        )

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
