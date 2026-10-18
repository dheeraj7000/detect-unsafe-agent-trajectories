"""Tests for task-family scenarios and deterministic replay."""

from __future__ import annotations

from harness import (
    Agent,
    HarnessConfig,
    Orchestrator,
    Sandbox,
    Telemetry,
    make_default_gateway,
    verify_determinism,
)
from harness.scenarios import TASK_FAMILIES, get_scenario


def test_all_families_deterministic():
    for family in TASK_FAMILIES:
        manifest, policy = get_scenario(family)
        telemetry = Telemetry(base_dir="data/traces")
        sandbox = Sandbox()
        gateway = make_default_gateway(telemetry)
        agent = Agent(name=f"test_{family}")
        orch = Orchestrator(
            agent=agent,
            gateway=gateway,
            sandbox=sandbox,
            telemetry=telemetry,
            config=HarnessConfig(max_steps=manifest.max_steps),
        )
        result = orch.run_episode(manifest, policy)
        check = verify_determinism(orch, manifest, policy, runs=3)
        assert check["deterministic"], f"{family} not deterministic"
        assert result["total_steps"] > 0
        assert result["intervention_step"] is not None
