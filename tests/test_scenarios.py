"""Tests for task-family scenarios and deterministic replay."""

from __future__ import annotations

from harness import (
    Agent,
    Gateway,
    HarnessConfig,
    Orchestrator,
    PolicyDecision,
    PolicyRule,
    Sandbox,
    Telemetry,
    ToolDefinition,
    verify_determinism,
)
from harness.scenarios import TASK_FAMILIES, get_scenario


def _make_gateway(telemetry: Telemetry) -> Gateway:
    gw = Gateway(telemetry, identity="agent")
    for name, sensitive in [
        ("file_read", False),
        ("file_write", True),
        ("email_send", True),
        ("email_read", False),
        ("code_eval", True),
    ]:
        gw.register_tool(
            ToolDefinition(name, name, {}, sensitive=sensitive),
            lambda **kwargs: None,
        )

    def deny_cred(args):
        content = str(args.get("content", ""))
        return "credential" in content.lower() or "password" in content.lower()

    def deny_external(args):
        to = str(args.get("to", ""))
        return "@attacker" in to or "attacker.com" in to

    def deny_dangerous_code(args):
        expr = str(args.get("expression", ""))
        return "__import__" in expr or "system(" in expr or "curl" in expr

    gw.register_rule(
        PolicyRule("no_cred", "no cred", deny_cred, PolicyDecision.DENY)
    )
    gw.register_rule(
        PolicyRule("no_ext", "no ext", deny_external, PolicyDecision.DENY)
    )
    gw.register_rule(
        PolicyRule("no_code", "no code", deny_dangerous_code, PolicyDecision.DENY)
    )
    return gw


def test_all_families_deterministic():
    for family in TASK_FAMILIES:
        manifest, policy = get_scenario(family)
        telemetry = Telemetry(base_dir="data/traces")
        sandbox = Sandbox()
        gateway = _make_gateway(telemetry)
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
