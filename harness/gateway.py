"""MCP-compatible tool gateway with identity and policy engine.

Mediates every external action taken by the agent. Each tool call is checked
against the active policy before being forwarded to the sandbox. Policy
decisions are recorded as telemetry events and materialize as graph nodes
(authorizes / escalates_privilege edges).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .schema import EdgeType, GraphEdge, GraphNode, NodeType, TraceEvent
from .telemetry import Telemetry


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    required_identity: Optional[str] = None  # e.g. "admin", "user", "guest"
    sensitive: bool = False  # touches credentials / external effects


@dataclass
class PolicyRule:
    rule_id: str
    description: str
    condition: Callable[[Dict[str, Any]], bool]
    decision: PolicyDecision


class Gateway:
    """MCP-style tool proxy. Every external action passes through here."""

    def __init__(self, telemetry: Telemetry, identity: str = "agent"):
        self.telemetry = telemetry
        self.identity = identity
        self.tools: Dict[str, ToolDefinition] = {}
        self.rules: List[PolicyRule] = []
        self._tool_registry: Dict[str, Callable[..., Any]] = {}
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}

    # ---- Tool registration -------------------------------------------------

    def register_tool(self, tool: ToolDefinition, handler: Callable[..., Any]) -> None:
        self.tools[tool.name] = tool
        self._tool_registry[tool.name] = handler

    def register_rule(self, rule: PolicyRule) -> None:
        self.rules.append(rule)

    # ---- Policy engine -----------------------------------------------------

    def evaluate(self, tool_name: str, arguments: Dict[str, Any]) -> PolicyDecision:
        tool = self.tools.get(tool_name)
        if tool is None:
            return PolicyDecision.DENY

        # Identity check.
        if tool.required_identity and tool.required_identity != self.identity:
            return PolicyDecision.DENY

        # Rule evaluation (first matching rule wins).
        for rule in self.rules:
            if rule.condition(arguments):
                return rule.decision

        # Default: allow non-sensitive tools, require approval for sensitive.
        if tool.sensitive:
            return PolicyDecision.REQUIRE_APPROVAL
        return PolicyDecision.ALLOW

    # ---- Mediation ---------------------------------------------------------

    def mediate(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        """Run a tool call through the policy engine and return the decision."""
        decision = self.evaluate(tool_name, arguments)

        # Record policy decision as a graph node.
        decision_node = GraphNode(
            node_type=NodeType.POLICY_DECISION,
            attributes={
                "tool": tool_name,
                "decision": decision.value,
                "identity": self.identity,
                "arguments": arguments,
            },
            episode_id=self.telemetry._episode_id or "unknown",
            step=step,
        )

        # Telemetry event.
        self.telemetry.record(
            event_type="policy_decision",
            actor="gateway",
            payload={
                "tool": tool_name,
                "decision": decision.value,
                "arguments": arguments,
                "node_id": decision_node.node_id,
            },
        )

        return {
            "decision": decision.value,
            "node_id": decision_node.node_id,
            "tool": tool_name,
            "arguments": arguments,
        }

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool (bypasses policy; intended for trusted/sandbox use)."""
        handler = self._tool_registry.get(tool_name)
        if handler is None:
            raise KeyError(f"Tool not registered: {tool_name}")
        return handler(**arguments)
