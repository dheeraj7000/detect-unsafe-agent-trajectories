"""Agent interface: LLM + planner + memory.

A real implementation wraps an LLM. Here we provide a deterministic mock
so the harness can run end-to-end without API keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AgentAction:
    """A single action proposed by the agent."""
    step: int
    tool: str
    arguments: Dict[str, Any]
    reasoning: str


@dataclass
class AgentMemory:
    """Simple agent memory store."""
    facts: List[str] = field(default_factory=list)

    def write(self, fact: str) -> None:
        self.facts.append(fact)

    def read(self) -> List[str]:
        return list(self.facts)


class Agent:
    """Abstract agent interface. A real implementation wraps an LLM."""

    def __init__(self, name: str = "mock_agent"):
        self.name = name
        self.memory = AgentMemory()
        self.history: List[AgentAction] = []
        self._policy: Optional[Callable[[str, int], str]] = None

    def set_policy(self, policy: Callable[[str, int], str]) -> None:
        """Inject a deterministic policy that chooses the next action.

        Policy signature: (observation: str, step: int) -> decision: str
        Decision format: "tool:<name>|args:<json>|reason:<text>"
        """
        self._policy = policy

    def act(self, observation: str, step: int) -> AgentAction:
        if self._policy is not None:
            decision = self._policy(observation, step)
            return self._parse_decision(decision, step)
        # Default no-op.
        return AgentAction(step=step, tool="noop", arguments={}, reasoning=observation)

    def _parse_decision(self, decision: str, step: int) -> AgentAction:
        try:
            parts = decision.split("|")
            tool = parts[0].split(":", 1)[1] if len(parts) > 0 else "noop"
            args_str = parts[1].split(":", 1)[1] if len(parts) > 1 else "{}"
            arguments = json.loads(args_str)
            reasoning = parts[2].split(":", 1)[1] if len(parts) > 2 else ""
        except Exception:  # noqa: BLE001
            tool, arguments, reasoning = "noop", {}, decision
        action = AgentAction(step=step, tool=tool, arguments=arguments, reasoning=reasoning)
        self.history.append(action)
        return action

    def remember(self, fact: str) -> None:
        self.memory.write(fact)
