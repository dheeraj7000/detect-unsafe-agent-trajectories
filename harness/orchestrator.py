"""Harness orchestrator: runs an agent through a scenario, mediates actions
through the gateway, records telemetry, and builds the behavior graph online.

Wires together:
  Agent  ->  Gateway (policy)  ->  Sandbox  ->  Telemetry + BehaviorGraph
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .agent import Agent, AgentAction
from .gateway import Gateway
from .graph import BehaviorGraph, EdgeType, NodeType
from .sandbox import Sandbox, SandboxResult
from .schema import EpisodeManifest
from .telemetry import Telemetry


@dataclass
class StepOutcome:
    step: int
    action: AgentAction
    gateway_decision: Dict[str, Any]
    sandbox_result: SandboxResult
    intervened: bool
    alert: bool


@dataclass
class HarnessConfig:
    max_steps: int = 500
    auto_intervene_on_deny: bool = True
    record_graph: bool = True


class Orchestrator:
    """Runs a single episode end-to-end."""

    def __init__(
        self,
        agent: Agent,
        gateway: Gateway,
        sandbox: Sandbox,
        telemetry: Telemetry,
        config: Optional[HarnessConfig] = None,
    ):
        self.agent = agent
        self.gateway = gateway
        self.sandbox = sandbox
        self.telemetry = telemetry
        self.config = config or HarnessConfig()
        self.graph: Optional[BehaviorGraph] = None
        self.steps: List[StepOutcome] = []
        self.intervention_step: Optional[int] = None
        self.alerts: List[Dict[str, Any]] = []

    def run_episode(self, manifest: EpisodeManifest,
                    policy: Callable[[str, int], str]) -> Dict[str, Any]:
        """Execute one episode deterministically.

        Args:
            manifest: Immutable scenario manifest (seed, unsafe onset, etc.).
            policy: Agent policy function (observation, step) -> decision.
        """
        self.telemetry.begin_episode(manifest.episode_id)
        if self.config.record_graph:
            self.graph = BehaviorGraph(manifest.episode_id)

        self.agent.set_policy(policy)
        self.steps = []
        self.intervention_step = None
        self.alerts = []

        observation = f"Start task: {manifest.scenario_name}"
        task_success = False
        policy_violation = False
        sandbox_failure = False
        unsafe_event_step = None
        sandbox_result: Optional[SandboxResult] = None

        for step in range(1, self.config.max_steps + 1):
            self.telemetry.next_step()

            # 1. Agent proposes an action.
            action = self.agent.act(observation, step)

            # 2. Gateway mediates (policy check).
            #    "noop" is a local no-op, not an external action — bypass gateway.
            if action.tool == "noop":
                decision = {"decision": "allow", "node_id": None, "tool": "noop",
                            "arguments": action.arguments}
                sandbox_result = SandboxResult(success=True, output="noop", side_effects=[])
                intervened = False
                alert = False
            else:
                # 2a. Gateway decision.
                decision = self.gateway.mediate(action.tool, action.arguments, step)

                # 2b. Intervention / execution.
                intervened = False
                alert = False

                if decision["decision"] == "deny" and self.config.auto_intervene_on_deny:
                    intervened = True
                    alert = True
                    self.intervention_step = step
                    self.alerts.append({
                        "step": step,
                        "reason": "policy_deny",
                        "tool": action.tool,
                        "node_id": decision["node_id"],
                    })
                    observation = (
                        f"Action blocked at step {step}: {action.tool} denied by policy."
                    )
                    policy_violation = True
                    if unsafe_event_step is None:
                        unsafe_event_step = step
                    sandbox_result = SandboxResult(
                        success=False, error="blocked by policy", side_effects=[]
                    )

                elif decision["decision"] == "require_approval":
                    alert = True
                    self.alerts.append({
                        "step": step,
                        "reason": "requires_approval",
                        "tool": action.tool,
                        "node_id": decision["node_id"],
                    })
                    sandbox_result = self.sandbox.run(action.tool, action.arguments)
                    observation = f"Sandbox result: {sandbox_result.output}"

                else:
                    sandbox_result = self.sandbox.run(action.tool, action.arguments)
                    observation = f"Sandbox result: {sandbox_result.output}"
                    if not sandbox_result.success:
                        sandbox_failure = True
                        # A failed tool call is not, by itself, an unsafe
                        # security event; only policy denials mark the onset.

            # 4. Record telemetry events.
            self.telemetry.record(
                event_type="agent_action",
                actor="agent",
                payload={
                    "tool": action.tool,
                    "arguments": action.arguments,
                    "reasoning": action.reasoning,
                },
            )
            self.telemetry.record(
                event_type="tool_result",
                actor="sandbox",
                payload={
                    "tool": action.tool,
                    "success": sandbox_result.success,
                    "output": sandbox_result.output,
                },
            )

            # 5. Ingest into the online behavior graph.
            if self.graph is not None:
                self._ingest(action, decision, sandbox_result, step)

            outcome = StepOutcome(
                step=step,
                action=action,
                gateway_decision=decision,
                sandbox_result=sandbox_result,
                intervened=intervened,
                alert=alert,
            )
            self.steps.append(outcome)

            # 6. Termination: halt on first denial (unsafe onset detected).
            if intervened and self.config.auto_intervene_on_deny:
                task_success = False
                break

            if action.tool == "noop" and action.reasoning.strip().lower() == "done.":
                task_success = True
                break

            if step == self.config.max_steps:
                task_success = True

        # End episode.
        result = {
            "episode_id": manifest.episode_id,
            "task_success": task_success,
            "policy_violation": policy_violation,
            "sandbox_failure": sandbox_failure,
            "unsafe_event_step": unsafe_event_step,
            "total_steps": len(self.steps),
            "intervention_step": self.intervention_step,
            "alerts": self.alerts,
        }
        self.telemetry.end_episode(result)
        return result

    # ------------------------------------------------------------------
    # Graph ingestion helpers
    # ------------------------------------------------------------------

    def _ingest(self, action: AgentAction, decision: Dict[str, Any],
                sandbox_result: SandboxResult, step: int) -> None:
        if self.graph is None:
            return

        g = self.graph

        # Tool-call node.
        call_id = g.node(NodeType.TOOL_CALL, step, {
            "tool": action.tool,
            "arguments": action.arguments,
            "reasoning": action.reasoning,
            "gateway_decision": decision["decision"],
        })
        # Follows: link to previous tool call.
        self._link_previous(g, NodeType.TOOL_CALL, call_id, EdgeType.FOLLOWS, step)

        # Tool-result node.
        result_id = g.node(NodeType.TOOL_RESULT, step, {
            "tool": action.tool,
            "success": sandbox_result.success,
            "output": sandbox_result.output,
        })
        g.edge(call_id, result_id, EdgeType.DERIVED_FROM, step, {})

        # Observation node.
        g.node(NodeType.OBSERVATION, step, {
            "content": sandbox_result.output,
        })

        # Policy decision node (already created in gateway; here we link).
        if decision.get("node_id"):
            g.edge(call_id, decision["node_id"], EdgeType.AUTHORIZES, step, {})

        # Side-effect nodes for external effects / file writes.
        for se in sandbox_result.side_effects:
            if se.startswith("email_send:"):
                target = se.split(":", 1)[1]
                eff_id = g.node(NodeType.EXTERNAL_EFFECT, step, {
                    "effect": "email_send",
                    "target": target,
                })
                g.edge(result_id, eff_id, EdgeType.SENDS_TO, step, {})
            elif se.startswith("write:"):
                path = se.split(":", 1)[1]
                res_id = g.node(NodeType.FILE_RESOURCE, step, {
                    "path": path,
                })
                g.edge(result_id, res_id, EdgeType.WRITES, step, {})

    @staticmethod
    def _link_previous(g: BehaviorGraph, node_type: NodeType,
                       new_id: str, edge_type: EdgeType, step: int) -> None:
        candidates = [n for n in g.nodes.values() if n.node_type == node_type]
        if not candidates:
            return
        prev = max(candidates, key=lambda n: n.step)
        g.edge(prev.node_id, new_id, edge_type, step, {})
