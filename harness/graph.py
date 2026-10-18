"""Online behavior-graph builder.

Constructs a typed temporal graph from telemetry events as the episode
unfolds. Nodes and edges follow the GraphGuard schema:

Nodes: goal, subgoal, observation, instruction, tool_call, tool_result,
       file_resource, credential, memory_write, memory_read, policy_decision,
       user_approval, external_effect
Edges: follows, derived_from, reads, writes, authorizes, invokes, depends_on,
       contradicts, escalates_privilege, sends_to, persists_to_memory
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import networkx as nx
except ImportError:  # pragma: no cover - networkx optional for prototyping
    nx = None

from .schema import EdgeType, GraphEdge, GraphNode, NodeType, TraceEvent


class BehaviorGraph:
    """Typed temporal behavior graph built online during an episode."""

    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self._adjacency: Dict[str, List[str]] = {}  # node_id -> [edge_id]
        if nx is not None:
            self._nx = nx.DiGraph()
        else:
            self._nx = None

    # ---- Node / edge registration -----------------------------------------

    def add_node(self, node: GraphNode) -> str:
        self.nodes[node.node_id] = node
        if self._nx is not None:
            self._nx.add_node(
                node.node_id,
                node_type=node.node_type.value,
                attributes=node.attributes,
                step=node.step,
            )
        return node.node_id

    def add_edge(self, edge: GraphEdge) -> str:
        self.edges[edge.edge_id] = edge
        self._adjacency.setdefault(edge.source_id, []).append(edge.edge_id)
        if self._nx is not None:
            self._nx.add_edge(
                edge.source_id,
                edge.target_id,
                edge_id=edge.edge_id,
                edge_type=edge.edge_type.value,
                step=edge.step,
            )
        return edge.edge_id

    def edge(self, source_id: str, target_id: str, edge_type: EdgeType,
             step: int, attributes: Optional[dict] = None) -> str:
        """Convenience: create and add a typed edge."""
        e = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            step=step,
            attributes=attributes or {},
            episode_id=self.episode_id,
        )
        return self.add_edge(e)

    def node(self, node_type: NodeType, step: int,
             attributes: Optional[dict] = None) -> str:
        """Convenience: create and add a typed node."""
        n = GraphNode(
            node_type=node_type,
            step=step,
            attributes=attributes or {},
            episode_id=self.episode_id,
        )
        return self.add_node(n)

    # ---- Online event ingestion -------------------------------------------

    def ingest_event(self, event: TraceEvent, step: int) -> None:
        """Derive graph nodes/edges from a telemetry event."""
        etype = event.event_type
        payload = event.payload

        if etype == "agent_action":
            tool = payload.get("tool", "noop")
            nid = self.node(NodeType.TOOL_CALL, step, {
                "tool": tool,
                "arguments": payload.get("arguments", {}),
                "reasoning": payload.get("reasoning", ""),
            })
            # Link to previous tool call (follows).
            self._link_previous(NodeType.TOOL_CALL, nid, EdgeType.FOLLOWS, step)

        elif etype == "tool_result":
            nid = self.node(NodeType.TOOL_RESULT, step, {
                "tool": payload.get("tool"),
                "success": payload.get("success"),
                "output": payload.get("output"),
            })
            # Link result to its tool call (derived_from / invokes).
            self._link_previous(NodeType.TOOL_CALL, nid, EdgeType.DERIVED_FROM, step)

        elif etype == "observation":
            self.node(NodeType.OBSERVATION, step, {
                "content": payload.get("content"),
            })

        elif etype == "policy_decision":
            self.node(NodeType.POLICY_DECISION, step, {
                "tool": payload.get("tool"),
                "decision": payload.get("decision"),
                "identity": payload.get("identity"),
            })

        elif etype == "memory_write":
            self.node(NodeType.MEMORY_WRITE, step, {
                "fact": payload.get("fact"),
            })
            self._link_previous(NodeType.MEMORY_WRITE, None, EdgeType.PERSISTS_TO_MEMORY, step)

        elif etype == "external_effect":
            self.node(NodeType.EXTERNAL_EFFECT, step, {
                "effect": payload.get("effect"),
                "target": payload.get("target"),
            })

    def _link_previous(self, node_type: NodeType, new_id: Optional[str],
                       edge_type: EdgeType, step: int) -> None:
        """Link the most recent node of a type to new_id (if provided)."""
        candidates = [
            n for n in self.nodes.values()
            if n.node_type == node_type
        ]
        if not candidates:
            return
        prev = max(candidates, key=lambda n: n.step)
        if new_id is not None:
            self.edge(prev.node_id, new_id, edge_type, step)

    def predecessors(self, node_id: str) -> List[GraphNode]:
        if self._nx is not None:
            return [self.nodes[n] for n in self._nx.predecessors(node_id)
                    if n in self.nodes]
        # Fallback without networkx.
        result = []
        for e in self.edges.values():
            if e.target_id == node_id and e.source_id in self.nodes:
                result.append(self.nodes[e.source_id])
        return result

    def subgraph_risk_pattern(self) -> List[Dict]:
        """Flag example risk pattern: untrusted tool output -> plan update ->
        credential read -> external send. Returns matching paths."""
        # Lightweight structural check (placeholder for GNN scoring).
        findings = []
        for e in self.edges.values():
            if e.edge_type == EdgeType.SENDS_TO:
                findings.append({
                    "pattern": "sensitive_data_exfiltration",
                    "edge_id": e.edge_id,
                    "source": e.source_id,
                    "target": e.target_id,
                    "step": e.step,
                })
        return findings
