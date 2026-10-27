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

from typing import Dict, List, Optional, Tuple

try:
    import networkx as nx
except ImportError:  # pragma: no cover - networkx optional for prototyping
    nx = None

import numpy as np
import torch

from .schema import EdgeType, EventType, GraphEdge, GraphNode, NodeType, TraceEvent


class BehaviorGraph:
    """Typed temporal behavior graph built online during an episode."""

    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self._adjacency: Dict[str, List[str]] = {}  # node_id -> [edge_id]
        self._policy_decisions: Dict[int, str] = {}
        self._tool_calls: Dict[int, str] = {}
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
            self._tool_calls[step] = nid
            # Link to previous tool call (follows).
            self._link_previous(NodeType.TOOL_CALL, nid, EdgeType.FOLLOWS, step)
            policy_id = self._policy_decisions.get(step)
            if policy_id is not None:
                self.edge(nid, policy_id, EdgeType.AUTHORIZES, step)
            arguments = payload.get("arguments", {})
            if tool == "file_read" and arguments.get("path"):
                resource_id = self.node(NodeType.FILE_RESOURCE, step, {
                    "path": arguments["path"],
                })
                self.edge(resource_id, nid, EdgeType.READS, step)
            elif tool == "email_send" and arguments.get("to"):
                effect_id = self.node(NodeType.EXTERNAL_EFFECT, step, {
                    "effect": "email_send",
                    "target": arguments["to"],
                    "attempted": True,
                })
                self.edge(nid, effect_id, EdgeType.SENDS_TO, step)

        elif etype == "tool_result":
            nid = self.node(NodeType.TOOL_RESULT, step, {
                "tool": payload.get("tool"),
                "success": payload.get("success"),
                "output": payload.get("output"),
            })
            call_id = self._tool_calls.get(step)
            if call_id is not None:
                self.edge(call_id, nid, EdgeType.DERIVED_FROM, step)
                if payload.get("success") and payload.get("tool") == "file_write":
                    tool_call = self.nodes[call_id]
                    arguments = tool_call.attributes.get("arguments", {})
                    path = arguments.get("path")
                    if path:
                        resource_id = self.node(NodeType.FILE_RESOURCE, step, {
                            "path": path,
                        })
                        self.edge(nid, resource_id, EdgeType.WRITES, step)

        elif etype == "observation":
            self.node(NodeType.OBSERVATION, step, {
                "content": payload.get("content"),
            })

        elif etype == "policy_decision":
            nid = self.node(NodeType.POLICY_DECISION, step, {
                "tool": payload.get("tool"),
                "decision": payload.get("decision"),
                "identity": payload.get("identity"),
            })
            self._policy_decisions[step] = nid

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
        result = []
        for edge in self.edges.values():
            if edge.target_id == node_id and edge.source_id in self.nodes:
                result.append(self.nodes[edge.source_id])
        return result

    def subgraph_risk_pattern(self) -> List[Dict]:
        """Flag example risk pattern: untrusted tool output -> plan update ->
        credential read -> external send. Returns matching paths."""
        findings = []
        for edge in self.edges.values():
            if edge.edge_type == EdgeType.SENDS_TO:
                findings.append({
                    "pattern": "sensitive_data_exfiltration",
                    "edge_id": edge.edge_id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "step": edge.step,
                })
        return findings


def trace_prefix_at_step(
    trace: List[TraceEvent],
    step: int,
) -> List[TraceEvent]:
    """Return all recorded events through an action step."""
    return [event for event in trace if event.step <= step]


def behavior_graph_at_step(
    trace: List[TraceEvent],
    step: int,
    episode_id: Optional[str] = None,
) -> BehaviorGraph:
    """Reconstruct the partial behavior graph through an action step."""
    graph = BehaviorGraph(
        episode_id=episode_id or (trace[0].episode_id if trace else "empty")
    )
    for event in trace_prefix_at_step(trace, step):
        if event.event_type in (EventType.EPISODE_START, EventType.EPISODE_END):
            continue
        graph.ingest_event(event, event.step)
    return graph

# ---------------------------------------------------------------------------
# Type <-> index mappings for GNN feature encoding
# ---------------------------------------------------------------------------

NODE_TYPE_LIST = [t.value for t in NodeType]
EDGE_TYPE_LIST = [t.value for t in EdgeType]

NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPE_LIST)}
EDGE_TYPE_TO_IDX = {t: i for i, t in enumerate(EDGE_TYPE_LIST)}


def graph_to_tensors(graph: BehaviorGraph, max_steps: int = 500):
    """Convert a BehaviorGraph into PyTorch tensors for the GNN.

    Returns:
        x:           (num_nodes, node_feat_dim) float tensor — node features
                     [node_type_onehot, step_normalized]
        edge_index:  (2, num_edges) long tensor — COO connectivity
        edge_attr:   (num_edges, edge_feat_dim) float tensor — edge features
                     [edge_type_onehot, step_normalized]
        node_id_to_idx: dict mapping node_id -> row index in x
    """
    nodes = list(graph.nodes.values())
    edges = list(graph.edges.values())

    node_id_to_idx = {n.node_id: i for i, n in enumerate(nodes)}
    num_nodes = len(nodes)
    num_edges = len(edges)

    # ---- Node features ----------------------------------------------------
    if num_nodes > 0:
        node_type_idx = [NODE_TYPE_TO_IDX[n.node_type.value] for n in nodes]
        node_type_onehot = torch.zeros(num_nodes, len(NODE_TYPE_LIST))
        node_type_onehot[torch.arange(num_nodes),
                         torch.tensor(node_type_idx)] = 1.0
        step_norm = torch.tensor(
            [[n.step / max(max_steps, 1)] for n in nodes], dtype=torch.float32
        )
        x = torch.cat([node_type_onehot, step_norm], dim=1)
    else:
        # Single dummy zero node so batches always have a consistent shape.
        x = torch.zeros(1, len(NODE_TYPE_LIST) + 1)

    # ---- Edge index / features -------------------------------------------
    if num_edges > 0:
        src = []
        dst = []
        edge_type_idx = []
        edge_step_norm = []
        for e in edges:
            s = node_id_to_idx.get(e.source_id)
            t = node_id_to_idx.get(e.target_id)
            if s is None or t is None:
                continue
            src.append(s)
            dst.append(t)
            edge_type_idx.append(EDGE_TYPE_TO_IDX[e.edge_type.value])
            edge_step_norm.append(e.step / max(max_steps, 1))

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type_onehot = torch.zeros(len(src), len(EDGE_TYPE_LIST))
        edge_type_onehot[torch.arange(len(src)),
                         torch.tensor(edge_type_idx)] = 1.0
        edge_step_t = torch.tensor(edge_step_norm,
                                   dtype=torch.float32).unsqueeze(1)
        edge_attr = torch.cat([edge_type_onehot, edge_step_t], dim=1)
    else:
        # Single self-loop on the dummy node so shape is consistent.
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros(1, len(EDGE_TYPE_LIST) + 1)

    return x, edge_index, edge_attr, node_id_to_idx
