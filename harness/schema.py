"""Canonical event schema for the GraphGuard Agent Control & Evaluation Lab.

Defines the typed temporal behavior-graph elements (nodes + edges) and the
OpenTelemetry-style trace events that feed both sequence and graph monitors.

Node types (from proposal):
    goal, subgoal, observation, instruction, tool_call, tool_result,
    file_resource, credential, memory_write, memory_read, policy_decision,
    user_approval, external_effect

Edge types (from proposal):
    follows, derived_from, reads, writes, authorizes, invokes, depends_on,
    contradicts, escalates_privilege, sends_to, persists_to_memory
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    GOAL = "goal"
    SUBGOAL = "subgoal"
    OBSERVATION = "observation"
    INSTRUCTION = "instruction"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE_RESOURCE = "file_resource"
    CREDENTIAL = "credential"
    MEMORY_WRITE = "memory_write"
    MEMORY_READ = "memory_read"
    POLICY_DECISION = "policy_decision"
    USER_APPROVAL = "user_approval"
    EXTERNAL_EFFECT = "external_effect"


class EdgeType(str, Enum):
    FOLLOWS = "follows"
    DERIVED_FROM = "derived_from"
    READS = "reads"
    WRITES = "writes"
    AUTHORIZES = "authorizes"
    INVOKES = "invokes"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    ESCALATES_PRIVILEGE = "escalates_privilege"
    SENDS_TO = "sends_to"
    PERSISTS_TO_MEMORY = "persists_to_memory"


class EventType(str, Enum):
    """Telemetry event types emitted during an episode."""
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    AGENT_ACTION = "agent_action"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OBSERVATION = "observation"
    MEMORY_WRITE = "memory_write"
    MEMORY_READ = "memory_read"
    POLICY_DECISION = "policy_decision"
    USER_APPROVAL = "user_approval"
    EXTERNAL_EFFECT = "external_effect"
    POLICY_VIOLATION = "policy_violation"
    INTERVENTION = "intervention"


class GraphNode(BaseModel):
    """A typed node in the partial behavior graph."""
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    node_type: NodeType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: Dict[str, Any] = Field(default_factory=dict)
    episode_id: str
    step: int


class GraphEdge(BaseModel):
    """A typed, directed edge in the partial behavior graph."""
    edge_id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    edge_type: EdgeType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: Dict[str, Any] = Field(default_factory=dict)
    episode_id: str
    step: int


class TraceEvent(BaseModel):
    """A single OpenTelemetry-style trace event recorded during an episode."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    episode_id: str
    step: int
    event_type: EventType
    actor: str  # e.g. "agent", "gateway", "sandbox", "monitor"
    payload: Dict[str, Any] = Field(default_factory=dict)
    parent_event_id: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class EpisodeManifest(BaseModel):
    """Immutable scenario manifest for deterministic replay."""
    episode_id: str
    task_family: str
    scenario_name: str
    attack_family: Optional[str] = None
    seed: int
    max_steps: int = 500
    unsafe_event_step: Optional[int] = None  # ground-truth onset
    policy_violation: bool = False
    affected_asset: Optional[str] = None
    trust_boundary: Optional[str] = None
    severity: Optional[str] = None
    labels: Dict[str, Any] = Field(default_factory=dict)


class EpisodeResult(BaseModel):
    """Outcome of a single episode run."""
    episode_id: str
    task_success: bool
    policy_violation: bool
    unsafe_event_step: Optional[int]
    total_steps: int
    intervention_step: Optional[int] = None
    monitor_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    trace_path: Optional[str] = None
    graph_path: Optional[str] = None
