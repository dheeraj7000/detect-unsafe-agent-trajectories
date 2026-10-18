"""GraphGuard — Early Warning of Unsafe Long-Horizon Agent Behavior.

Public package re-exporting the core modules:

  harness   — Agent Control & Evaluation Lab (gateway, sandbox, telemetry, graph)
  dataset   — Labeled episode generation, labeling, storage
  monitors  — Rules, feature, sequence, temporal, graph monitors
  evaluator — Study runner, statistics, ablations, counterfactual replay

Version: 0.1.0
"""

from harness import (  # noqa: F401
    Agent,
    AgentAction,
    AgentMemory,
    BehaviorGraph,
    EdgeType,
    Gateway,
    HarnessConfig,
    NodeType,
    Orchestrator,
    Sandbox,
    SandboxResult,
    StepOutcome,
    Telemetry,
    TraceStore,
    make_default_gateway,
    replay_episode,
    verify_determinism,
)
from monitors import (  # noqa: F401
    BaseMonitor,
    GraphMonitor,
    FeatureMonitor,
    RulesMonitor,
    SequenceMonitor,
    TemporalSequenceMonitor,
    MonitorRegistry,
    calibrate_monitor,
    calibrate_all,
    evaluate_episode,
    make_baselines,
    make_proposed,
    make_all,
)
from evaluator import (  # noqa: F401
    StudyRunner,
    StudyResult,
    AblationConfig,
    DEFAULT_ABLATIONS,
    run_ablation_study,
    counterfactual_replay,
    CounterfactualResult,
    remove_edge_type_transform,
    remove_sends_to_external_transform,
    sanitize_tool_output_transform,
    bootstrap_ci,
    paired_ttest,
    wilcoxon_signed_rank,
)
from dataset import (  # noqa: F401
    EpisodeGenerator,
    GenerationConfig,
    DatasetStore,
    Labeler,
    EpisodeLabel,
    save_episode,
    load_episode,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # harness
    "Agent",
    "AgentAction",
    "AgentMemory",
    "BehaviorGraph",
    "EdgeType",
    "Gateway",
    "HarnessConfig",
    "NodeType",
    "Orchestrator",
    "Sandbox",
    "SandboxResult",
    "StepOutcome",
    "Telemetry",
    "TraceStore",
    "make_default_gateway",
    "replay_episode",
    "verify_determinism",
    # monitors
    "BaseMonitor",
    "GraphMonitor",
    "FeatureMonitor",
    "RulesMonitor",
    "SequenceMonitor",
    "TemporalSequenceMonitor",
    "MonitorRegistry",
    "calibrate_monitor",
    "calibrate_all",
    "evaluate_episode",
    "make_baselines",
    "make_proposed",
    "make_all",
    # evaluator
    "StudyRunner",
    "StudyResult",
    "AblationConfig",
    "DEFAULT_ABLATIONS",
    "run_ablation_study",
    "counterfactual_replay",
    "CounterfactualResult",
    "remove_edge_type_transform",
    "remove_sends_to_external_transform",
    "sanitize_tool_output_transform",
    "bootstrap_ci",
    "paired_ttest",
    "wilcoxon_signed_rank",
    # dataset
    "EpisodeGenerator",
    "GenerationConfig",
    "DatasetStore",
    "Labeler",
    "EpisodeLabel",
    "save_episode",
    "load_episode",
]
