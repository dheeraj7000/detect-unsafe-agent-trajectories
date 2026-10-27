"""GraphGuard Evaluator — Phase 5 study framework.

Matched-FPR early-detection experiments, H1–H5 hypothesis testing, ablations,
counterfactual replay, and statistical comparison with confidence intervals.
"""

from .study import StudyRunner, StudyResult, SeedAggregate, summarize_seed_results
from .statistics import bootstrap_ci, paired_ttest, wilcoxon_signed_rank
from .ablations import AblationConfig, DEFAULT_ABLATIONS, run_ablation_study
from .counterfactual import (counterfactual_replay, CounterfactualResult,
                             remove_edge_type_transform,
                             remove_sends_to_external_transform,
                             sanitize_tool_output_transform)
from .protocol import freeze_experiment, verify_frozen_experiment

__all__ = [
    "StudyRunner",
    "StudyResult",
    "SeedAggregate",
    "summarize_seed_results",
    "bootstrap_ci",
    "paired_ttest",
    "wilcoxon_signed_rank",
    "AblationConfig",
    "DEFAULT_ABLATIONS",
    "run_ablation_study",
    "counterfactual_replay",
    "CounterfactualResult",
    "remove_edge_type_transform",
    "remove_sends_to_external_transform",
    "sanitize_tool_output_transform",
    "freeze_experiment",
    "verify_frozen_experiment",
]
