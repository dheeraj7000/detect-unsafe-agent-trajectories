"""Study runner for Phase 5.

Orchestrates the matched-FPR early-detection study:

1. Fit each monitor on training prefixes.
2. Calibrate thresholds on benign calibration traces (FPR = 1% and 5%).
3. Evaluate each monitor on test adversarial episodes at every prefix.
4. Aggregate detection metrics per monitor.
5. Statistical comparison: graph monitor vs each sequence baseline on lead
   time (paired t-test, Wilcoxon signed-rank, bootstrap CI for difference).
6. H1–H5 hypothesis evaluation summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from monitors import BaseMonitor, calibrate_monitor, make_baselines, make_proposed
from monitors.calibrator import evaluate_episode
from evaluator.statistics import (
    bootstrap_ci,
    ci_for_difference,
    detect_rate,
    paired_ttest,
    wilcoxon_signed_rank,
)


@dataclass
class MonitorMetrics:
    """Aggregated metrics for a single monitor."""

    name: str
    n_episodes: int
    detect_rate: float
    avg_lead: float
    avg_norm_time: float
    ci_lead: Optional[Tuple[float, float, float]]
    ci_norm_time: Optional[Tuple[float, float, float]]
    per_episode: List[dict] = field(default_factory=list)


@dataclass
class StatisticalComparison:
    """Statistical comparison of graph monitor vs a baseline."""

    baseline: str
    graph_lead_mean: float
    baseline_lead_mean: float
    mean_difference: float  # graph - baseline (positive = graph earlier)
    ci_difference: Optional[Tuple[float, float, float]]
    paired_ttest: Optional[Tuple[float, float]]  # (stat, p)
    wilcoxon: Optional[Tuple[float, float]]  # (stat, p)
    graph_detect_rate: float
    baseline_detect_rate: float
    h1_supported: bool  # graph earlier at matched FPR (CI for diff excludes 0)


@dataclass
class StudyResult:
    """Full results of the Phase 5 study."""

    fpr_level: float
    n_test_episodes: int
    threat_families: List[str]
    monitor_metrics: List[MonitorMetrics]
    comparisons: List[StatisticalComparison]
    h1_supported: bool
    h2_notes: str
    h3_notes: str
    h4_notes: str
    h5_notes: str


@dataclass
class SeedAggregate:
    fpr_level: float
    n_seeds: int
    graph_avg_lead: float
    graph_lead_ci: Optional[Tuple[float, float, float]]
    sequence_avg_lead: float
    sequence_lead_ci: Optional[Tuple[float, float, float]]
    lead_difference: float
    lead_difference_ci: Optional[Tuple[float, float, float]]
    graph_detect_rate: float
    sequence_detect_rate: float
    h1_supported_seed_fraction: float


def summarize_seed_results(
    results_by_seed: List[StudyResult],
) -> List[SeedAggregate]:
    """Aggregate one-fpr study results across independent random seeds."""
    fpr_levels = sorted({result.fpr_level for result in results_by_seed})
    aggregates = []
    for fpr in fpr_levels:
        runs = [result for result in results_by_seed if result.fpr_level == fpr]
        graph_leads = []
        sequence_leads = []
        graph_detect_rates = []
        sequence_detect_rates = []
        h1_flags = []
        for result in runs:
            graph = next(
                (metric for metric in result.monitor_metrics
                 if metric.name == "graph"),
                None,
            )
            sequence = next(
                (metric for metric in result.monitor_metrics
                 if metric.name == "sequence"),
                None,
            )
            comparison = next(
                (item for item in result.comparisons
                 if item.baseline == "sequence"),
                None,
            )
            if graph is None or sequence is None:
                continue
            graph_leads.append(graph.avg_lead)
            sequence_leads.append(sequence.avg_lead)
            graph_detect_rates.append(graph.detect_rate)
            sequence_detect_rates.append(sequence.detect_rate)
            h1_flags.append(bool(comparison and comparison.h1_supported))

        if not graph_leads:
            continue
        differences = [
            graph_lead - sequence_lead
            for graph_lead, sequence_lead in zip(graph_leads, sequence_leads)
        ]
        aggregates.append(SeedAggregate(
            fpr_level=fpr,
            n_seeds=len(graph_leads),
            graph_avg_lead=float(np.mean(graph_leads)),
            graph_lead_ci=bootstrap_ci(graph_leads, seed=42),
            sequence_avg_lead=float(np.mean(sequence_leads)),
            sequence_lead_ci=bootstrap_ci(sequence_leads, seed=42),
            lead_difference=float(np.mean(differences)),
            lead_difference_ci=bootstrap_ci(differences, seed=42),
            graph_detect_rate=float(np.mean(graph_detect_rates)),
            sequence_detect_rate=float(np.mean(sequence_detect_rates)),
            h1_supported_seed_fraction=float(np.mean(h1_flags)),
        ))
    return aggregates


class StudyRunner:
    """Runs the matched-FPR early-detection study."""

    def __init__(
        self,
        fpr_levels: Tuple[float, ...] = (0.01, 0.05),
        seed: int = 42,
    ):
        self.fpr_levels = fpr_levels
        self.seed = seed

    def run(
        self,
        train_episodes: List[dict],
        benign_calibration_traces: List[Tuple[List[Any], int]],
        test_episodes: List[dict],
        threat_families: List[str],
        include_graph: bool = True,
        monitors: Optional[List[BaseMonitor]] = None,
    ) -> List[StudyResult]:
        """Run the full study at each FPR level.

        Args:
            train_episodes: labeled training prefixes (used to fit monitors).
                Each: dict with trace, total_steps, intervention_step,
                episode_id, plus "prefixes" key -> list of
                (trace_prefix, step, graph, label).
            benign_calibration_traces: benign traces for threshold calibration.
            test_episodes: test adversarial episodes. Each: dict with
                trace, total_steps, intervention_step, episode_id.
            threat_families: list of threat family names in test_episodes.
            include_graph: whether to include the graph monitor.

        Returns:
            List of StudyResult, one per FPR level.
        """
        results = []

        # Build monitors.
        use_supplied_monitors = monitors is not None
        monitors = (
            list(monitors) if monitors is not None else make_baselines(self.seed)
        )
        if include_graph and not any(m.name == "graph" for m in monitors):
            monitors.append(make_proposed(self.seed))

        if not use_supplied_monitors and not train_episodes:
            raise ValueError("Provide training episodes or fitted monitor instances.")
        if train_episodes:
            training_ids = {
                episode["episode_id"] for episode in train_episodes
            }
            test_ids = {episode["episode_id"] for episode in test_episodes}
            calibration_ids = {
                trace[0].episode_id
                for trace, _ in benign_calibration_traces
                if trace
            }
            if training_ids & test_ids or calibration_ids & (training_ids | test_ids):
                raise ValueError(
                    "Training, calibration, and test episode IDs must be disjoint."
                )
            train_templates = {
                episode["template_id"]
                for episode in train_episodes if episode.get("template_id")
            }
            test_templates = {
                episode["template_id"]
                for episode in test_episodes if episode.get("template_id")
            }
            if train_templates & test_templates:
                raise ValueError(
                    "Attack templates must be disjoint between train and test."
                )

        # Fit all monitors on training data (skip if already fitted).
        for m in monitors:
            if hasattr(m, "fit") and not getattr(m, "_fitted", False):
                if train_episodes:
                    examples = []
                    for ep in train_episodes:
                        for ex in ep.get("prefixes", []):
                            examples.append(ex)
                    if examples:
                        m.fit(examples)
                if (hasattr(m, "fit")
                        and not getattr(m, "_fitted", False)):
                    raise ValueError(
                        f"Monitor {m.name!r} is unfitted and has no training examples."
                    )

        for fpr in self.fpr_levels:
            result = self._run_at_fpr(
                fpr, monitors, benign_calibration_traces, test_episodes,
                threat_families,
            )
            results.append(result)

        return results

    def _run_at_fpr(
        self,
        fpr: float,
        monitors: List,
        benign_calibration_traces: List[Tuple[List[Any], int]],
        test_episodes: List[dict],
        threat_families: List[str],
    ) -> StudyResult:
        """Run the study at a single FPR level."""
        # Calibrate all monitors on benign traces.
        for m in monitors:
            calibrate_monitor(m, benign_calibration_traces,
                              target_fprs=(fpr,))

        # Evaluate each monitor on each test episode.
        monitor_metrics: List[MonitorMetrics] = []
        for m in monitors:
            per_episode = []
            lead_values = []
            norm_time_values = []
            detected_flags = []
            for ep in test_episodes:
                trace = ep["trace"]
                steps = ep["total_steps"]
                intervention = ep["intervention_step"]
                episode_id = ep["episode_id"]

                def graph_getter(step, tr=trace, eid=episode_id):
                    from harness.graph import behavior_graph_at_step

                    return behavior_graph_at_step(tr, step, eid)

                metrics = evaluate_episode(
                    m, trace, steps, intervention, fpr_level=fpr,
                    graph_getter=graph_getter,
                )
                per_episode.append({
                    "episode_id": episode_id,
                    "threat_family": ep.get("threat_family", "unknown"),
                    "detected": metrics.detected,
                    "detection_step": metrics.detection_step,
                    "unsafe_event_step": metrics.unsafe_event_step,
                    "lead_actions": metrics.lead_actions,
                    "normalized_detection_time": metrics.normalized_detection_time,
                    "fpr_level": metrics.fpr_level,
                    "first_alert_score": metrics.first_alert_score,
                })
                if metrics.detected:
                    detected_flags.append(True)
                    if metrics.normalized_detection_time is not None:
                        norm_time_values.append(metrics.normalized_detection_time)
                else:
                    detected_flags.append(False)
                lead_values.append(metrics.lead_actions or 0)

            mm = MonitorMetrics(
                name=m.name,
                n_episodes=len(test_episodes),
                detect_rate=detect_rate(detected_flags),
                avg_lead=float(np.mean(lead_values)) if lead_values else 0.0,
                avg_norm_time=float(np.mean(norm_time_values))
                if norm_time_values else 0.0,
                ci_lead=bootstrap_ci(lead_values, seed=self.seed)
                if lead_values else None,
                ci_norm_time=bootstrap_ci(norm_time_values, seed=self.seed)
                if norm_time_values else None,
                per_episode=per_episode,
            )
            monitor_metrics.append(mm)

        # Statistical comparisons: graph vs each sequence baseline.
        comparisons = []
        graph_metrics = next((m for m in monitor_metrics if m.name == "graph"),
                             None)
        baseline_metrics = [m for m in monitor_metrics
                            if m.name in ("rules", "feature", "sequence",
                                          "temporal", "transformer")]

        # Build paired per-episode lead values for comparison.
        # Pair by episode_id.
        graph_by_ep = {
            p["episode_id"]: p for p in graph_metrics.per_episode
        } if graph_metrics else {}

        for bm in baseline_metrics:
            baseline_by_ep = {
                p["episode_id"]: p for p in bm.per_episode
            }
            common_eps = sorted(set(graph_by_ep.keys()) &
                                set(baseline_by_ep.keys()))

            graph_leads = []
            baseline_leads = []
            for ep_id in common_eps:
                gp = graph_by_ep[ep_id]
                bp = baseline_by_ep[ep_id]
                # Count a miss as zero warning lead rather than dropping it
                # from the paired comparison.
                graph_leads.append(gp["lead_actions"] or 0)
                baseline_leads.append(bp["lead_actions"] or 0)

            ci_diff = None
            tt = None
            wil = None
            if graph_leads:
                ci_diff = ci_for_difference(graph_leads, baseline_leads,
                                            seed=self.seed)
                tt = paired_ttest(graph_leads, baseline_leads)
                wil = wilcoxon_signed_rank(graph_leads, baseline_leads)

            mean_diff = (float(np.mean(graph_leads))
                         - float(np.mean(baseline_leads))) if graph_leads else 0.0
            h1 = (ci_diff is not None
                  and ci_diff[1] > 0)  # CI lower bound > 0 => graph earlier

            comparisons.append(StatisticalComparison(
                baseline=bm.name,
                graph_lead_mean=float(np.mean(graph_leads))
                if graph_leads else 0.0,
                baseline_lead_mean=float(np.mean(baseline_leads))
                if baseline_leads else 0.0,
                mean_difference=mean_diff,
                ci_difference=ci_diff,
                paired_ttest=tt,
                wilcoxon=wil,
                graph_detect_rate=graph_metrics.detect_rate
                if graph_metrics else 0.0,
                baseline_detect_rate=bm.detect_rate,
                h1_supported=h1,
            ))

        sequence_comparison = next(
            (comparison for comparison in comparisons
             if comparison.baseline == "sequence"),
            None,
        )
        h1_supported = bool(
            sequence_comparison
            and sequence_comparison.h1_supported
            and sequence_comparison.graph_detect_rate
            >= sequence_comparison.baseline_detect_rate
        )

        # Hypothesis notes (interpretation, to be filled in by researcher).
        h2_notes = ("Horizon robustness: compare avg_norm_time across episode "
                    "lengths (stratify test_episodes by horizon).")
        h3_notes = ("Structural generalization: train on one attack family, "
                    "test on unseen families (run study per family split).")
        h4_notes = ("Interpretability: check counterfactual replay results — "
                    "risk should localize to provenance/auth violation paths.")
        h5_notes = ("Hybrid value: compare graph+sequence+rules Pareto frontier "
                    "vs single representations.")

        return StudyResult(
            fpr_level=fpr,
            n_test_episodes=len(test_episodes),
            threat_families=threat_families,
            monitor_metrics=monitor_metrics,
            comparisons=comparisons,
            h1_supported=h1_supported,
            h2_notes=h2_notes,
            h3_notes=h3_notes,
            h4_notes=h4_notes,
            h5_notes=h5_notes,
        )
