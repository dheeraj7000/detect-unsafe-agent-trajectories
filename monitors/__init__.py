"""GraphGuard monitors — baseline and proposed early-warning detectors.

All monitors share a common interface:

    monitor = SomeMonitor(name="...")
    score = monitor.score_prefix(trace_prefix, step, graph=graph)
    alert = monitor.is_alert(score, fpr_level=0.05)

Monitors are calibrated on a validation set to hit target false-positive
rates (1% and 5%) via :mod:`monitors.calibrator`.
"""

from .base import BaseMonitor, MonitorOutput
from .calibrator import calibrate_all, calibrate_monitor, evaluate_episode
from .feature import FeatureMonitor
from .registry import (MonitorRegistry, make_baselines, make_all_monitors,
                       make_monitor)
from .rules import RulesMonitor
from .sequence import SequenceMonitor
from .temporal import TemporalSequenceMonitor

__all__ = [
    "BaseMonitor",
    "MonitorOutput",
    "calibrate_all",
    "calibrate_monitor",
    "evaluate_episode",
    "FeatureMonitor",
    "MonitorRegistry",
    "RulesMonitor",
    "SequenceMonitor",
    "TemporalSequenceMonitor",
    "make_monitor",
    "make_all_monitors",
    "make_baselines",
]
