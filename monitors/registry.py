"""Monitor registry and factory functions.

Provides a central place to instantiate all GraphGuard monitors (baselines
and the proposed graph monitor) so experiments can be run apples-to-apples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseMonitor
from .feature import FeatureMonitor
from .rules import RulesMonitor
from .sequence import SequenceMonitor
from .temporal import TemporalSequenceMonitor


class MonitorRegistry:
    """Registry of monitor constructors."""

    def __init__(self):
        self._constructors: Dict[str, Any] = {
            "rules": RulesMonitor,
            "feature": FeatureMonitor,
            "sequence": SequenceMonitor,
            "temporal": TemporalSequenceMonitor,
        }

    def register(self, name: str, constructor: Any) -> None:
        self._constructors[name] = constructor

    def make(self, name: str, **kwargs) -> BaseMonitor:
        if name not in self._constructors:
            raise KeyError(f"Unknown monitor: {name}. Available: {list(self._constructors)}")
        return self._constructors[name](**kwargs)

    def names(self) -> List[str]:
        return list(self._constructors.keys())


# Global default registry.
default_registry = MonitorRegistry()


def make_monitor(name: str, **kwargs) -> BaseMonitor:
    """Build a single monitor by name."""
    return default_registry.make(name, **kwargs)


def make_all_monitors() -> List[BaseMonitor]:
    """Build one instance of every registered monitor."""
    return [default_registry.make(n) for n in default_registry.names()]


def make_baselines() -> List[BaseMonitor]:
    """Build the ordinary-trajectory baselines (Phase 3)."""
    return [
        default_registry.make("rules"),
        default_registry.make("feature"),
        default_registry.make("sequence", window=None),
        default_registry.make("temporal", window=10),
    ]
