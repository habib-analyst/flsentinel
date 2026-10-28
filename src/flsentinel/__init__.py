"""flsentinel: federated-learning collapse early warning from update
statistics alone."""

from .detectors import accuracy_drop_watcher, flsentinel, norm_threshold_watcher
from .simulator import run_scenario

__all__ = [
    "accuracy_drop_watcher",
    "flsentinel",
    "norm_threshold_watcher",
    "run_scenario",
]
__version__ = "0.1.0"
