"""Collapse detectors over a per-round FL stream.

All detectors consume the same stream records produced by
``flsentinel.simulator.run_scenario`` and return the first warning round
(or ``None``). The two baselines are deliberately naive:

- ``accuracy_drop_watcher`` needs a server validation set (extra
  information the sentinel does not get) and can only react after damage
  is already visible in accuracy.
- ``norm_threshold_watcher`` watches a single statistic against a fixed
  multiple of its calibration median.

``flsentinel`` combines four update-statistic channels, each z-scored
against a trailing window with robust centre/scale, and fires only on a
sustained composite anomaly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# Floors keep a channel's scale from collapsing to zero when the trailing
# window is very quiet (units are per-channel; see PREREG.md).
_CHANNEL_FLOORS = {
    "log_norm": 0.05,     # ~5% change in mean update norm
    "cos_drift": 0.01,    # 1 point of (1 - cosine)
    "loss_iqr": 0.01,     # nats of cross-client loss spread
    "participation": 0.5,  # half a client
}


def accuracy_drop_watcher(stream: List[Dict], drop: float = 0.05) -> Optional[int]:
    """First round where val accuracy falls `drop` below its running peak."""
    peak = -np.inf
    for rec in stream:
        peak = max(peak, rec["val_accuracy"])
        if rec["val_accuracy"] < peak - drop:
            return int(rec["round"])
    return None


def norm_threshold_watcher(stream: List[Dict], calib_rounds: int = 5,
                           multiplier: float = 3.0) -> Optional[int]:
    """First round after calibration with mean update norm above
    `multiplier` x the calibration median."""
    calib = [r["update_norm_mean"] for r in stream[:calib_rounds]]
    threshold = multiplier * float(np.median(calib))
    for rec in stream[calib_rounds:]:
        if rec["update_norm_mean"] > threshold:
            return int(rec["round"])
    return None


def _channels(rec: Dict) -> Dict[str, float]:
    losses = np.array(rec["client_losses"], dtype=float)
    q75, q25 = np.percentile(losses, [75, 25])
    return {
        "log_norm": float(np.log(max(rec["update_norm_mean"], 1e-12))),
        "cos_drift": float(1.0 - rec["mean_pairwise_cosine"]),
        "loss_iqr": float(q75 - q25),
        "participation": float(rec["n_active"]),
    }


def _robust_z(window_vals: List[float], x: float, floor: float) -> float:
    med = float(np.median(window_vals))
    mad = float(np.median(np.abs(np.array(window_vals) - med)))
    scale = max(1.4826 * mad, floor)
    return (x - med) / scale


def sentinel_scores(stream: List[Dict], window: int = 8) -> List[Optional[float]]:
    """Composite anomaly score per round (None until 5 rounds of history).

    Score = max channel |z| vs the trailing window, except participation,
    which is one-sided (only drops count)."""
    chan = [_channels(r) for r in stream]
    scores: List[Optional[float]] = []
    for t in range(len(stream)):
        if t < 5:
            scores.append(None)
            continue
        hist = chan[max(0, t - window):t]
        zmax = 0.0
        for name in ("log_norm", "cos_drift", "loss_iqr"):
            z = _robust_z([h[name] for h in hist], chan[t][name],
                          _CHANNEL_FLOORS[name])
            zmax = max(zmax, abs(z))
        zp = _robust_z([h["participation"] for h in hist],
                       chan[t]["participation"], _CHANNEL_FLOORS["participation"])
        zmax = max(zmax, -zp)  # only a drop in participation is anomalous
        scores.append(float(zmax))
    return scores


def flsentinel(stream: List[Dict], threshold: float = 3.0,
               consecutive: int = 2) -> Optional[int]:
    """First round completing `consecutive` rounds at/above `threshold`.

    Never fires before round 6 (calibration + one confirming round)."""
    scores = sentinel_scores(stream)
    run = 0
    for t, s in enumerate(scores):
        if s is not None and s >= threshold:
            run += 1
            if run >= consecutive and t >= 6:
                return t
        else:
            run = 0
    return None
