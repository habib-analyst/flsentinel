"""Formal evaluation per PREREG.md: scenarios x seeds, all detectors on
identical streams, warning lead times in rounds."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .detectors import accuracy_drop_watcher, flsentinel, norm_threshold_watcher
from .simulator import TOTAL_ROUNDS, run_scenario

SCENARIOS = ["clean", "labelflip_ramp", "lr_blowup", "dropout_cascade"]
FORMAL_SEEDS = [101, 102, 103, 104, 105]


def _lead(watcher: Optional[int], sentinel: Optional[int],
          last_round: int) -> Optional[int]:
    """Rounds of warning gained. None if the sentinel never fired."""
    if sentinel is None:
        return None
    if watcher is None:
        return int(last_round - sentinel)
    return int(watcher - sentinel)


def evaluate(seeds: List[int] = FORMAL_SEEDS) -> Dict:
    rows = []
    for scenario in SCENARIOS:
        for seed in seeds:
            run = run_scenario(scenario, seed)
            stream = run["stream"]
            last = int(stream[-1]["round"])
            acc_w = accuracy_drop_watcher(stream)
            norm_w = norm_threshold_watcher(stream)
            sent = flsentinel(stream)
            rows.append({
                "scenario": scenario,
                "seed": seed,
                "final_val_accuracy": round(stream[-1]["val_accuracy"], 4),
                "peak_val_accuracy": round(max(r["val_accuracy"] for r in stream), 4),
                "accuracy_watcher_round": acc_w,
                "norm_watcher_round": norm_w,
                "sentinel_round": sent,
                "lead_vs_accuracy": _lead(acc_w, sent, last),
                "lead_vs_norm": _lead(norm_w, sent, last),
            })

    summary = {}
    for scenario in SCENARIOS:
        sub = [r for r in rows if r["scenario"] == scenario]
        if scenario == "clean":
            summary[scenario] = {
                "sentinel_false_alarms": sum(1 for r in sub if r["sentinel_round"] is not None),
                "runs": len(sub),
            }
        else:
            leads = [r["lead_vs_accuracy"] for r in sub if r["lead_vs_accuracy"] is not None]
            leads_n = [r["lead_vs_norm"] for r in sub if r["lead_vs_norm"] is not None]
            summary[scenario] = {
                "runs": len(sub),
                "sentinel_fired": sum(1 for r in sub if r["sentinel_round"] is not None),
                "median_lead_vs_accuracy": float(np.median(leads)) if leads else None,
                "median_lead_vs_norm": float(np.median(leads_n)) if leads_n else None,
                "leads_vs_accuracy": leads,
            }

    modes_passing = [
        s for s in SCENARIOS[1:]
        if summary[s]["median_lead_vs_accuracy"] is not None
        and summary[s]["median_lead_vs_accuracy"] >= 3
    ]
    return {
        "config": {
            "rounds": TOTAL_ROUNDS,
            "seeds": seeds,
            "scenarios": SCENARIOS,
            "bars": {
                "B1": "median lead vs accuracy watcher >= 3 rounds on >= 2 of 3 collapse modes",
                "B2": "zero sentinel warnings on clean runs",
            },
        },
        "rows": rows,
        "summary": summary,
        "verdict": {
            "B1_modes_passing": modes_passing,
            "B1_pass": len(modes_passing) >= 2,
            "B2_pass": summary["clean"]["sentinel_false_alarms"] == 0,
        },
    }
