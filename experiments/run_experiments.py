#!/usr/bin/env python3
"""Run the PREREG evaluation and write results/results.json.

Usage:  python experiments/run_experiments.py
Works from a fresh clone without installing (adds ../src to sys.path);
an installed `flsentinel` package takes precedence if present.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flsentinel.experiment import evaluate  # noqa: E402


def main() -> None:
    results = evaluate()
    out = os.path.join(os.path.dirname(__file__), "..", "results", "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")
    for scenario, s in results["summary"].items():
        print(f"  {scenario}: {s}")
    print(f"  verdict: {results['verdict']}")


if __name__ == "__main__":
    main()
