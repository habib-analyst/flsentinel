[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

# flsentinel

**Federated-learning collapse early warning, from update statistics alone.**

By the time validation accuracy admits an FL run is dying, the cheap rounds
to save it are usually gone. flsentinel watches what the server already
sees every round — update norms, pairwise cosine drift between client
updates, per-client loss divergence, participation — and warns *before*
accuracy craters. It never touches a validation set, and it does not filter
or exclude anyone: it is a smoke alarm, not a sprinkler.

It ships with a compact self-contained FL simulator (numpy, fully synthetic
data, nothing to download) that induces three collapse modes plus clean
runs, and scores every detector in **rounds of warning gained** against two
naive baselines on identical runs.

## Quickstart

```bash
git clone https://github.com/habib-analyst/flsentinel.git && cd flsentinel
pip install -e .            # or just run from the tree, numpy is the only dep
python experiments/run_experiments.py   # reproduces results/results.json (~7 s)
pytest -q                   # 17 tests
```

```python
from flsentinel import run_scenario, flsentinel

run = run_scenario("lr_blowup", seed=101)
print(flsentinel(run["stream"]))   # -> 18  (first warning round)
```

## Measured results (formal protocol, seeds 101–105, onset at round 15 of 70)

Pre-registered bars (see [PREREG.md](PREREG.md)): median lead ≥ 3 rounds
over the accuracy watcher on ≥ 2 of 3 collapse modes (**B1**), and zero
false alarms on clean runs (**B2**). Both bars met.

| scenario | sentinel warns | accuracy watcher | norm watcher | median lead vs accuracy |
|---|---|---|---|---|
| clean (5 runs, 350 rounds) | never | never | never | **0 false alarms** |
| label-flip poisoning ramp | round 16 (5/5) | rounds 17–18 | never | **2 rounds** |
| learning-rate blowup | round 18 (5/5) | rounds 23–27 | rounds 23–26 | **6 rounds** |
| client dropout cascade | round 17 (5/5) | rounds 56, 68, never ×3 | never | **52 rounds** (censored, see limits) |

Read the two honest rows carefully:

- **Poisoning is the sentinel's weakest mode.** Against an abrupt label-flip
  ramp, accuracy itself is a fast signal; the sentinel's lead is only
  1–2 rounds. (The norm watcher never fires at all here — flipped updates
  have normal norms.)
- **The dropout lead is horizon-censored.** Orphan-class accuracy erodes
  slowly (end of run: −2 to −6 points). In 3 of 5 seeds the accuracy
  watcher's 5-point threshold was never crossed within 70 rounds, so the
  lead is recorded as rounds remaining after the sentinel's round-17
  warning. In the 2 seeds where it did fire, the leads were 39 and 51
  rounds. The claim is narrow and real: a 5-point accuracy watcher is
  blind to slow erosion for a very long time; update statistics are not.

The baselines are not strawmen: the accuracy watcher is *given* a server
validation set that flsentinel never sees, and still loses on 2 of 3 modes.

## How it works

Each round produces a stream record: mean/std update norm, mean pairwise
cosine similarity between client updates, per-client losses of the global
model, active client count, and (for scoring only) validation accuracy.

- **AccuracyDropWatcher** — fires when validation accuracy falls 5 points
  below its running peak. Needs server-held labelled data.
- **NormThresholdWatcher** — fires when the mean update norm exceeds 3× its
  calibration median (rounds 0–4).
- **FLSentinel** — four channels (log norm, cosine drift, loss IQR,
  participation drop), each z-scored against a trailing 8-round window with
  robust median/MAD scaling and floored scales; fires when the composite
  (max channel z) stays ≥ 3.0 for 2 consecutive rounds. Rules were locked
  in [PREREG.md](PREREG.md) before the formal runs; the three dated
  amendments there changed only simulator difficulty constants after pilot
  runs showed two faults initially produced no failure to detect.

## The simulation world

Fully synthetic 6-class Gaussian blobs (24 features), 10 clients, softmax
regression, FedAvg. The two rarest classes are site-concentrated — all of
their samples live at a primary group of 4 clients, the rare-cohort-at-a-
few-sites setting — which is what makes the dropout cascade an orphan-class
failure rather than a shrug. Collapse modes: label-flip ramp (one more
client flips per round from round 15, up to 5), LR blowup (local lr ×1.35
compounding from round 15), dropout cascade (one more client per round
leaves permanently, rarest-class holders first). The world mirrors
[fedsec-sim](https://github.com/habib-analyst/fedsec-sim)'s testbed;
flsentinel is the monitoring module that world was missing.

## Using it on your own FL loop

Any training loop that can emit, per round, the update vectors (or their
norms + pairwise cosines), per-client losses, and the participant count
can feed `flsentinel.detectors.flsentinel(stream)`. Records are plain
dicts; see `flsentinel/simulator.py` for the exact schema.

## Limits, read before citing

- Toy scale: softmax regression on synthetic blobs, 10 clients, full
  participation except in the cascade. Deep nets, partial participation,
  and heterogeneous round times will move the numbers.
- Under **secure aggregation** the server cannot see per-client updates at
  all, so the per-client channels are unavailable by design; only
  participation and aggregate-norm variants could run there.
- The z ≥ 3 / 2-consecutive rule is a fixed heuristic, not a calibrated
  probability. Thresholds were not tuned per scenario, and the poisoning
  row shows what that costs.
- A warning is not a diagnosis: the sentinel says *something is breaking*,
  not *what*. Pair it with a mitigation (robust aggregation, lr rollback,
  client quarantine) of your choice.

Prior-art position and differentiation: [KILLGATE.md](KILLGATE.md).
Full writeup: PAPER.md in the project papers collection.
