# PREREG: flsentinel evaluation

Locked 2026-10-28, before any formal experiment run. Detector rules, scenario
definitions, seeds, and bars below are fixed. A mechanics smoke seed (7) is
used while writing code; it is **not** part of the formal seed set, and no
detector constant may be changed after the formal runs start. Misses are
reported as measured.

## Simulator (fixed)

- Fully synthetic multiclass Gaussian blobs: 6 classes, 24 features.
- 10 clients; class-concentrated non-IID split (each class sends 70% of its
  samples to a rotating primary group of 4 clients, remainder Dirichlet
  alpha=0.5). ~500 train samples per client.
- Model: softmax regression (numpy), full-batch local GD, 3 local epochs,
  base lr 0.5. FedAvg weighted by client sample count.
- 45 rounds per run. Server-side IID validation set (1,200 samples) exists
  **only** for the accuracy watcher baseline and for scoring; flsentinel
  never sees it.
- Per-round stream: val_accuracy, mean/std update norm, mean pairwise cosine
  similarity between client updates, per-client losses (global model's loss
  on each client's local train data, measured at round start), active
  client count.

## Scenarios (collapse onset round 15 in all)

1. `clean` — no fault.
2. `labelflip_ramp` — from round 15, one additional client per round (up to
   5 of 10) permanently flips its labels by a fixed cyclic permutation.
3. `lr_blowup` — from round 15, every client's local lr is multiplied by
   1.35 each round (compounding).
4. `dropout_cascade` — from round 15, one additional client per round (up
   to 5) permanently drops out; drop order is descending local share of the
   two globally rarest classes (targeted loss of class coverage).

## Detectors (fixed rules)

- **AccuracyDropWatcher**: fires at the first round t with
  val_acc[t] < max(val_acc[0..t]) − 0.05.
- **NormThresholdWatcher**: threshold = 3.0 × median(mean update norm,
  rounds 0–4); fires at the first t > 4 above threshold.
- **FLSentinel**: channels per round — log mean update norm, cosine drift
  (1 − mean pairwise cosine), per-client loss IQR, active-client count
  (one-sided, drops only). Each channel is z-scored against a trailing
  window of the previous 8 rounds (rounds 0–4 seed the first window;
  robust centre/scale = median/MAD with floored scale). Composite =
  max |z| across channels (participation one-sided). Fires when composite
  ≥ 3.0 on 2 consecutive rounds; warning round = the second of the two.
  No warning before round 6.

## Formal protocol

- Seeds 101, 102, 103, 104, 105 for every scenario (20 runs).
- Lead time for a run = watcher warning round − sentinel warning round.
  If the sentinel fires and the watcher never fires, lead = (44 − sentinel
  round). If the sentinel never fires on a collapse run, that seed counts
  as a miss (no lead).

## Bars

- **B1 (primary):** median sentinel lead over the accuracy watcher ≥ **3
  rounds** on at least **2 of 3** collapse modes.
- **B2 (false alarms):** **zero** sentinel warnings across all 5 clean runs
  (225 clean rounds).
- **B3 (secondary, reported, not gated):** median sentinel lead over the
  norm-threshold watcher, per mode.

Whatever the outcome, results.json and the README table report the measured
per-mode medians, per-seed warning rounds, and any misses.

## Amendment (2026-10-28, before formal runs)

Pilot mechanics runs (seed 7 only) showed the original difficulty constants
(class-mean scale 2.4, lr 0.5, 3 local epochs) saturate validation accuracy
at 1.000 within a few rounds: gradients vanish, so the lr_blowup and
dropout_cascade faults produced *no accuracy collapse at all* — a benchmark
in which the fault causes no failure measures nothing. Simulator difficulty
constants only were changed: class-mean scale 1.15, base lr 0.35, 2 local
epochs, the two rarest classes get 1.35x within-class scale. Bars, seeds,
detector rules, onset, and all scenario schedules are unchanged.

## Amendment 2 (2026-10-28, before formal runs)

Further pilot mechanics (seeds 7–9): (a) dropout_cascade still produced no
collapse — with a 30% spread of every class to every client, no class is
ever orphaned, and FedAvg correctly routes around the lost clients;
(b) seeds whose clean run saturated at accuracy 1.000 delayed the
lr_blowup statistical signature. Simulator-only changes: rare classes are
now site-concentrated (100% of samples in their primary client group — the
canonical rare-cohort-at-few-sites setting), class-mean scale 0.95, common
class scale 1.1, rare class scale 1.45. Detector rules, bars, seeds, onset
and schedules unchanged. No further simulator changes after this point.

## Amendment 3 (2026-10-28, before formal runs)

Pilot mechanics (seeds 7–11): orphan-class forgetting after the dropout
cascade is real but slow — by round 45 accuracy had fallen only ~3–4pp and
had not crossed the accuracy watcher's 5pp threshold, so dropout lead times
would be entirely horizon-censored. Protocol lengthened: 70 rounds per run
(onset unchanged at 15), lead censoring constant becomes (69 − sentinel
round), clean-run exposure becomes 5 × 70 = 350 rounds. Pilot medians at
this point, for the record: labelflip lead ~1–2 rounds, lr_blowup lead
~5–15 rounds, clean sentinel silent on 5/5 pilot seeds. Bars, detector
rules, seeds and schedules unchanged. This is the final amendment; the
formal runs follow.
