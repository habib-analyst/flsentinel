# Kill-gate: flsentinel

**Question:** does an established FL collapse / divergence *early-warning* detector
with released code already exist — one that watches update statistics and warns
*before* accuracy craters, scored by warning lead time?

**Verdict: PASS — no equivalent found. Build proceeds.**

## What the search found (2026-10-28)

### 1. Per-round Byzantine detection and robust aggregation (the crowded shelf)
Krum, Bulyan, coordinate median / trimmed mean, FLAME, FoolsGold, Deepsight,
RFLBAT, and FLDetector (Zhang et al., code at github.com/zaixizhang/FLDetector)
classify or filter individual client updates **in the round they arrive**, so a
robust aggregator can mitigate poisoning. FLDetector is the closest neighbour:
it tracks each client's update consistency across rounds. But its output is a
malicious-client verdict followed by exclusion and a training restart. It is
not an early-warning system:

- no forecast horizon ("collapse in k rounds"), no warning lead-time metric;
- poisoning only — it says nothing about learning-rate blowup or dropout
  cascades, which are operational failures, not attacks;
- success is measured as detection accuracy / post-exclusion accuracy, never
  as rounds of warning gained over an accuracy watcher.

### 2. Federated anomaly detection *as a task*
FedAD and similar repos use FL to *train* anomaly detectors on client data.
The federation is the method, not the thing being monitored. Unrelated.

### 3. Client data-drift detection
FL-MalDrift and ADWIN-style client drift detectors flag distribution change in
client *data*. Different signal (data, not update statistics), different
decision (retrain / reweight), no collapse lead time.

### 4. Monitoring advice in courses and blogs
FL course material (e.g. apxml's monitoring chapter) and engineering blogs
advise operators to eyeball update norms and cosine similarities. Prose, no
released detector, no benchmark, no false-alarm budget, no lead-time numbers.

### 5. Server-side validation / update rejection
Schemes that validate each aggregated model on server-held data and reject bad
rounds are accuracy-based by construction — they are the naive baseline class
flsentinel is compared against, not prior art for update-statistic warning.

## Differentiation (what flsentinel is)

1. **Forecasting framing, lead-time metric.** Every detector is scored in
   *rounds of warning gained* against an accuracy-drop watcher and a
   norm-threshold watcher on identical runs — a comparison none of the above
   publishes.
2. **Update statistics only.** Norms, pairwise cosine drift, per-client loss
   divergence, participation. No server validation set (the accuracy watcher
   baseline is deliberately given that extra information and still loses if
   the bars hold).
3. **Adversarial + operational collapse.** Label-flip poisoning ramp,
   learning-rate blowup, and client dropout cascade, plus a zero-false-alarm
   requirement on clean runs.
4. **A module, not a mitigation.** flsentinel warns; it does not filter,
   exclude, or aggregate. It is designed to sit alongside a simulator /
   deployment such as fedsec-sim and hand a human (or an autoscaler) time to
   act.

## Residual risk

"Early warning for training instability" exists in centralised deep learning
as folklore and internal tooling at large labs; nothing released was found
that does it for federated rounds with a lead-time benchmark. If a maintained
equivalent surfaces, this repo's benchmark harness is the part that remains
useful.
