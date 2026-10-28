"""Scenario smoke tests on the mechanics seed (7). The formal claim lives
in results/results.json (seeds 101-105); these tests pin the verified
mechanics of each collapse mode."""

from flsentinel.detectors import accuracy_drop_watcher, flsentinel
from flsentinel.simulator import run_scenario

SEED = 7


def test_labelflip_collapses_and_both_fire():
    stream = run_scenario("labelflip_ramp", SEED)["stream"]
    peak = max(r["val_accuracy"] for r in stream)
    assert stream[-1]["val_accuracy"] < peak - 0.20
    assert flsentinel(stream) is not None
    assert accuracy_drop_watcher(stream) is not None


def test_lr_blowup_sentinel_precedes_accuracy_watcher():
    stream = run_scenario("lr_blowup", SEED)["stream"]
    s = flsentinel(stream)
    a = accuracy_drop_watcher(stream)
    assert s is not None and a is not None and s < a


def test_dropout_shrinks_federation_and_sentinel_fires():
    stream = run_scenario("dropout_cascade", SEED)["stream"]
    assert stream[0]["n_active"] == 10
    assert stream[-1]["n_active"] == 5
    assert flsentinel(stream) is not None
    assert stream[-1]["val_accuracy"] < stream[14]["val_accuracy"]


def test_clean_seed7_no_sentinel_warning():
    stream = run_scenario("clean", SEED)["stream"]
    assert flsentinel(stream) is None
    assert accuracy_drop_watcher(stream) is None
