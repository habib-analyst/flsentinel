from flsentinel.detectors import (
    accuracy_drop_watcher,
    flsentinel,
    norm_threshold_watcher,
)


def make_stream(n=30, acc=0.9, norm=1.0, cos=0.8, n_active=10):
    return [{
        "round": t,
        "val_accuracy": acc,
        "update_norm_mean": norm,
        "update_norm_std": 0.1,
        "mean_pairwise_cosine": cos,
        "client_losses": [0.30, 0.32, 0.28, 0.31],
        "n_active": n_active,
    } for t in range(n)]


def test_accuracy_watcher_fires_on_drop():
    stream = make_stream()
    for t, a in [(0, 0.90), (1, 0.92), (2, 0.94), (3, 0.88)]:
        stream[t]["val_accuracy"] = a
    assert accuracy_drop_watcher(stream) == 3


def test_accuracy_watcher_silent_when_stable():
    assert accuracy_drop_watcher(make_stream()) is None


def test_norm_watcher_fires_on_spike():
    stream = make_stream()
    stream[10]["update_norm_mean"] = 4.0
    assert norm_threshold_watcher(stream) == 10


def test_norm_watcher_silent_when_flat():
    assert norm_threshold_watcher(make_stream()) is None


def test_sentinel_silent_on_flat_stream():
    assert flsentinel(make_stream(n=40)) is None


def test_sentinel_fires_on_cosine_collapse():
    stream = make_stream(n=30)
    for t in range(10, 30):
        stream[t]["mean_pairwise_cosine"] = 0.2
    w = flsentinel(stream)
    assert w is not None and 10 <= w <= 13


def test_sentinel_fires_on_participation_drop():
    stream = make_stream(n=30)
    for t in range(10, 30):
        stream[t]["n_active"] = 8
    w = flsentinel(stream)
    assert w is not None and 10 <= w <= 13


def test_sentinel_needs_sustained_anomaly():
    stream = make_stream(n=30)
    stream[12]["mean_pairwise_cosine"] = 0.0  # single-round blip
    assert flsentinel(stream) is None
