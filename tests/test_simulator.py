import numpy as np

from flsentinel.simulator import (
    BASE_LR,
    N_CLIENTS,
    dropped_at_round,
    generate_world,
    lr_at_round,
    poisoned_at_round,
    run_scenario,
)


def test_world_deterministic():
    w1 = generate_world(7)
    w2 = generate_world(7)
    assert [len(c.y) for c in w1.clients] == [len(c.y) for c in w2.clients]
    assert w1.drop_order == w2.drop_order
    assert np.allclose(w1.clients[0].X, w2.clients[0].X)


def test_rare_classes_site_concentrated():
    w = generate_world(7)
    for c in w.rare_classes:
        primary = {(2 * c + i) % N_CLIENTS for i in range(4)}
        for i, client in enumerate(w.clients):
            n_c = int((client.y == c).sum())
            if i not in primary:
                assert n_c == 0
            else:
                assert n_c > 0


def test_schedules():
    assert poisoned_at_round("labelflip_ramp", 14) == set()
    assert poisoned_at_round("labelflip_ramp", 15) == {0}
    assert poisoned_at_round("labelflip_ramp", 19) == {0, 1, 2, 3, 4}
    assert poisoned_at_round("labelflip_ramp", 40) == {0, 1, 2, 3, 4}
    assert poisoned_at_round("clean", 20) == set()
    order = list(range(10))
    assert dropped_at_round("dropout_cascade", 14, order) == set()
    assert dropped_at_round("dropout_cascade", 16, order) == {0, 1}
    assert dropped_at_round("dropout_cascade", 40, order) == {0, 1, 2, 3, 4}
    assert abs(lr_at_round("lr_blowup", 15) - BASE_LR * 1.35) < 1e-12
    assert lr_at_round("lr_blowup", 14) == BASE_LR
    assert lr_at_round("clean", 30) == BASE_LR


def test_run_deterministic():
    a = run_scenario("clean", 7)["stream"]
    b = run_scenario("clean", 7)["stream"]
    assert [r["val_accuracy"] for r in a] == [r["val_accuracy"] for r in b]


def test_stream_schema_and_clean_convergence():
    stream = run_scenario("clean", 7)["stream"]
    assert len(stream) == 70
    for rec in stream:
        assert rec["n_active"] == N_CLIENTS
        assert len(rec["client_losses"]) == rec["n_active"]
        assert 0.0 <= rec["val_accuracy"] <= 1.0
    assert stream[-1]["val_accuracy"] >= 0.90
