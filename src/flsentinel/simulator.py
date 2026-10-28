"""Compact, self-contained federated-learning simulator.

The simulation world mirrors habib-analyst/fedsec-sim (synthetic Gaussian
blobs, non-IID clients, FedAvg, poisoning) but is implemented from scratch
in numpy so flsentinel runs anywhere with no framework and no downloads:
the data is **fully synthetic**, generated per seed.

Model: softmax regression trained with full-batch gradient descent.
Every run is deterministic given its seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

N_CLASSES = 6
N_FEATURES = 24
N_CLIENTS = 10
# Classes 0 and 1 are the globally rarest; the dropout cascade targets the
# clients that hold them.
CLASS_PRIORS = np.array([0.10, 0.10, 0.20, 0.20, 0.20, 0.20])
ONSET_ROUND = 15
TOTAL_ROUNDS = 70
BASE_LR = 0.35
LOCAL_EPOCHS = 2
SAMPLES_PER_CLIENT = 500
N_VAL = 1200
_WEIGHT_CLIP = 1e4  # numerical safety only; divergence is obvious long before


@dataclass
class Client:
    X: np.ndarray
    y: np.ndarray


@dataclass
class World:
    clients: List[Client]
    val_X: np.ndarray
    val_y: np.ndarray
    drop_order: List[int] = field(default_factory=list)
    rare_classes: Tuple[int, int] = (0, 1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _loss_and_grad(X, y, W, b):
    n = len(X)
    probs = _softmax(X @ W + b)
    loss = -np.log(np.clip(probs[np.arange(n), y], 1e-12, 1.0)).mean()
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y] = 1.0
    grad = (probs - onehot) / n
    return loss, X.T @ grad, grad.sum(axis=0)


def _accuracy(X, y, W, b) -> float:
    return float((np.argmax(X @ W + b, axis=1) == y).mean())


def generate_world(seed: int) -> World:
    """Build the synthetic federation for one seed (deterministic)."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.0, (N_CLASSES, N_FEATURES)) * 0.95
    # The two rarest classes are also the noisiest, so their accuracy is
    # the most fragile and depends on continued training signal.
    rare = tuple(int(c) for c in np.argsort(CLASS_PRIORS)[:2])
    class_scale = np.full(N_CLASSES, 1.1)
    class_scale[list(rare)] = 1.45

    total = SAMPLES_PER_CLIENT * N_CLIENTS
    class_counts = (CLASS_PRIORS * total).astype(int)
    pools = {
        c: rng.normal(means[c], class_scale[c], (class_counts[c], N_FEATURES))
        for c in range(N_CLASSES)
    }

    # Class-concentrated split. Common classes: 70% of samples go to a
    # rotating primary group of 4 clients, the rest spreads Dirichlet(0.5)
    # over everyone. Rare classes are site-concentrated: 100% of their
    # samples live in their primary group (a rare cohort that exists at a
    # few sites only), so losing those sites orphans the class.
    assign = np.zeros((N_CLASSES, N_CLIENTS))
    for c in range(N_CLASSES):
        primary = [(2 * c + i) % N_CLIENTS for i in range(4)]
        if c in rare:
            w = np.zeros(N_CLIENTS)
            pw = rng.dirichlet(np.full(4, 1.0))
            for j, cl in enumerate(primary):
                w[cl] = pw[j]
        else:
            w = rng.dirichlet(np.full(N_CLIENTS, 0.5))
            w = w / w.sum() * 0.30
            pw = rng.dirichlet(np.full(4, 1.0)) * 0.70
            for j, cl in enumerate(primary):
                w[cl] += pw[j]
        assign[c] = w

    client_X: List[List[np.ndarray]] = [[] for _ in range(N_CLIENTS)]
    client_y: List[List[np.ndarray]] = [[] for _ in range(N_CLIENTS)]
    for c in range(N_CLASSES):
        counts = rng.multinomial(class_counts[c], assign[c])
        start = 0
        for cl in range(N_CLIENTS):
            k = counts[cl]
            if k:
                client_X[cl].append(pools[c][start:start + k])
                client_y[cl].append(np.full(k, c, dtype=int))
            start += k

    clients = []
    for cl in range(N_CLIENTS):
        X = np.vstack(client_X[cl])
        y = np.concatenate(client_y[cl])
        perm = rng.permutation(len(y))
        clients.append(Client(X=X[perm], y=y[perm]))

    val_counts = (CLASS_PRIORS * N_VAL).astype(int)
    val_X = np.vstack([
        rng.normal(means[c], class_scale[c], (val_counts[c], N_FEATURES))
        for c in range(N_CLASSES)
    ])
    val_y = np.concatenate([
        np.full(val_counts[c], c, dtype=int) for c in range(N_CLASSES)
    ])

    # Dropout order: descending local share of the two rarest classes.
    rare = tuple(int(c) for c in np.argsort(CLASS_PRIORS)[:2])
    shares = []
    for cl in range(N_CLIENTS):
        mask = np.isin(clients[cl].y, list(rare))
        shares.append(mask.mean())
    drop_order = [int(i) for i in np.argsort(-np.array(shares), kind="stable")]

    return World(clients=clients, val_X=val_X, val_y=val_y,
                 drop_order=drop_order, rare_classes=rare)


def flip_labels(y: np.ndarray) -> np.ndarray:
    """Fixed cyclic label permutation used by the poisoning ramp."""
    return (y + 1) % N_CLASSES


def lr_at_round(scenario: str, t: int) -> float:
    if scenario == "lr_blowup" and t >= ONSET_ROUND:
        return BASE_LR * (1.35 ** (t - ONSET_ROUND + 1))
    return BASE_LR


def poisoned_at_round(scenario: str, t: int) -> set:
    if scenario == "labelflip_ramp" and t >= ONSET_ROUND:
        return set(range(min(t - ONSET_ROUND + 1, 5)))
    return set()


def dropped_at_round(scenario: str, t: int, drop_order: List[int]) -> set:
    if scenario == "dropout_cascade" and t >= ONSET_ROUND:
        return set(drop_order[:min(t - ONSET_ROUND + 1, 5)])
    return set()


def run_scenario(scenario: str, seed: int, rounds: int = TOTAL_ROUNDS) -> Dict:
    """Run one federated training run and return its per-round stream.

    The stream is the only thing detectors ever see: validation accuracy
    (for the baseline watcher / scoring) plus update statistics.
    """
    world = generate_world(seed)
    rng = np.random.default_rng(seed + 999)
    W = rng.normal(0.0, 0.01, (N_FEATURES, N_CLASSES))
    b = np.zeros(N_CLASSES)

    stream = []
    for t in range(rounds):
        poisoned = poisoned_at_round(scenario, t)
        dropped = dropped_at_round(scenario, t, world.drop_order)
        active = [i for i in range(N_CLIENTS) if i not in dropped]
        lr = lr_at_round(scenario, t)

        updates, norms, losses = [], [], []
        weights = []
        for i in active:
            client = world.clients[i]
            y_eff = flip_labels(client.y) if i in poisoned else client.y
            # Client-reported loss of the incoming global model on local data.
            loss, _, _ = _loss_and_grad(client.X, y_eff, W, b)
            losses.append(loss)
            Wi, bi = W.copy(), b.copy()
            for _ in range(LOCAL_EPOCHS):
                _, gW, gb = _loss_and_grad(client.X, y_eff, Wi, bi)
                Wi = Wi - lr * gW
                bi = bi - lr * gb
            upd = np.concatenate([(Wi - W).ravel(), (bi - b).ravel()])
            updates.append(upd)
            norms.append(float(np.linalg.norm(upd)))
            weights.append(len(client.X))

        U = np.vstack(updates)
        denom = (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
        cos = (U @ U.T) / (denom @ denom.T)
        iu = np.triu_indices(len(active), k=1)
        mean_cos = float(cos[iu].mean()) if len(active) > 1 else 1.0

        w_arr = np.array(weights, dtype=float)
        w_arr /= w_arr.sum()
        W_new = W + sum(w * (u[:N_FEATURES * N_CLASSES].reshape(N_FEATURES, N_CLASSES))
                        for w, u in zip(w_arr, updates))
        b_new = b + sum(w * u[N_FEATURES * N_CLASSES:]
                        for w, u in zip(w_arr, updates))
        W = np.clip(W_new, -_WEIGHT_CLIP, _WEIGHT_CLIP)
        b = np.clip(b_new, -_WEIGHT_CLIP, _WEIGHT_CLIP)

        stream.append({
            "round": t,
            "val_accuracy": _accuracy(world.val_X, world.val_y, W, b),
            "update_norm_mean": float(np.mean(norms)),
            "update_norm_std": float(np.std(norms)),
            "mean_pairwise_cosine": mean_cos,
            "client_losses": [float(x) for x in losses],
            "n_active": len(active),
        })

    return {"scenario": scenario, "seed": seed, "stream": stream}
