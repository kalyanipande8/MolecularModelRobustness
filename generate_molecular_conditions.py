"""
generate_molecular_conditions.py

Builds ONE synthetic molecular "compound-property classification" task
(e.g. standing in for "is this compound active / does it pass an ADMET
filter?") from 32 engineered molecular-descriptor-like features (mix of
informative + redundant/noise dims, plus one nonlinear interaction term),
and then generates 95 distinct molecular "condition" variants of the held
out test set standing in for realistic deployment/evaluation hazards.

IMPORTANT BUG THIS FILE DELIBERATELY AVOIDS (found in the sibling project
BioModelRobustnessFramework during that project's development, and
documented in its README): an early draft called
`np.random.RandomState(seed)` freshly inside each condition-generating
function, which under certain refactors caused the "latent" decision
weight vector used to derive the ground-truth label to be re-derived
per-call instead of shared, silently decoupling train/test/condition
labels from a common decision boundary and producing chance-level
"everything fails" results across the whole matrix.

The fix (used here from the start): the latent weight vector `W_LATENT`,
the interaction feature indices, and the bias term are generated EXACTLY
ONCE at module import time from a single fixed seed, saved to disk
(`data/latent_boundary.npz`), and every downstream function (train set,
test set, all 95 conditions) reads/derives labels from that same fixed
object. There is no function in this file that re-samples the boundary.
"""

import numpy as np
import os
import json

SEED = 42
N_FEATURES = 32
N_INFORMATIVE = 12
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Generate the shared latent decision boundary ONCE.
# ---------------------------------------------------------------------------
_rng_boundary = np.random.RandomState(SEED)
INFORMATIVE_IDX = _rng_boundary.choice(N_FEATURES, size=N_INFORMATIVE, replace=False)
W_LATENT = np.zeros(N_FEATURES)
W_LATENT[INFORMATIVE_IDX] = _rng_boundary.normal(0, 1.5, size=N_INFORMATIVE)
BIAS = _rng_boundary.normal(0, 0.2)
# one nonlinear interaction between two informative descriptors
# (e.g. standing in for logP * molecular-weight interaction)
INTERACTION_IDX = tuple(_rng_boundary.choice(INFORMATIVE_IDX, size=2, replace=False))
INTERACTION_COEF = _rng_boundary.normal(1.0, 0.3)

np.savez(
    os.path.join(DATA_DIR, "latent_boundary.npz"),
    w=W_LATENT,
    bias=BIAS,
    informative_idx=INFORMATIVE_IDX,
    interaction_idx=np.array(INTERACTION_IDX),
    interaction_coef=INTERACTION_COEF,
)


def _labels_from_features(X):
    """Single shared ground-truth function used EVERYWHERE (train, test,
    all 95 conditions). Never regenerates W_LATENT."""
    z = X @ W_LATENT + BIAS
    z += INTERACTION_COEF * X[:, INTERACTION_IDX[0]] * X[:, INTERACTION_IDX[1]]
    prob = 1.0 / (1.0 + np.exp(-z))
    return (prob > 0.5).astype(int), prob


def _base_features(n, rng):
    """Engineered-molecular-descriptor-like features: standardized normals
    for informative dims, correlated/redundant dims derived from them, and
    pure noise dims. Standing in for descriptors like logP, TPSA, MW,
    H-bond donors/acceptors, ring counts, etc."""
    X = np.zeros((n, N_FEATURES))
    X[:, INFORMATIVE_IDX] = rng.normal(0, 1.0, size=(n, N_INFORMATIVE))
    redundant_pool = [i for i in range(N_FEATURES) if i not in INFORMATIVE_IDX]
    n_redundant = len(redundant_pool) // 2
    redundant_idx = redundant_pool[:n_redundant]
    noise_idx = redundant_pool[n_redundant:]
    for j in redundant_idx:
        src = INFORMATIVE_IDX[rng.randint(0, N_INFORMATIVE)]
        X[:, j] = 0.7 * X[:, src] + 0.3 * rng.normal(0, 1.0, size=n)
    for j in noise_idx:
        X[:, j] = rng.normal(0, 1.0, size=n)
    return X


def generate_base_dataset():
    rng = np.random.RandomState(SEED)
    X_train = _base_features(2000, rng)
    y_train, _ = _labels_from_features(X_train)
    X_test = _base_features(500, rng)
    y_test, _ = _labels_from_features(X_test)
    np.savez(os.path.join(DATA_DIR, "base_train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(DATA_DIR, "base_test.npz"), X=X_test, y=y_test)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 2. 95 molecular condition variants of the held-out test set.
# ---------------------------------------------------------------------------
CONDITIONS = []  # list of dicts: name, category, X, y


def _add(name, category, X, rng_seed_note=""):
    y, _ = _labels_from_features(X)
    CONDITIONS.append({"name": name, "category": category, "X": X, "y": y})


def build_conditions(X_test):
    CONDITIONS.clear()

    # --- 1) measurement_noise (22): additive Gaussian noise standing in
    # for experimental descriptor measurement error, sigma swept.
    for i, sigma in enumerate(np.linspace(0.05, 3.0, 22)):
        rng = np.random.RandomState(1000 + i)
        Xc = X_test + rng.normal(0, sigma, size=X_test.shape)
        _add(f"noise_sigma_{sigma:.2f}", "measurement_noise", Xc)

    # --- 2) scaffold_shift (20): mean shift + variance inflation along a
    # random direction, standing in for novel chemical scaffolds /
    # distribution shift relative to the training chemical space.
    rng_dir = np.random.RandomState(7)
    direction = rng_dir.normal(0, 1, size=X_test.shape[1])
    direction /= np.linalg.norm(direction)
    for i, mag in enumerate(np.linspace(0.1, 4.0, 20)):
        rng = np.random.RandomState(2000 + i)
        Xc = X_test + mag * direction + rng.normal(0, 0.1, size=X_test.shape) * mag
        _add(f"scaffold_shift_{mag:.2f}", "scaffold_shift", Xc)

    # --- 3) sparse_descriptors (16): increasing fraction of descriptor
    # entries missing/mean-imputed, standing in for incomplete descriptor
    # panels (e.g. failed computation of a 3D descriptor).
    col_means = X_test.mean(axis=0)
    for i, frac in enumerate(np.linspace(0.05, 0.95, 16)):
        rng = np.random.RandomState(3000 + i)
        Xc = X_test.copy()
        mask = rng.rand(*X_test.shape) < frac
        Xc[mask] = np.take(col_means, np.where(mask)[1])
        _add(f"sparse_{frac:.2f}", "sparse_descriptors", Xc)

    # --- 4) class_imbalance (14): resampled subsets with swept positive
    # ratio, standing in for rare-activity-class prevalence shift.
    y_full, _ = _labels_from_features(X_test)
    pos_idx = np.where(y_full == 1)[0]
    neg_idx = np.where(y_full == 0)[0]
    for i, ratio in enumerate(np.linspace(0.05, 0.85, 14)):
        rng = np.random.RandomState(4000 + i)
        n_total = 200
        n_pos = max(1, int(n_total * ratio))
        n_neg = n_total - n_pos
        n_pos = min(n_pos, len(pos_idx))
        n_neg = min(n_neg, len(neg_idx))
        sel_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        sel_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        sel = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(sel)
        Xc = X_test[sel]
        _add(f"imbalance_ratio_{ratio:.2f}", "class_imbalance", Xc)

    # --- 5) adversarial_perturbation (15): targeted perturbation of the
    # most class-discriminative descriptors toward the opposite class.
    top_feats = np.argsort(-np.abs(W_LATENT))[:6]
    sign = np.sign(W_LATENT[top_feats])
    for i, mag in enumerate(np.linspace(0.1, 5.0, 15)):
        rng = np.random.RandomState(5000 + i)
        Xc = X_test.copy()
        Xc[:, top_feats] -= mag * sign
        Xc += rng.normal(0, 0.05, size=X_test.shape)
        _add(f"adversarial_{mag:.2f}", "adversarial_perturbation", Xc)

    # --- 6) extrapolation (8): out-of-range descriptor values, standing in
    # for compounds far outside the training chemical property range.
    for i, scale in enumerate(np.linspace(3.0, 10.0, 8)):
        rng = np.random.RandomState(6000 + i)
        Xc = X_test.copy()
        Xc[:, INFORMATIVE_IDX] *= scale
        _add(f"extrapolation_scale_{scale:.2f}", "extrapolation", Xc)

    assert len(CONDITIONS) == 95, f"expected 95 conditions, got {len(CONDITIONS)}"
    return CONDITIONS


def save_conditions():
    meta = []
    cond_dir = os.path.join(DATA_DIR, "conditions")
    os.makedirs(cond_dir, exist_ok=True)
    for c in CONDITIONS:
        np.savez(os.path.join(cond_dir, f"{c['name']}.npz"), X=c["X"], y=c["y"])
        meta.append({"name": c["name"], "category": c["category"], "n": int(len(c["y"]))})
    with open(os.path.join(DATA_DIR, "conditions_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    X_train, y_train, X_test, y_test = generate_base_dataset()
    conds = build_conditions(X_test)
    meta = save_conditions()
    print(f"Base train: {X_train.shape}, base test: {X_test.shape}")
    print(f"Generated {len(conds)} molecular conditions.")
    cats = {}
    for c in meta:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
    for k, v in cats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
