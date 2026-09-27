"""
train_architectures.py

Trains 12 distinct, genuinely-fit small architectures on the shared base
training set (data/base_train.npz). No PyTorch (see README for why:
`import torch` failed on this machine and disk headroom was < 1GB, so this
uses the scikit-learn fallback, same pattern as the sibling
BioModelRobustnessFramework project).

Each architecture is a real, distinct model family / config -- this is
NOT 12 copies of one model with different names. Every model is fit with
`model.fit(...)` on the real synthetic training data.
"""

import os
import json
import pickle
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ARCHITECTURES = [
    {
        "id": "arch_01", "name": "logreg_l2_C0.1", "family": "LogisticRegression",
        "build": lambda: LogisticRegression(C=0.1, max_iter=2000),
        "config": "C=0.1, penalty=l2", "subset": 1.0,
    },
    {
        "id": "arch_02", "name": "logreg_l2_C10", "family": "LogisticRegression",
        "build": lambda: LogisticRegression(C=10.0, max_iter=2000),
        "config": "C=10.0, penalty=l2", "subset": 1.0,
    },
    {
        "id": "arch_03", "name": "mlp_shallow_16", "family": "MLPClassifier",
        "build": lambda: MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-4, max_iter=1000, random_state=42),
        "config": "hidden=(16,), alpha=1e-4", "subset": 1.0,
    },
    {
        "id": "arch_04", "name": "mlp_deep_64x32x16", "family": "MLPClassifier",
        "build": lambda: MLPClassifier(hidden_layer_sizes=(64, 32, 16), alpha=1e-4, max_iter=1000, random_state=42),
        "config": "hidden=(64,32,16), alpha=1e-4", "subset": 1.0,
    },
    {
        "id": "arch_05", "name": "mlp_wide_high_reg", "family": "MLPClassifier",
        "build": lambda: MLPClassifier(hidden_layer_sizes=(128,), alpha=1.0, max_iter=1000, random_state=42),
        "config": "hidden=(128,), alpha=1.0 (strong L2)", "subset": 1.0,
    },
    {
        "id": "arch_06", "name": "mlp_32x16_starved25pct", "family": "MLPClassifier",
        "build": lambda: MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-4, max_iter=1000, random_state=42),
        "config": "hidden=(32,16), alpha=1e-4", "subset": 0.25,
    },
    {
        "id": "arch_07", "name": "rf_shallow_n20", "family": "RandomForestClassifier",
        "build": lambda: RandomForestClassifier(max_depth=3, n_estimators=20, random_state=42),
        "config": "max_depth=3, n_estimators=20", "subset": 1.0,
    },
    {
        "id": "arch_08", "name": "rf_deep_n150", "family": "RandomForestClassifier",
        "build": lambda: RandomForestClassifier(max_depth=None, n_estimators=150, random_state=42),
        "config": "max_depth=None, n_estimators=150", "subset": 1.0,
    },
    {
        "id": "arch_09", "name": "gboost_default", "family": "GradientBoostingClassifier",
        "build": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        "config": "n_estimators=100, max_depth=3, lr=0.1", "subset": 1.0,
    },
    {
        "id": "arch_10", "name": "extratrees_default", "family": "ExtraTreesClassifier",
        "build": lambda: ExtraTreesClassifier(n_estimators=120, max_depth=None, random_state=42),
        "config": "n_estimators=120, max_depth=None", "subset": 1.0,
    },
    {
        "id": "arch_11", "name": "decisiontree_depth4", "family": "DecisionTreeClassifier",
        "build": lambda: DecisionTreeClassifier(max_depth=4, random_state=42),
        "config": "max_depth=4", "subset": 1.0,
    },
    {
        "id": "arch_12", "name": "svm_rbf", "family": "SVC",
        "build": lambda: SVC(kernel="rbf", C=1.0, probability=True),
        "config": "kernel=rbf, C=1.0", "subset": 1.0,
    },
]

# two extra families kept available for the fix-testing stage (k-NN, Naive
# Bayes) -- not part of the 12, but used as ensemble ingredients / to
# demonstrate architectural-family diversity was considered.
EXTRA_FAMILIES = {
    "knn": lambda: KNeighborsClassifier(n_neighbors=15),
    "naive_bayes": lambda: GaussianNB(),
}


def load_train():
    d = np.load(os.path.join(DATA_DIR, "base_train.npz"))
    return d["X"], d["y"]


def train_all():
    X, y = load_train()
    trained = []
    for arch in ARCHITECTURES:
        subset = arch["subset"]
        if subset < 1.0:
            rng = np.random.RandomState(42 + hash(arch["id"]) % 1000)
            n = int(len(X) * subset)
            idx = rng.choice(len(X), size=n, replace=False)
            Xa, ya = X[idx], y[idx]
        else:
            Xa, ya = X, y
        model = arch["build"]()
        model.fit(Xa, ya)
        train_acc = model.score(Xa, ya)
        with open(os.path.join(MODEL_DIR, f"{arch['id']}.pkl"), "wb") as f:
            pickle.dump(model, f)
        trained.append({
            "id": arch["id"], "name": arch["name"], "family": arch["family"],
            "config": arch["config"], "subset": subset, "n_train": int(len(Xa)),
            "train_accuracy": float(train_acc),
        })
        print(f"Trained {arch['id']} ({arch['name']}): train_acc={train_acc:.3f}, n={len(Xa)}")

    with open(os.path.join(MODEL_DIR, "architectures_meta.json"), "w") as f:
        json.dump(trained, f, indent=2)
    assert len(trained) == 12
    return trained


if __name__ == "__main__":
    train_all()
