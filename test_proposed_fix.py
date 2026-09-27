"""
test_proposed_fix.py

Implements ONE of the proposed architectural fixes from
run_robustness_evaluation.py as a real 13th trained variant, and tests
whether it genuinely improves robustness on the worst-performing condition
subset(s) discovered in the real 12x95 matrix.

From the real run on this machine (see output/findings.json):
  - Worst-performing category overall: extrapolation (mean acc 0.760)
  - Second worst: measurement_noise (mean acc 0.844)
  - Worst single architecture overall: svm_rbf (mean acc 0.552 on
    extrapolation, 0.735 on measurement_noise) -- an RBF-kernel SVM fit on
    raw (non-noise-augmented) descriptors, which overfits to local
    structure near training-distribution support vectors and degrades
    sharply once inputs move outside that support (extrapolation) or are
    perturbed by noise.

FIX IMPLEMENTED: an ensemble/averaging architecture (soft-voting
classifier) combining the two most-robust individual architectures from
the 12 (logreg_l2_C10 and mlp_shallow_16, the top two by mean accuracy
across all 95 conditions) -- directly implementing the "use an
averaging/ensemble scheme" proposal from run_robustness_evaluation.py.

This is trained as a genuine 13th model (arch_13_ensemble_fix) via
sklearn's VotingClassifier with soft voting, fit on the same base training
set, and evaluated on the same extrapolation + measurement_noise condition
subsets where svm_rbf (the original worst-offender architecture) failed
most badly. Reports real before/after numbers -- not assumed to fully fix
the problem.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "output")

TARGET_CATEGORIES = ["extrapolation", "measurement_noise"]
BASELINE_ARCH = "arch_12"  # svm_rbf: worst-offender architecture on target categories


def load_train():
    d = np.load(os.path.join(DATA_DIR, "base_train.npz"))
    return d["X"], d["y"]


def load_condition(name):
    d = np.load(os.path.join(DATA_DIR, "conditions", f"{name}.npz"))
    return d["X"], d["y"]


def build_and_train_fix():
    X, y = load_train()
    ensemble = VotingClassifier(
        estimators=[
            ("logreg_l2_C10", LogisticRegression(C=10.0, max_iter=2000)),
            ("mlp_shallow_16", MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-4, max_iter=1000, random_state=42)),
        ],
        voting="soft",
    )
    ensemble.fit(X, y)
    with open(os.path.join(MODEL_DIR, "arch_13_ensemble_fix.pkl"), "wb") as f:
        pickle.dump(ensemble, f)
    return ensemble


def load_model(arch_id):
    with open(os.path.join(MODEL_DIR, f"{arch_id}.pkl"), "rb") as f:
        return pickle.load(f)


def evaluate_on_target_conditions(model, cond_meta):
    accs = []
    per_cond = []
    for cond in cond_meta:
        if cond["category"] not in TARGET_CATEGORIES:
            continue
        X, y = load_condition(cond["name"])
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        accs.append(acc)
        per_cond.append({"condition": cond["name"], "category": cond["category"], "accuracy": acc})
    return float(np.mean(accs)), per_cond


def main():
    with open(os.path.join(DATA_DIR, "conditions_meta.json")) as f:
        cond_meta = json.load(f)

    baseline_model = load_model(BASELINE_ARCH)
    before_mean, before_per_cond = evaluate_on_target_conditions(baseline_model, cond_meta)

    fix_model = build_and_train_fix()
    after_mean, after_per_cond = evaluate_on_target_conditions(fix_model, cond_meta)

    # also compare against the two individual ensemble members for context
    logreg_model = load_model("arch_02")  # logreg_l2_C10
    mlp_model = load_model("arch_03")  # mlp_shallow_16
    logreg_mean, _ = evaluate_on_target_conditions(logreg_model, cond_meta)
    mlp_mean, _ = evaluate_on_target_conditions(mlp_model, cond_meta)

    result = {
        "target_categories": TARGET_CATEGORIES,
        "baseline_architecture": BASELINE_ARCH,
        "before_mean_accuracy": before_mean,
        "after_mean_accuracy": after_mean,
        "improvement": after_mean - before_mean,
        "ensemble_member_logreg_l2_C10_mean_accuracy": logreg_mean,
        "ensemble_member_mlp_shallow_16_mean_accuracy": mlp_mean,
        "before_per_condition": before_per_cond,
        "after_per_condition": after_per_cond,
        "genuinely_improved": bool(after_mean > before_mean),
    }
    with open(os.path.join(OUT_DIR, "fix_test_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"Target categories: {TARGET_CATEGORIES}")
    print(f"BEFORE (baseline arch = {BASELINE_ARCH}): mean accuracy = {before_mean:.3f}")
    print(f"AFTER  (arch_13_ensemble_fix):            mean accuracy = {after_mean:.3f}")
    print(f"Change: {result['improvement']:+.3f}")
    print(f"(context) logreg_l2_C10 alone: {logreg_mean:.3f}, mlp_shallow_16 alone: {mlp_mean:.3f}")
    return result


if __name__ == "__main__":
    main()
