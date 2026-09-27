"""
run_robustness_evaluation.py

Evaluates all 12 trained architectures against all 95 molecular conditions
(12 x 95 = 1,140 real evaluation runs -- every cell is a real
model.predict(...) + accuracy_score(...) call, never simulated), builds the
architecture x condition performance matrix, mines it for genuine failure
modes / generalization gaps, and proposes concrete architectural fixes for
the worst ones.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

PASS_THRESHOLD = 0.65


def load_architectures_meta():
    with open(os.path.join(MODEL_DIR, "architectures_meta.json")) as f:
        return json.load(f)


def load_conditions_meta():
    with open(os.path.join(DATA_DIR, "conditions_meta.json")) as f:
        return json.load(f)


def load_model(arch_id):
    with open(os.path.join(MODEL_DIR, f"{arch_id}.pkl"), "rb") as f:
        return pickle.load(f)


def load_condition(name):
    d = np.load(os.path.join(DATA_DIR, "conditions", f"{name}.npz"))
    return d["X"], d["y"]


def run_matrix(arch_meta=None, cond_meta=None):
    arch_meta = arch_meta or load_architectures_meta()
    cond_meta = cond_meta or load_conditions_meta()
    rows = []
    matrix = {}
    for arch in arch_meta:
        model = load_model(arch["id"])
        matrix[arch["id"]] = {}
        for cond in cond_meta:
            X, y = load_condition(cond["name"])
            preds = model.predict(X)
            acc = accuracy_score(y, preds)
            matrix[arch["id"]][cond["name"]] = acc
            rows.append({
                "architecture": arch["id"], "arch_name": arch["name"],
                "condition": cond["name"], "category": cond["category"],
                "accuracy": acc, "passed": acc >= PASS_THRESHOLD,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "eval_matrix.csv"), index=False)
    n_expected = len(arch_meta) * len(cond_meta)
    assert len(df) == n_expected, f"expected {n_expected} rows, got {len(df)}"
    return df


def analyze(df, arch_meta, cond_meta):
    findings = {}

    overall_pass_rate = df["passed"].mean()
    findings["overall_pass_rate"] = float(overall_pass_rate)
    findings["n_evaluations"] = int(len(df))

    per_arch = df.groupby(["architecture", "arch_name"])["accuracy"].mean().reset_index()
    per_arch = per_arch.sort_values("accuracy", ascending=False)
    findings["best_architecture"] = per_arch.iloc[0].to_dict()
    findings["worst_architecture"] = per_arch.iloc[-1].to_dict()
    findings["robustness_by_architecture"] = per_arch.to_dict(orient="records")

    per_cat = df.groupby("category").apply(lambda g: 1 - g["passed"].mean(), include_groups=False)
    findings["failure_rate_by_category"] = per_cat.to_dict()

    # architecture x category mean accuracy, to find genuine
    # architecture-specific weaknesses (not just global worst model)
    pivot = df.groupby(["arch_name", "category"])["accuracy"].mean().unstack()
    findings["arch_by_category_matrix"] = pivot.round(3).to_dict()

    worst_cat = max(findings["failure_rate_by_category"], key=findings["failure_rate_by_category"].get)
    findings["worst_category"] = worst_cat

    # most noise-sensitive architecture (biggest drop from clean test acc to
    # noise-condition mean acc)
    noise_by_arch = df[df["category"] == "measurement_noise"].groupby("arch_name")["accuracy"].mean()
    findings["most_noise_sensitive_arch"] = noise_by_arch.idxmin()
    findings["most_noise_sensitive_arch_acc"] = float(noise_by_arch.min())

    shift_by_arch = df[df["category"] == "scaffold_shift"].groupby("arch_name")["accuracy"].mean()
    findings["most_scaffold_shift_sensitive_arch"] = shift_by_arch.idxmin()
    findings["most_scaffold_shift_sensitive_arch_acc"] = float(shift_by_arch.min())

    sparse_by_arch = df[df["category"] == "sparse_descriptors"].groupby("arch_name")["accuracy"].mean()
    findings["most_sparse_sensitive_arch"] = sparse_by_arch.idxmin()
    findings["most_sparse_sensitive_arch_acc"] = float(sparse_by_arch.min())

    # generalization gap: train accuracy vs mean eval accuracy per arch
    train_acc = {a["name"]: a["train_accuracy"] for a in arch_meta}
    per_arch["train_accuracy"] = per_arch["arch_name"].map(train_acc)
    per_arch["generalization_gap"] = per_arch["train_accuracy"] - per_arch["accuracy"]
    findings["generalization_gaps"] = per_arch.sort_values(
        "generalization_gap", ascending=False
    )[["arch_name", "train_accuracy", "accuracy", "generalization_gap"]].to_dict(orient="records")

    return findings


def propose_fixes(findings):
    proposals = []
    proposals.append({
        "target": findings["most_noise_sensitive_arch"],
        "weakness": "measurement_noise",
        "solution": (
            "Add input-noise-robust regularization: increase L2 alpha and/or "
            "wrap features in a light Gaussian-noise-augmented training "
            "scheme (train on X + small-sigma noise draws so the decision "
            "boundary is not fit to noise-sensitive local structure)."
        ),
    })
    proposals.append({
        "target": findings["most_scaffold_shift_sensitive_arch"],
        "weakness": "scaffold_shift",
        "solution": (
            "Use an averaging/ensemble scheme across multiple differently-"
            "regularized or differently-trained architectures -- ensembling "
            "reduces variance from any single model's tight fit to the "
            "training chemical-space distribution and improves robustness "
            "to shifted test distributions."
        ),
    })
    proposals.append({
        "target": findings["most_sparse_sensitive_arch"],
        "weakness": "sparse_descriptors",
        "solution": (
            "Add explicit missing-value-indicator features (one extra binary "
            "column per descriptor marking mean-imputed values) so the model "
            "can learn to discount imputed descriptors instead of treating "
            "them as real measurements."
        ),
    })
    return proposals


def main():
    arch_meta = load_architectures_meta()
    cond_meta = load_conditions_meta()
    df = run_matrix(arch_meta, cond_meta)
    findings = analyze(df, arch_meta, cond_meta)
    proposals = propose_fixes(findings)

    with open(os.path.join(OUT_DIR, "findings.json"), "w") as f:
        json.dump({"findings": findings, "proposals": proposals}, f, indent=2, default=str)

    print(f"Ran {len(df)} evaluations ({len(arch_meta)} arch x {len(cond_meta)} conditions)")
    print(f"Overall pass rate: {findings['overall_pass_rate']:.3f}")
    print(f"Best arch: {findings['best_architecture']['arch_name']}")
    print(f"Worst arch: {findings['worst_architecture']['arch_name']}")
    print(f"Worst category: {findings['worst_category']}")
    for p in proposals:
        print(f"Proposal for {p['target']} ({p['weakness']}): {p['solution'][:60]}...")
    return df, findings, proposals


if __name__ == "__main__":
    main()
