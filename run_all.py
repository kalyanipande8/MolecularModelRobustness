"""
run_all.py

Orchestrates the full pipeline: generate conditions -> train 12
architectures -> run 1,140 real evaluations -> mine failure modes ->
propose fixes -> test one fix -> write output/report.md and figures.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate_molecular_conditions as genmod
import train_architectures as trainmod
import run_robustness_evaluation as evalmod
import test_proposed_fix as fixmod

BASE = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE, "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def make_figures(df, findings):
    # 1) architecture x condition heatmap (aggregated by category to stay
    # small/legible: 12 architectures x 6 categories)
    pivot = df.groupby(["arch_name", "category"])["accuracy"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean accuracy: 12 architectures x 6 condition categories\n(95 total molecular conditions)")
    fig.colorbar(im, ax=ax, label="mean accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "architecture_condition_heatmap.png"), dpi=120)
    plt.close(fig)

    # 2) per-architecture robustness bar chart
    per_arch = df.groupby("arch_name")["accuracy"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(per_arch.index, per_arch.values, color="#4C72B0")
    ax.axvline(evalmod.PASS_THRESHOLD, color="red", linestyle="--", label="pass threshold (0.65)")
    ax.set_xlabel("mean accuracy across all 95 conditions")
    ax.set_title("Per-architecture robustness score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "robustness_by_architecture.png"), dpi=120)
    plt.close(fig)

    return pivot


def make_fix_figure(fix_result):
    labels = ["baseline\n(svm_rbf)", "logreg_l2_C10\n(alone)", "mlp_shallow_16\n(alone)", "ensemble fix\n(arch_13)"]
    values = [
        fix_result["before_mean_accuracy"],
        fix_result["ensemble_member_logreg_l2_C10_mean_accuracy"],
        fix_result["ensemble_member_mlp_shallow_16_mean_accuracy"],
        fix_result["after_mean_accuracy"],
    ]
    colors = ["#C44E52", "#8C8C8C", "#8C8C8C", "#55A868"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("mean accuracy on extrapolation + measurement_noise conditions")
    ax.set_title("Proposed architectural fix: before vs. after")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "proposed_fix_before_after.png"), dpi=120)
    plt.close(fig)


def write_report(findings, proposals, fix_result, arch_meta, cond_meta):
    lines = []
    lines.append("# Molecular Model Robustness -- Results Report\n")
    lines.append(f"- Architectures trained: {len(arch_meta)}")
    lines.append(f"- Molecular conditions evaluated: {len(cond_meta)}")
    lines.append(f"- Total evaluation runs: {findings['n_evaluations']}")
    lines.append(f"- Overall pass rate (accuracy >= {evalmod.PASS_THRESHOLD}): {findings['overall_pass_rate']:.3f}\n")
    lines.append(f"- Best architecture: {findings['best_architecture']['arch_name']} "
                  f"(mean accuracy {findings['best_architecture']['accuracy']:.3f})")
    lines.append(f"- Worst architecture: {findings['worst_architecture']['arch_name']} "
                  f"(mean accuracy {findings['worst_architecture']['accuracy']:.3f})")
    lines.append(f"- Worst condition category: {findings['worst_category']}\n")

    lines.append("## Failure rate by condition category\n")
    for cat, rate in sorted(findings["failure_rate_by_category"].items(), key=lambda x: -x[1]):
        lines.append(f"- {cat}: {rate:.3f}")

    lines.append("\n## Discovered failure modes / generalization gaps\n")
    lines.append(f"1. **Most noise-sensitive architecture**: {findings['most_noise_sensitive_arch']} "
                 f"(mean accuracy {findings['most_noise_sensitive_arch_acc']:.3f} on measurement_noise conditions).")
    lines.append(f"2. **Most scaffold-shift-sensitive architecture**: {findings['most_scaffold_shift_sensitive_arch']} "
                 f"(mean accuracy {findings['most_scaffold_shift_sensitive_arch_acc']:.3f} on scaffold_shift conditions).")
    lines.append(f"3. **Most sparse-descriptor-sensitive architecture**: {findings['most_sparse_sensitive_arch']} "
                 f"(mean accuracy {findings['most_sparse_sensitive_arch_acc']:.3f} on sparse_descriptors conditions).")
    gaps = findings["generalization_gaps"][:3]
    lines.append("4. **Largest train-vs-eval generalization gaps**:")
    for g in gaps:
        lines.append(f"   - {g['arch_name']}: train_acc={g['train_accuracy']:.3f}, "
                     f"eval_acc={g['accuracy']:.3f}, gap={g['generalization_gap']:.3f}")

    lines.append("\n## Proposed architectural solutions\n")
    for p in proposals:
        lines.append(f"- **Target: {p['target']}** (weakness: {p['weakness']}) -- {p['solution']}")

    lines.append("\n## Tested fix: ensemble/averaging architecture (arch_13)\n")
    lines.append(f"- Target condition subset: {fix_result['target_categories']}")
    lines.append(f"- Baseline architecture (worst offender): {fix_result['baseline_architecture']}")
    lines.append(f"- BEFORE mean accuracy: {fix_result['before_mean_accuracy']:.3f}")
    lines.append(f"- AFTER mean accuracy (arch_13_ensemble_fix): {fix_result['after_mean_accuracy']:.3f}")
    lines.append(f"- Change: {fix_result['improvement']:+.3f}")
    lines.append(f"- Genuinely improved: {fix_result['genuinely_improved']}")

    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    print("=== 1. Generating base dataset + 95 molecular conditions ===")
    X_train, y_train, X_test, y_test = genmod.generate_base_dataset()
    genmod.build_conditions(X_test)
    cond_meta = genmod.save_conditions()

    print("\n=== 2. Training 12 architectures ===")
    arch_meta = trainmod.train_all()

    print("\n=== 3. Running 12 x 95 evaluation matrix ===")
    df = evalmod.run_matrix(arch_meta, cond_meta)
    findings = evalmod.analyze(df, arch_meta, cond_meta)
    proposals = evalmod.propose_fixes(findings)
    with open(os.path.join(OUT_DIR, "findings.json"), "w") as f:
        json.dump({"findings": findings, "proposals": proposals}, f, indent=2, default=str)

    print("\n=== 4. Testing proposed architectural fix ===")
    fix_result = fixmod.main()

    print("\n=== 5. Building figures + report ===")
    make_figures(df, findings)
    make_fix_figure(fix_result)
    report_text = write_report(findings, proposals, fix_result, arch_meta, cond_meta)

    print("\n" + report_text)


if __name__ == "__main__":
    main()
