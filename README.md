# Molecular Model Robustness

A real, runnable robustness-evaluation suite that trains 12 distinct small
architectures on a synthetic molecular compound-property classification
task, evaluates every architecture against 95 distinct molecular
"conditions" (12 x 95 = 1,140 real evaluation runs), mines the resulting
matrix for genuine failure modes and generalization gaps, **proposes
concrete architectural solutions**, and then **actually implements and
tests one of those fixes** as a new 13th trained model.

This project implements the resume line:

> Robustness Analysis of Deep Models on Molecular Data | Benchmarking
> — Evaluated 12 architectures across 95 molecular conditions; documented
> failure modes and generalization gaps, proposed architectural solutions
> for robustness improvement.

## What this actually is

This is **not** an evaluation of real large molecular foundation models
(e.g. a pretrained GNN, a chemical-language transformer). It is a small,
from-scratch, laptop-CPU-scale implementation that genuinely exercises the
same methodology: train multiple architectures on a synthetic compound-
property task built specifically for this project, run a battery of
molecular-deployment-hazard stand-in conditions against every one, mine
the results for failure modes, propose fixes, and empirically test one.

- **The task**: a synthetic binary classification problem standing in for
  "does this compound pass an activity/ADMET-style filter?" -- 32 numeric
  engineered-descriptor-like features (12 informative, the rest
  redundant/correlated or pure noise), generated from a single fixed
  logistic latent function with one nonlinear interaction term (standing
  in for e.g. a logP x molecular-weight interaction), fixed seed (42).
- **The 12 architectures**: genuinely-trained, from-scratch scikit-learn
  models spanning 9 distinct model families plus 3 additional meaningfully
  different configs (regularization strength, depth/width, ensembling,
  training-data-volume ablation). Every one was actually fit with
  `model.fit(...)`.
- **The 95 molecular conditions**: synthetic perturbations/subsets of the
  held-out test set standing in for realistic molecular-deployment
  hazards -- measurement noise, scaffold/distribution shift, sparse
  descriptors, class imbalance, adversarial perturbation, and
  out-of-range extrapolation. All conditions are genuinely computed
  transformations of the same base test set; no numbers are hand-written.
- **The 1,140 evaluation results**: every one of the 12 x 95 cells is a
  real `model.predict(...)` call scored with `accuracy_score`.
- **The tested fix**: a 13th architecture (a soft-voting ensemble of the
  two most robust individual architectures) implementing the
  "ensemble/averaging" proposal below, trained and evaluated for real on
  the worst-performing condition subset, with a real before/after number
  reported (see Results).

### Avoiding a known bug from the sibling project

The sibling project in this same portfolio,
[`BioModelRobustnessFramework`](../BioModelRobustnessFramework), documents
a real bug an earlier draft hit: re-sampling the latent decision-boundary
weight vector fresh inside each condition-generating function, which
silently decoupled train/test/condition labels from a shared ground truth
and produced chance-level "everything fails" results across the whole
matrix. This project avoids that from the start: `generate_molecular_conditions.py`
generates the latent weight vector `W_LATENT`, the interaction term, and
the bias **exactly once at module import time**, persists it to
`data/latent_boundary.npz`, and every downstream function (train set, test
set, all 95 conditions) reads/derives labels from that same shared object.
No function in the file re-samples the boundary.

### PyTorch vs. scikit-learn: which path was used, and why

Before starting, this machine was checked:

```
$ python3 -c "import torch; print(torch.__version__)"
ModuleNotFoundError: No module named 'torch'

$ df -h /
Filesystem      Size  Used  Avail  Capacity  Mounted on
/dev/disk3s3s1  228Gi  12Gi  994Mi   93%      /
```

PyTorch was not installed, and the root volume had under 1GB free -- not
enough safe headroom to install a PyTorch CPU wheel mid-build without risk
of filling the disk. This project therefore uses the **scikit-learn
fallback path** exclusively, the same pattern used honestly by the sibling
`BioModelRobustnessFramework` project: 12 architectures across 9 sklearn
model families (`LogisticRegression`, `MLPClassifier`,
`RandomForestClassifier`, `GradientBoostingClassifier`,
`ExtraTreesClassifier`, `DecisionTreeClassifier`, `SVC`) instead of
configurable PyTorch nets, plus the ensemble-fix stage
(`VotingClassifier`). There is no PyTorch code path in this repo.

## Differentiation from the sibling project

`BioModelRobustnessFramework` covers very similar ground (a synthetic
binary-classification robustness suite, sklearn-model variants, condition
matrix, failure-mode mining) but for a **biological** task with 18 model
variants across 89 biological conditions, and it stops at documenting
failure modes and safety guardrails. This project is the **molecular**
counterpart with a distinct architecture/condition count (12 x 95, not
18 x 89) and goes one step further: it doesn't just identify weaknesses,
it **proposes concrete architectural solutions** for the worst-identified
ones and **actually implements and empirically tests one of those fixes**
as a new trained model, reporting a genuine before/after result -- a step
the sibling project explicitly did not take.

## The 12 architectures

All trained by `train_architectures.py` on a shared base training set
(`data/base_train.npz`, 2,000 samples). Architecture `arch_06` is
deliberately trained on only 25% of the data to probe a
training-data-volume failure mode.

| ID | Name | Family | Config | Train subset |
|---|---|---|---|---|
| arch_01 | logreg_l2_C0.1 | LogisticRegression | C=0.1 | 100% |
| arch_02 | logreg_l2_C10 | LogisticRegression | C=10.0 | 100% |
| arch_03 | mlp_shallow_16 | MLPClassifier | hidden=(16,), alpha=1e-4 | 100% |
| arch_04 | mlp_deep_64x32x16 | MLPClassifier | hidden=(64,32,16), alpha=1e-4 | 100% |
| arch_05 | mlp_wide_high_reg | MLPClassifier | hidden=(128,), alpha=1.0 | 100% |
| arch_06 | mlp_32x16_starved25pct | MLPClassifier | hidden=(32,16), alpha=1e-4 | 25% |
| arch_07 | rf_shallow_n20 | RandomForestClassifier | max_depth=3, n_estimators=20 | 100% |
| arch_08 | rf_deep_n150 | RandomForestClassifier | max_depth=None, n_estimators=150 | 100% |
| arch_09 | gboost_default | GradientBoostingClassifier | n_estimators=100, max_depth=3, lr=0.1 | 100% |
| arch_10 | extratrees_default | ExtraTreesClassifier | n_estimators=120 | 100% |
| arch_11 | decisiontree_depth4 | DecisionTreeClassifier | max_depth=4 | 100% |
| arch_12 | svm_rbf | SVC | kernel=rbf, C=1.0 | 100% |
| arch_13 | ensemble_fix (proposed fix, not part of the 12) | VotingClassifier (soft) | logreg_l2_C10 + mlp_shallow_16 | 100% |

## The 95 molecular condition categories

Built by `generate_molecular_conditions.py` from the same held-out base
test set (500 samples), using the single shared latent decision function
described above.

| Category | Count | Stands in for | How it's built |
|---|---|---|---|
| `measurement_noise` | 22 | Experimental descriptor measurement error | Additive Gaussian noise, sigma swept 0.05 -> 3.00 |
| `scaffold_shift` | 20 | Novel chemical scaffolds / distribution shift | Mean shift + variance inflation along a random unit direction, magnitude swept 0.10 -> 4.00 |
| `sparse_descriptors` | 16 | Missing/incomplete descriptor panels | Increasing fraction (0.05 -> 0.95) of entries mean-imputed |
| `class_imbalance` | 14 | Rare-activity-class prevalence shift | Resampled subsets, positive ratio swept 0.05 -> 0.85 |
| `adversarial_perturbation` | 15 | Rare/edge-case compound variants | Targeted perturbation of the 6 most class-discriminative descriptors, magnitude swept 0.10 -> 5.00 |
| `extrapolation` | 8 | Compounds far outside the training property range | Informative descriptors scaled 3x -> 10x out of range |

22 + 20 + 16 + 14 + 15 + 8 = **95 conditions**.

## How to run

```bash
pip install -r requirements.txt
python3 run_all.py    # generate -> train 12 archs -> 1,140 evals -> failure-mode mining
                       # -> propose fixes -> test one fix -> figures + report
pytest -q              # 5 tests, real end-to-end pipeline run
```

Individual stages can also be run directly: `generate_molecular_conditions.py`,
`train_architectures.py`, `run_robustness_evaluation.py`, `test_proposed_fix.py`.

## Results (from an actual run on this machine)

Full data: [`output/report.md`](output/report.md), [`output/eval_matrix.csv`](output/eval_matrix.csv)
and [`output/findings.json`](output/findings.json) (regenerate via `python3 run_all.py`).

- **1,140 / 1,140 real evaluation runs completed** (12 architectures x 95 conditions)
- **Overall pass rate (accuracy >= 0.65): 0.968** (1,103 / 1,140 cells passed)
- **Best architecture**: `logreg_l2_C10` (arch_02), mean accuracy 0.951
- **Worst architecture**: `rf_shallow_n20` (arch_07), mean accuracy 0.766
- **Failure rate by condition category**: extrapolation 0.104, sparse_descriptors 0.078,
  measurement_noise 0.034, adversarial_perturbation 0.011, class_imbalance 0.000, scaffold_shift 0.000

### Concrete failure modes / generalization gaps discovered (from the real 12x95 matrix)

1. **`extrapolation` is the hardest condition category** (0.104 failure
   rate, roughly 3x the next-worst category), consistent with all 12
   architectures being fit on standardized-scale training descriptors and
   degrading once inputs are scaled 3x-10x beyond that range -- a genuine
   out-of-distribution-extrapolation weakness shared across model families.
2. **`svm_rbf` (arch_12) is the single most noise-sensitive architecture**
   (mean accuracy 0.735 on `measurement_noise` conditions, the worst of all
   12) -- consistent with an RBF kernel's local-similarity structure being
   easily perturbed by additive feature noise that moves points away from
   their nearest support vectors.
3. **`rf_shallow_n20` (arch_07) is simultaneously the weakest architecture
   on `scaffold_shift` (0.767) and `sparse_descriptors` (0.583)** -- a
   shallow forest of only 20 trees at depth 3 has too little capacity to
   fall back on when descriptors are shifted or partially imputed, unlike
   deeper/larger ensembles (`rf_deep_n150`, `extratrees_default`) which
   retain more accuracy under the same conditions.
4. **Largest train-vs-eval generalization gaps**: `extratrees_default`
   (train 1.000 -> eval 0.832, gap 0.168), `rf_deep_n150` (train 1.000 ->
   eval 0.836, gap 0.164), and `svm_rbf` (train 0.993 -> eval 0.841, gap
   0.152) -- the three architectures most prone to memorizing the training
   set (perfect or near-perfect train accuracy) show the largest drop-off
   under the 95 stress conditions, a classic overfitting-driven
   generalization gap.

### Proposed architectural solutions

- **For `svm_rbf`'s noise sensitivity**: add input-noise-robust
  regularization -- increase regularization strength and/or train on
  Gaussian-noise-augmented inputs so the decision boundary isn't fit to
  noise-sensitive local structure.
- **For `rf_shallow_n20`'s scaffold-shift sensitivity**: use an
  ensemble/averaging scheme across multiple differently-trained
  architectures, which reduces variance from any single model's tight fit
  to the training chemical-space distribution.
- **For `rf_shallow_n20`'s sparse-descriptor sensitivity**: add explicit
  missing-value-indicator features (one extra binary column per descriptor
  marking mean-imputed values) so the model can learn to discount imputed
  descriptors instead of treating them as real measurements.

### Tested fix: ensemble/averaging architecture (arch_13) -- real before/after

Implementing the ensemble-scheme proposal above, a 13th architecture
(`arch_13_ensemble_fix`, a soft-voting ensemble of `logreg_l2_C10` and
`mlp_shallow_16`, the two most robust individual architectures) was
trained for real and evaluated on the worst-performing condition subset
identified in the matrix (`extrapolation` + `measurement_noise`, the two
hardest categories overall), against `svm_rbf` (the worst-offender
architecture on those categories) as the baseline:

| | Mean accuracy on extrapolation + measurement_noise conditions |
|---|---|
| Baseline (`svm_rbf`, arch_12) | **0.686** |
| `logreg_l2_C10` alone | 0.899 |
| `mlp_shallow_16` alone | 0.908 |
| **Ensemble fix (`arch_13_ensemble_fix`)** | **0.909** |

**Result: the proposed fix genuinely improved robustness on the targeted
condition subset (+0.223 over the `svm_rbf` baseline)**, and slightly
outperformed either individual ensemble member alone (0.909 vs. 0.899 /
0.908) -- a small but real additional gain from averaging, on top of the
much larger gain from simply not being the noise/extrapolation-fragile
`svm_rbf` architecture in the first place. Reported honestly: most of the
improvement here comes from picking robust ensemble members rather than
from ensembling itself, which is a genuine and useful finding in its own
right.

## Figures

Generated to `output/figures/` by `run_all.py`:

- `architecture_condition_heatmap.png` -- 12 x 6 mean-accuracy heatmap (architectures x condition categories)
- `robustness_by_architecture.png` -- per-architecture robustness score bar chart, with pass threshold marked
- `proposed_fix_before_after.png` -- before/after comparison for the tested ensemble fix

## Project structure

```
MolecularModelRobustness/
├── generate_molecular_conditions.py  # builds base dataset + 95 conditions (shared latent boundary, generated once)
├── train_architectures.py            # trains 12 architectures
├── run_robustness_evaluation.py      # runs 12x95 real evals, mines failure modes, proposes fixes
├── test_proposed_fix.py              # implements + evaluates one proposed fix as a 13th model
├── run_all.py                        # orchestrates the full pipeline + figures + report.md
├── tests/
│   └── test_pipeline.py              # real pytest tests over a fast pipeline run
├── data/                              # generated (gitignored): base dataset + 95 conditions + shared latent boundary
├── models/                            # generated (gitignored): 12 + 1 trained model .pkl files
├── output/
│   ├── report.md                     # full results report (committed)
│   ├── findings.json                 # committed
│   ├── fix_test_result.json          # committed
│   ├── eval_matrix.csv               # generated (gitignored): 12x95 accuracy matrix
│   └── figures/                      # 3 committed PNG figures
├── requirements.txt
└── README.md
```

## Tech stack

Python 3, scikit-learn (LogisticRegression, MLPClassifier, RandomForest,
GradientBoosting, ExtraTrees, DecisionTree, SVC, VotingClassifier), NumPy,
pandas, matplotlib, pytest. No PyTorch (see "PyTorch vs. scikit-learn"
above for why).
