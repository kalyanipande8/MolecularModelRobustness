# Molecular Model Robustness -- Results Report

- Architectures trained: 12
- Molecular conditions evaluated: 95
- Total evaluation runs: 1140
- Overall pass rate (accuracy >= 0.65): 0.968

- Best architecture: logreg_l2_C10 (mean accuracy 0.951)
- Worst architecture: rf_shallow_n20 (mean accuracy 0.766)
- Worst condition category: extrapolation

## Failure rate by condition category

- extrapolation: 0.104
- sparse_descriptors: 0.078
- measurement_noise: 0.034
- adversarial_perturbation: 0.011
- class_imbalance: 0.000
- scaffold_shift: 0.000

## Discovered failure modes / generalization gaps

1. **Most noise-sensitive architecture**: svm_rbf (mean accuracy 0.735 on measurement_noise conditions).
2. **Most scaffold-shift-sensitive architecture**: rf_shallow_n20 (mean accuracy 0.767 on scaffold_shift conditions).
3. **Most sparse-descriptor-sensitive architecture**: rf_shallow_n20 (mean accuracy 0.583 on sparse_descriptors conditions).
4. **Largest train-vs-eval generalization gaps**:
   - extratrees_default: train_acc=1.000, eval_acc=0.832, gap=0.168
   - rf_deep_n150: train_acc=1.000, eval_acc=0.836, gap=0.164
   - svm_rbf: train_acc=0.993, eval_acc=0.841, gap=0.152

## Proposed architectural solutions

- **Target: svm_rbf** (weakness: measurement_noise) -- Add input-noise-robust regularization: increase L2 alpha and/or wrap features in a light Gaussian-noise-augmented training scheme (train on X + small-sigma noise draws so the decision boundary is not fit to noise-sensitive local structure).
- **Target: rf_shallow_n20** (weakness: scaffold_shift) -- Use an averaging/ensemble scheme across multiple differently-regularized or differently-trained architectures -- ensembling reduces variance from any single model's tight fit to the training chemical-space distribution and improves robustness to shifted test distributions.
- **Target: rf_shallow_n20** (weakness: sparse_descriptors) -- Add explicit missing-value-indicator features (one extra binary column per descriptor marking mean-imputed values) so the model can learn to discount imputed descriptors instead of treating them as real measurements.

## Tested fix: ensemble/averaging architecture (arch_13)

- Target condition subset: ['extrapolation', 'measurement_noise']
- Baseline architecture (worst offender): arch_12
- BEFORE mean accuracy: 0.686
- AFTER mean accuracy (arch_13_ensemble_fix): 0.909
- Change: +0.223
- Genuinely improved: True
