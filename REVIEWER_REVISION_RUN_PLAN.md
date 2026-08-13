# Reviewer revision experiment plan

This file maps the unresolved experimental reviewer comments to a minimal, controlled run plan.
It deliberately separates experiments that are required for the revision from experiments that can
be answered by clarification or by explicitly framing a point as future work.

## Before running anything

1. Use `examples/reviewer_experiments.py` for revision-only studies. Do not overwrite the canonical
   reproduction configuration in `examples/bulk_experiments_canonical.py`.
2. Preserve the reported default posterior scale with `posterior_log_std=-4.0` unless running the
   sensitivity study.
3. Record the exact commit, GPU, PyTorch version, CUDA runtime, seed, train size, and model geometry.
4. Keep all comparisons seed-matched and use the same sampled train/validation/test partition within a seed.

## Required study A: CIFAR-100 n=15,000 stability

Reviewer issue: anomalously large Ising standard deviation at n=15,000, especially at regularization
setting 0.1.

Run:
- Dataset: CIFAR-100
- Train samples: 15,000
- Method: Ising LM
- `dropconnect_delta`: 0.1 and 0.5
- Seeds: 0-9
- Total: 20 runs

Report individual seed accuracies, mean, standard deviation, median, min/max, ECE, Brier score, and
sparsity. Do not report only a new mean/SD; the reviewer specifically asked whether a few failed seeds
explain the variance.

## Required study B: computational efficiency / Hessian cost

Reviewer issue: claims of minimal overhead, GPU memory usage, runtime, inference cost, and Hessian cost
are not quantitatively supported.

Run seed-matched comparisons on MNIST n=6,000 and CIFAR-100 n=15,000 for:
- Ising LM
- Ising without saliency
- fixed dropout
- fixed DropConnect

Use three seeds. Additionally run exact diagonal Hessian Ising on MNIST n=6,000 for three seeds.
Total: 27 runs.

Report:
- total training wall time;
- pilot / Ising / fine-tuning phase times;
- exact-Hessian time when applicable;
- peak allocated GPU memory;
- deterministic test-set inference time;
- 50-pass Monte Carlo posterior inference time;
- test accuracy and sparsity, so speed is not shown without the associated predictive tradeoff.

The exact-Hessian study is intentionally restricted to the smaller setting. Its role is to quantify the
cost difference between the paper's LM approximation and a true diagonal-Hessian calculation, not to
replace the primary estimator throughout the full benchmark.

## Required study C: warm-start sensitivity

Reviewer issue: sensitivity to the pilot/warm-start stopping point.

Use CIFAR-100 n=15,000, Ising LM, five matched seeds. Hold pilot+Ising budget at 15 epochs:
- pilot 1 / Ising 14
- pilot 5 / Ising 10
- pilot 10 / Ising 5

Total: 15 runs.

Report accuracy, ECE/Brier, hard-threshold sparsity count, and final validation loss. This isolates the
allocation of a fixed training budget instead of confounding warm-start length with total compute.

## Required study D: hyperparameter sensitivity

Reviewer issue: insufficient sensitivity analysis of the Ising field / regularization offset and
continuous posterior scale.

Use CIFAR-100 n=15,000, three matched seeds, one-factor-at-a-time around the reported default:
- delta: 0.10, 0.25, 0.50, 0.75 at log(sigma)=-4;
- log(sigma): -5, -4, -3 at delta=0.50.

The shared default cell is run once, giving 6 configurations x 3 seeds = 18 runs.

Report accuracy, ECE, MCE, Brier, and sparsity. The implementation uses one shared fixed posterior
Gaussian scale for spike and slab components. It does not implement two independently tunable prior
variances; the manuscript must not claim that such a two-variance sensitivity experiment was run.

## Required study E: stronger sparse Bayesian baseline

The existing repository contains only fixed dropout and fixed DropConnect baselines. That does not
fully satisfy either reviewer's request for a stronger unstructured pruning / sparse Bayesian baseline.

Preferred addition: Sparse Variational Dropout (Molchanov et al., 2017), applied per weight to the same
ViT geometry. It is a better experimental fit than Movement Pruning because the present ViTs are
trained from scratch rather than pruned during pretrained-model fine-tuning.

Implementation status: implemented as `SparseVDLinear`, ported from the uploaded paper-author
Theano/Lasagne repository at commit `3d9b78a`. The port preserves its clipped `log(alpha)` range
[-8, 8], local-reparameterization variance, KL approximation (including its additive constant),
normal weight initialization with standard deviation 0.01, and deterministic pruning at
`log(alpha) >= 3`. MNIST also preserves the authors' `train_clip=True` mean calculation.

Optimization follows the author repository rather than the earlier failed 15-epoch approximation:
batch size 100, 200 epochs, Adam, no KL through zero-based epoch 5 followed by the authors' 15-epoch
linear ramp, and the dataset-specific author learning-rate schedules (MNIST starts at 1e-3 and
decays linearly to zero; CIFAR starts at 1e-5 and begins its linear decay after epoch 100). The ViT
geometry and reviewer data subsets remain fixed, so this is an architecture/data-matched baseline
with method-appropriate optimization rather than an identical-short-budget comparison.

Minimum comparison after implementation:
- MNIST n=6,000, 3 seeds;
- CIFAR-100 n=15,000, 3 seeds;
- same optimizer/training budget where the method permits;
- report accuracy, ECE/Brier, sparsity, runtime, and peak GPU memory.

Total: 6 additional runs for one stronger baseline.

Launch only these six runs with `scripts/arc/submit_sparse_vd.sh`; do not resubmit the completed
80-run reviewer matrix.

## No additional run strictly required

- Expectation / gradient-estimator clarification: code and manuscript clarification are enough, provided
  they accurately describe one-sample weight/mask Monte Carlo per stochastic forward pass, continuous
  reparameterization, and non-differentiated Bernoulli mask updates.
- Variance reduction in Step 7: describe the paired common-random-number keep/drop masks and 128-pair
  averaging already implemented for the final-layer likelihood difference. The efficiency study above
  provides empirical context without requiring a separate estimator-variance experiment.
- Proposed underconfidence mitigation: either validate a precisely defined prior-downweighting mechanism
  or explicitly move it to future work. The current code has no distinct scalar that cleanly implements
  "downweight the prior toward empirical Bayes," so do not reinterpret `dropconnect_delta` as that
  mechanism without changing the mathematics.

## Quantitative calibration

No separate training runs are needed. Every reviewer run now computes ECE, MCE, and multiclass Brier
score from 50 Monte Carlo posterior passes. These metrics should also be computed on the final models
used in the paper tables if the corresponding checkpoints are available.
