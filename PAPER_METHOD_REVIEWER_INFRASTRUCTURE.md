# Paper method with reviewer-ready infrastructure

This repository uses the reviewer-ready repository as its base so the complete
380-run experiment framework remains available. Commit archive
`4dec1b37b83deef07a13c21256c6e20e6f1ad1bf` is used only as the authoritative
reference for the proposed likelihood-guided method.

## Reviewer functionality retained

- all six reviewer studies and all 380 matrix entries;
- Sparse Variational Dropout through `SparseVDLinear`;
- the method-specific Sparse-VD KL and learning-rate schedules;
- the configurable posterior-scale sensitivity study;
- runtime, phase-time, Hessian-time, memory, calibration, and sparsity outputs;
- race-free per-run CSV files and result merging;
- ARC array launchers and the packed missing-dataset launcher;
- the local resume-safe serial launcher;
- data-loader reuse and chunked prediction utilities; and
- exact blockwise Hessian acceleration.

## Proposed-method behavior restored from the paper commit

### LM saliency timing

The later reviewer code paired gradients and activations in a new cache. The
paper method instead uses the gradient left by the preceding optimization step
and activations captured by the current Ising forward. The cache has therefore
been removed from the proposed-method path.

### Averaged-mask timing

The later reviewer code applied the accumulated averaged Ising masks at the
start of fine-tuning. The paper code applies them only when the final overall
training epoch begins. The reviewer-ready `_finalize_ising_masks` helper is
retained, but its call occurs at the paper-defined time.

### Multi-model gradient clearing

The paper clears every model optimizer before each model's backward pass. This
matters because the cross-model similarity term writes gradients to more than
one model. That ordering is restored.

### Probability safeguards

The paper's unconditional finite-mask check is restored. An additional explicit
range check reports probabilities outside `[0, 1]` before CUDA Bernoulli emits
an opaque device assertion. The check does not replace, clamp, or otherwise
alter valid probabilities.

## Equivalent speed improvements retained

### Final-layer likelihood calculation

The final-layer implementation retains the paper's Monte Carlo paired
keep/drop masks, `dropconnect_delta`, loss differences, averaging, and
probability formula. It evaluates the same binary-mask products with a matrix
multiplication instead of materializing several expanded tensors.

### Exact Hessian diagonal

The blockwise function evaluates the same individual second derivatives as the
paper's scalar loop. It groups derivative requests into autograd blocks and is
not a stochastic Hessian approximation.

### Orthogonal multi-model penalty

The paper expression

```text
||v w^T - I||_F^2
```

is evaluated using the exact identity

```text
||v||^2 ||w||^2 - 2(v dot w) + numel(v)
```

so the original option remains available without constructing a quadratic-size
identity matrix.

## Matrix totals

| Study | Runs |
|---|---:|
| `cifar100_variance` | 20 |
| `efficiency` | 51 |
| `efficiency_missing_datasets` | 240 |
| `warmstart` | 15 |
| `sensitivity` | 18 |
| `sparse_vd` | 36 |
| **Total** | **380** |

## Smoke tests

From the repository root:

```bash
python tests/smoke_paper_method_reviewer_ready.py
python tests/smoke_sparse_vd.py
python tests/smoke_reviewer_missing_datasets.py
python scripts/run_reviewer_serial.py --dry-run
```

These checks do not train models or download datasets.
