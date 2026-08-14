#!/usr/bin/env python
"""Smoke tests for the reviewer-ready, paper-method restoration.

Run from the repository root with:

    python tests/smoke_speed_equivalence.py

The tests compare optimized operations directly against small reference
implementations from commit 4dec1b37. They also check the complete 380-run
reviewer matrix and the source-level training-order invariants that distinguish
the paper method from the later reviewer-modified method. They do not train a
model or download any data.
"""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from main.VisionTransformer_Trainer import (  # noqa: E402
    exact_hessian_diag,
    fast_compute_weight_dropout,
)
from examples.reviewer_experiments import build_study  # noqa: E402


class _FinalLayer:
    """Minimal final-layer object required by fast_compute_weight_dropout."""

    def __init__(self, mean_weight):
        self.mean_weight = mean_weight


def _paper_reference_fast_compute(
    final_layer,
    activations,
    targets,
    dropconnect_delta,
    epsilon,
    masks_keep_all,
    masks_drop_all,
):
    """Direct transcription of the paper commit's expanded-tensor method."""
    with torch.no_grad():
        weights = final_layer.mean_weight
        batch_size, input_features = activations.shape
        output_features = weights.shape[0]
        monte_carlo_samples = masks_keep_all.shape[1]

        activations_expanded = activations.view(
            batch_size,
            1,
            1,
            input_features,
        ).expand(
            batch_size,
            input_features,
            monte_carlo_samples,
            input_features,
        )

        weights_expanded = weights.view(
            1,
            output_features,
            1,
            input_features,
        ).expand(
            input_features,
            output_features,
            monte_carlo_samples,
            input_features,
        ).transpose(0, 1)

        keep_masks = masks_keep_all.unsqueeze(0)
        drop_masks = masks_drop_all.unsqueeze(0)

        logits_keep = torch.einsum(
            'bdmd,cdmd->bdmc',
            activations_expanded * keep_masks,
            weights_expanded * keep_masks,
        )
        logits_drop = torch.einsum(
            'bdmd,cdmd->bdmc',
            activations_expanded * drop_masks,
            weights_expanded * drop_masks,
        )

        targets_flat = targets.view(batch_size, 1, 1).expand(
            -1,
            input_features,
            monte_carlo_samples,
        ).reshape(-1)

        loss_keep = -F.cross_entropy(
            logits_keep.reshape(-1, output_features),
            targets_flat,
            reduction='none',
        ).view(batch_size, input_features, monte_carlo_samples)
        loss_drop = -F.cross_entropy(
            logits_drop.reshape(-1, output_features),
            targets_flat,
            reduction='none',
        ).view(batch_size, input_features, monte_carlo_samples)

        loss_difference = (
            loss_keep - loss_drop
        ).mean(dim=0).mean(dim=1).unsqueeze(0).expand(
            output_features,
            input_features,
        )

        delta_term = torch.log(
            torch.tensor(
                dropconnect_delta / (1.0 - dropconnect_delta),
                device=activations.device,
                dtype=activations.dtype,
            )
        )
        output_logits = -2.0 * (0.5 * loss_difference) + delta_term
        output_logits = torch.nan_to_num(
            output_logits,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        return torch.sigmoid(output_logits).clamp(
            min=epsilon,
            max=1.0 - epsilon,
        )


def smoke_fast_compute_equivalence():
    """The low-memory logits calculation must match the paper calculation."""
    torch.manual_seed(7)

    batch_size = 3
    input_features = 5
    output_features = 4
    monte_carlo_samples = 6

    activations = torch.randn(batch_size, input_features, dtype=torch.float64)
    weights = torch.randn(output_features, input_features, dtype=torch.float64)
    targets = torch.tensor([0, 2, 3], dtype=torch.long)

    masks_keep_all = torch.randint(
        low=0,
        high=2,
        size=(input_features, monte_carlo_samples, input_features),
        dtype=torch.float64,
    )
    diagonal_indices = torch.arange(input_features)
    masks_keep_all[diagonal_indices, :, diagonal_indices] = 1.0

    masks_drop_all = masks_keep_all.clone()
    masks_drop_all[diagonal_indices, :, diagonal_indices] = 0.0

    final_layer = _FinalLayer(weights)
    dropconnect_delta = 0.1
    epsilon = 1e-9

    expected = _paper_reference_fast_compute(
        final_layer,
        activations,
        targets,
        dropconnect_delta,
        epsilon,
        masks_keep_all,
        masks_drop_all,
    )
    actual = fast_compute_weight_dropout(
        final_layer=final_layer,
        activations=activations,
        targets=targets,
        dropconnect_delta=dropconnect_delta,
        epsilon=epsilon,
        masks_keep_all=masks_keep_all,
        masks_drop_all=masks_drop_all,
        mc_samples=monte_carlo_samples,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def smoke_exact_hessian_equivalence():
    """The blockwise Hessian diagonal must match the original scalar loop."""
    parameter = torch.tensor(
        [[0.3, -0.7], [1.2, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss = (
        parameter.pow(4).sum()
        + 0.25 * parameter.sum().pow(2)
    )

    first_derivative = torch.autograd.grad(
        loss,
        parameter,
        create_graph=True,
    )[0]
    reference_diagonal = torch.empty_like(parameter).reshape(-1)

    for index in range(parameter.numel()):
        second_derivative = torch.autograd.grad(
            first_derivative.reshape(-1)[index],
            parameter,
            retain_graph=True,
        )[0]
        reference_diagonal[index] = second_derivative.reshape(-1)[index]

    actual_diagonal = exact_hessian_diag(
        loss,
        parameter,
        block_size=2,
    )

    torch.testing.assert_close(
        actual_diagonal,
        reference_diagonal.view_as(parameter),
        rtol=1e-10,
        atol=1e-10,
    )


def smoke_orthogonal_penalty_identity():
    """The low-memory orthogonal penalty must equal the paper matrix form."""
    torch.manual_seed(13)
    left = torch.randn(7, dtype=torch.float64)
    right = torch.randn(7, dtype=torch.float64)

    paper_value = torch.norm(
        torch.outer(left, right) - torch.eye(7, dtype=torch.float64),
        p=2,
    ).square()
    optimized_value = (
        left.square().sum() * right.square().sum()
        - 2.0 * torch.dot(left, right)
        + left.numel()
    )

    torch.testing.assert_close(
        optimized_value,
        paper_value,
        rtol=1e-10,
        atol=1e-10,
    )


def smoke_complete_reviewer_matrix():
    """All six reviewer studies must still contain exactly 380 runs."""
    expected_counts = {
        'cifar100_variance': 20,
        'efficiency': 51,
        'efficiency_missing_datasets': 240,
        'warmstart': 15,
        'sensitivity': 18,
        'sparse_vd': 36,
    }

    observed_counts = {
        study: len(build_study(study))
        for study in expected_counts
    }
    assert observed_counts == expected_counts
    assert sum(observed_counts.values()) == 380


def smoke_paper_training_order_source():
    """Reject reintroduction of the reviewer-modified LM/cache ordering."""
    trainer_source = (
        REPOSITORY_ROOT
        / 'main'
        / 'VisionTransformer_Trainer.py'
    ).read_text(encoding='utf-8')

    assert '_lm_grad_cache' not in trainer_source
    assert '_lm_input_cache' not in trainer_source
    assert 'grad = final_layer.mean_weight.grad' in trainer_source
    assert 'curr_param.grad' in trainer_source

    final_epoch_condition = (
        'self.args.train_epochs\n'
        '                        + self.args.ising_epochs\n'
        '                        + self.args.addtl_ft\n'
        '                        - 1'
    )
    assert final_epoch_condition in trainer_source

    zero_all_marker = (
        'for optimizer in model_optim:\n'
        '                        optimizer.zero_grad()\n\n'
        '                    loss.backward()'
    )
    assert zero_all_marker in trainer_source


def main():
    smoke_fast_compute_equivalence()
    smoke_exact_hessian_equivalence()
    smoke_orthogonal_penalty_identity()
    smoke_complete_reviewer_matrix()
    smoke_paper_training_order_source()
    print('All reviewer-ready paper-method smoke tests passed.')


if __name__ == '__main__':
    main()
