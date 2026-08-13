"""Sparse Variational Dropout optimization policies.

These functions port the schedules used by the paper-author repository while
keeping them independent from the trainer.  Epochs are zero-based, matching
both the author code and the training loop in this repository.
"""

from __future__ import annotations


def sparse_vd_kl_weight(
    epoch: int,
    delay_epochs: int = 5,
    ramp_epochs: int = 15,
) -> float:
    """Return the author's delayed linear Sparse VD KL weight.

    With the published defaults, epochs 0 through 5 receive zero KL weight,
    epochs 6 through 19 increase by 1/15, and epoch 20 onward receives full
    KL weight.
    """
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative; received {epoch}")
    if delay_epochs < 0:
        raise ValueError(
            f"delay_epochs must be non-negative; received {delay_epochs}"
        )
    if ramp_epochs <= 0:
        raise ValueError(f"ramp_epochs must be positive; received {ramp_epochs}")

    return float(min(max((epoch - delay_epochs) / ramp_epochs, 0.0), 1.0))


def sparse_vd_learning_rate(
    epoch: int,
    initial_lr: float,
    schedule: str,
    total_epochs: int = 200,
) -> float:
    """Return a learning rate from an author-repository Sparse VD schedule."""
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative; received {epoch}")
    if initial_lr <= 0.0:
        raise ValueError(f"initial_lr must be positive; received {initial_lr}")
    if total_epochs <= 0:
        raise ValueError(f"total_epochs must be positive; received {total_epochs}")

    if schedule == "author_mnist_linear_to_zero":
        multiplier = max(0.0, (total_epochs - epoch) / total_epochs)
    elif schedule == "author_cifar_linear_after_100":
        multiplier = max(0.0, min(2.0 - epoch / 100.0, 1.0))
    else:
        raise ValueError(f"Unknown Sparse VD learning-rate schedule: {schedule}")

    return float(initial_lr * multiplier)


def set_optimizer_learning_rate(optimizer, learning_rate: float) -> None:
    """Set every parameter group's learning rate to one validated value."""
    if learning_rate < 0.0:
        raise ValueError(
            f"learning_rate must be non-negative; received {learning_rate}"
        )
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
