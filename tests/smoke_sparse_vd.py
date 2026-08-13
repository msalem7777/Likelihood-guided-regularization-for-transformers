"""CPU smoke tests for the Sparse Variational Dropout reviewer baseline.

Run from the repository root:

    python tests/smoke_sparse_vd.py

The test intentionally avoids downloading datasets or starting a real training
job. It checks the new mathematical layer, its gradients, deterministic pruning,
ViT integration, and the exact six-run reviewer matrix.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from examples.reviewer_experiments import build_args, build_study
from transformer_layers.bbb_ViT import VisionTransformerWithBBB
from transformer_layers.sparse_vd_linear import SparseVDLinear
from utils.sparse_vd import sparse_vd_kl_weight, sparse_vd_learning_rate


def assert_finite_gradient(parameter: torch.nn.Parameter, name: str) -> None:
    """Fail clearly if a required variational parameter did not receive a gradient."""
    assert parameter.grad is not None, f"{name} did not receive a gradient"
    assert torch.isfinite(parameter.grad).all(), f"{name} gradient is not finite"
    assert parameter.grad.abs().sum().item() > 0.0, f"{name} gradient is identically zero"


def smoke_layer_forward_backward() -> None:
    """Check local reparameterization, KL evaluation, and variational gradients."""
    torch.manual_seed(7)
    layer = SparseVDLinear(4, 3)
    layer.train()

    x = torch.randn(5, 4)
    target = torch.randn(5, 3)
    prediction = layer(x)
    kl = layer.kl_divergence()
    loss = F.mse_loss(prediction, target) + kl / x.shape[0]
    loss.backward()

    assert prediction.shape == (5, 3)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(kl)
    assert kl.item() >= 0.0
    assert_finite_gradient(layer.mean_weight, "mean_weight")
    assert_finite_gradient(layer.log_sigma_weight, "log_sigma_weight")


def smoke_author_equations() -> None:
    """Check the dense-layer equations against the uploaded author source."""
    layer = SparseVDLinear(
        2,
        1,
        bias=False,
        threshold=3.0,
        log_alpha_clip=8.0,
        train_clip=True,
    )
    with torch.no_grad():
        layer.mean_weight.copy_(torch.tensor([[1e-8, 1.0]]))
        layer.log_sigma_weight.copy_(torch.tensor([[4.0, -10.0]]))

    expected_log_alpha = torch.tensor([[8.0, -8.0]])
    expected_variance = torch.exp(expected_log_alpha) * layer.mean_weight.square()
    expected_training_mean = torch.tensor([[0.0, 1.0]])

    assert torch.allclose(layer.log_alpha(), expected_log_alpha)
    assert torch.allclose(layer.posterior_variance_weight(), expected_variance)
    assert torch.equal(layer.training_mean_weight(), expected_training_mean)

    k1, k2, k3 = 0.63576, 1.87320, 1.48695
    reference_negative_kl = (
        k1 * torch.sigmoid(k2 + k3 * expected_log_alpha)
        - 0.5 * F.softplus(-expected_log_alpha)
        - k1
    )
    assert torch.allclose(
        layer.kl_divergence(),
        -reference_negative_kl.sum(),
        rtol=1e-6,
        atol=1e-7,
    )


def smoke_deterministic_pruning() -> None:
    """Check that evaluation exactly zeros weights whose log(alpha) exceeds 3."""
    layer = SparseVDLinear(2, 1, bias=False, threshold=3.0)
    with torch.no_grad():
        layer.mean_weight.copy_(torch.tensor([[1.0, 1e-8]]))
        layer.log_sigma_weight.copy_(torch.tensor([[-10.0, 0.0]]))

    layer.eval()
    x = torch.tensor([[2.0, 5.0]])
    output = layer(x)
    expected = torch.tensor([[2.0]])
    pruned, total = layer.sparsity_stats()

    assert torch.allclose(output, expected, atol=1e-6)
    assert (pruned, total) == (1, 2)


def smoke_vit_integration() -> None:
    """Check that every ViT projection is replaced and a full pass is differentiable."""
    torch.manual_seed(11)
    model = VisionTransformerWithBBB(
        img_size=8,
        patch_size=4,
        num_classes=3,
        embed_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        linear_layer_cls=SparseVDLinear,
        linear_layer_kwargs={
            "threshold": 3.0,
            "log_sigma_init": -5.0,
            "log_alpha_clip": 8.0,
            "train_clip": False,
        },
    )
    sparse_layers = [module for module in model.modules() if isinstance(module, SparseVDLinear)]
    assert len(sparse_layers) == 9, f"expected 9 SparseVDLinear layers, found {len(sparse_layers)}"

    model.train()
    logits = model(torch.randn(2, 3, 8, 8))
    objective = F.cross_entropy(logits, torch.tensor([0, 2]))
    objective = objective + sum(layer.kl_divergence() for layer in sparse_layers) / 2
    objective.backward()

    assert logits.shape == (2, 3)
    for index, layer in enumerate(sparse_layers):
        assert_finite_gradient(layer.log_sigma_weight, f"layer[{index}].log_sigma_weight")


def smoke_reviewer_matrix() -> None:
    """Check the six prespecified jobs and their trainer-facing arguments."""
    specs = build_study("sparse_vd")
    assert len(specs) == 6
    assert [(spec.dataset, spec.seed) for spec in specs] == [
        ("mnist", 0),
        ("mnist", 1),
        ("mnist", 2),
        ("cifar100", 0),
        ("cifar100", 1),
        ("cifar100", 2),
    ]

    for spec in specs:
        args = build_args(spec, Path("reviewer_results"))
        assert args.method == "sparse_vd"
        assert args.train_epochs == 200
        assert args.ising_epochs == 0
        assert args.batch_size == 100
        assert args.sparse_vd_threshold == 3.0
        assert args.sparse_vd_log_sigma_init == -5.0
        assert args.sparse_vd_log_alpha_clip == 8.0
        assert args.sparse_vd_kl_delay_epochs == 5
        assert args.sparse_vd_kl_warmup_epochs == 15
        if spec.dataset == "mnist":
            assert args.learning_rate == 1e-3
            assert args.sparse_vd_train_clip is True
            assert args.sparse_vd_lr_schedule == "author_mnist_linear_to_zero"
        else:
            assert args.learning_rate == 1e-5
            assert args.sparse_vd_train_clip is False
            assert args.sparse_vd_lr_schedule == "author_cifar_linear_after_100"


def smoke_author_schedules() -> None:
    """Check exact zero-based epoch boundaries from ``nets/optpolicy.py``."""
    expected_kl = {
        0: 0.0,
        4: 0.0,
        5: 0.0,
        6: 1.0 / 15.0,
        19: 14.0 / 15.0,
        20: 1.0,
        199: 1.0,
    }
    for epoch, expected in expected_kl.items():
        assert math.isclose(sparse_vd_kl_weight(epoch), expected, abs_tol=1e-12)

    mnist_expected = {0: 1e-3, 100: 5e-4, 199: 5e-6}
    for epoch, expected in mnist_expected.items():
        actual = sparse_vd_learning_rate(
            epoch, 1e-3, "author_mnist_linear_to_zero"
        )
        assert math.isclose(actual, expected, rel_tol=1e-12)

    cifar_expected = {0: 1e-5, 100: 1e-5, 150: 5e-6, 199: 1e-7}
    for epoch, expected in cifar_expected.items():
        actual = sparse_vd_learning_rate(
            epoch, 1e-5, "author_cifar_linear_after_100"
        )
        assert math.isclose(actual, expected, rel_tol=1e-12)


def main() -> None:
    smoke_layer_forward_backward()
    smoke_author_equations()
    smoke_deterministic_pruning()
    smoke_vit_integration()
    smoke_reviewer_matrix()
    smoke_author_schedules()
    print("SPARSE VD SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
