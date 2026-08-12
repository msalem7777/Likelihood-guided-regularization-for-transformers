"""CPU smoke tests for the Sparse Variational Dropout reviewer baseline.

Run from the repository root:

    python tests/smoke_sparse_vd.py

The test intentionally avoids downloading datasets or starting a real training
job. It checks the new mathematical layer, its gradients, deterministic pruning,
ViT integration, and the exact six-run reviewer matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from examples.reviewer_experiments import build_args, build_study
from transformer_layers.bbb_ViT import VisionTransformerWithBBB
from transformer_layers.sparse_vd_linear import SparseVDLinear


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
    assert_finite_gradient(layer.mean_weight, "mean_weight")
    assert_finite_gradient(layer.log_sigma_weight, "log_sigma_weight")


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
        linear_layer_kwargs={"threshold": 3.0, "log_sigma_init": -5.0},
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
        assert args.train_epochs == 15
        assert args.ising_epochs == 0
        assert args.sparse_vd_threshold == 3.0
        assert args.sparse_vd_log_sigma_init == -5.0
        assert args.sparse_vd_kl_warmup_epochs == 15


def main() -> None:
    smoke_layer_forward_backward()
    smoke_deterministic_pruning()
    smoke_vit_integration()
    smoke_reviewer_matrix()
    print("SPARSE VD SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
