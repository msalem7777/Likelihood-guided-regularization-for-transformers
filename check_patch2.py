import torch
import torch.nn.functional as F
from transformer_layers.bbb_linear import BBBLinear

torch.manual_seed(0)
layer = BBBLinear(16, 8)
x = torch.randn(4, 16)

# --- eval path: must be exactly (1-p) * mean_weight, deterministic ---
layer.eval()
out1 = layer(x)
out2 = layer(x)
assert torch.equal(out1, out2), "eval path not deterministic"
expected = F.linear(x, (1 - layer.p) * layer.mean_weight, layer.mean_bias)
assert torch.equal(out1, expected), "eval path formula changed"

# --- eval path with a custom mask ---
mask = torch.rand_like(layer.mean_weight)
layer.apply_custom_dropout_prob(mask)
out3 = layer(x)
expected3 = F.linear(x, (1 - mask) * layer.mean_weight, layer.mean_bias)
assert torch.equal(out3, expected3), "eval masked path formula changed"

# --- training path: runs without error, output finite, right shape ---
layer.train()

class FakeTracker:  # exercise both phase branches
    current_epoch = 'pilot'
layer.epoch_tracker = FakeTracker()
out4 = layer(x)
assert out4.shape == (4, 8) and torch.all(torch.isfinite(out4))

FakeTracker.current_epoch = 'fine-tuning'
out5 = layer(x)
assert out5.shape == (4, 8) and torch.all(torch.isfinite(out5))

# --- fine-tuning branch is deterministic in the mask (only weight noise varies):
#     with std ~ e^-4 ≈ 0.018, two calls differ only slightly ---
print("all checks passed")