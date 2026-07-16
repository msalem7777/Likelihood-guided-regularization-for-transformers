import torch
from data_loader.dataloader_master import get_vit_dataloaders

# 1. Two independent calls must yield identical partitions
dl1 = get_vit_dataloaders('mnist', './data', batch_size=32, val_split=0.2, test_split=0.1, image_size=28)
dl2 = get_vit_dataloaders('mnist', './data', batch_size=32, val_split=0.2, test_split=0.1, image_size=28)

i1 = dl1['test'].dataset.indices
i2 = dl2['test'].dataset.indices
assert i1 == i2, "test split differs between calls - seeding not applied"

# 2. No overlap between train and test within one call
train_idx = set(dl1['train'].dataset.indices)
test_idx = set(i1)
assert not (train_idx & test_idx), "train/test overlap!"

# 3. Different seed must yield a different partition
dl3 = get_vit_dataloaders('mnist', './data', batch_size=32, val_split=0.2, test_split=0.1, image_size=28, split_seed=7)
assert dl3['test'].dataset.indices != i1, "split_seed has no effect"

# 4. Fast label count matches direct index-based count (first 2000 indices)
sub = dl1['val'].dataset
base_targets = torch.as_tensor(sub.dataset.targets)
fast_first2000 = int((base_targets[sub.indices[:2000]] == 1).sum())
slow_first2000 = sum(1 for idx in sub.indices[:2000] if int(base_targets[idx]) == 1)
assert fast_first2000 == slow_first2000, "fast label count disagrees with direct count"

print("all checks passed | test size:", len(i1), "| val label==1 count (first 2000):", fast_first2000)