import torch
from data_loader.dataloader_master import get_vit_dataloaders

def main():
    # workers=0 path must still construct (prefetch_factor must not be passed)
    dl0 = get_vit_dataloaders('mnist', './data', batch_size=32, val_split=0.2,
                              test_split=0.1, image_size=28, num_workers=0)
    xb, yb = next(iter(dl0['train']))
    assert xb.shape == (32, 3, 28, 28), f"unexpected batch shape {xb.shape}"

    # workers=2 path constructs and yields identical split
    dl2 = get_vit_dataloaders('mnist', './data', batch_size=32, val_split=0.2,
                              test_split=0.1, image_size=28, num_workers=2)
    assert dl0['test'].dataset.indices == dl2['test'].dataset.indices, "split changed with workers"
    xb2, yb2 = next(iter(dl2['train']))
    assert xb2.shape == (32, 3, 28, 28)

    print("all checks passed | pin_memory =", dl2['train'].pin_memory)

if __name__ == "__main__":
    main()