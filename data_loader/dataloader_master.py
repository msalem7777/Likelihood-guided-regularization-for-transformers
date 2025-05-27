#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import torch.optim as optim
import time
import random
from torch.nn.parallel import DataParallel
import pandas as pd

class To3Channels:
    """
    A custom transform to replicate a single-channel image across 3 channels.
    """

    def __call__(self, img):
        return img.repeat(3, 1, 1)  # Repeat the single channel to create 3 channels

def get_vit_dataloaders(dataset_name, data_dir, batch_size=32, val_split=0.2, test_split=0.1, image_size=224, num_workers = 0):
    """
    Creates DataLoaders for a dataset compatible with Vision Transformers.

    Args:
        dataset_name (str): Name of the dataset ('cifar10', 'cifar100', 'mnist', 'fashionmnist', 'tinyimagenet').
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the dataloaders.
        val_split (float): Fraction of the training set to use for validation.
        test_split (float): Fraction of the full dataset to use for testing.
        image_size (int): Size to which images will be resized.

    Returns:
        dict: A dictionary containing train, validation, and test DataLoaders.
    """
    # Conditional transforms
    if dataset_name in ['mnist', 'fashionmnist']:
        vit_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            To3Channels(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    else:
        vit_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    # Load the dataset
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir, download=False, transform=vit_transforms)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=data_dir, download=False, transform=vit_transforms)
    elif dataset_name == 'mnist':
        dataset = datasets.MNIST(root=data_dir, download=False, transform=vit_transforms)
    elif dataset_name == 'fashionmnist':
        dataset = datasets.FashionMNIST(root=data_dir, download=False, transform=vit_transforms)
    elif dataset_name == 'tinyimagenet':
        dataset = datasets.ImageFolder(root=f"{data_dir}/tiny-imagenet-200/train", transform=vit_transforms)
    else:
        raise ValueError("Unsupported dataset name. Choose from ['cifar10', 'cifar100', 'mnist', 'fashionmnist', 'tinyimagenet'].")

    # Calculate sizes for splitting
    total_size = len(dataset)
    # Apply all splits as fractions of the *total* dataset
    test_size = int(total_size * test_split)
    val_size  = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    assert train_size > 0, "Training set size is zero! Adjust your val/test splits."


    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=min(1024, val_size), shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=min(1024, test_size), shuffle=False, num_workers=num_workers)
    }

    return dataloaders

