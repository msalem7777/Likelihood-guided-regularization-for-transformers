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

def get_vit_dataloaders(dataset_name, data_dir, batch_size=32, val_split=0.2, test_split=0.1, image_size=224):
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
    # Define standard transformations for Vision Transformers
    vit_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to image_size x image_size
        transforms.ToTensor(),  # Convert to PyTorch Tensor
        To3Channels(),  # Convert single-channel images to 3 channels
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
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
    test_size = int(total_size * test_split)
    val_size = int((total_size - test_size) * val_split)
    train_size = total_size - test_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }

    return dataloaders

