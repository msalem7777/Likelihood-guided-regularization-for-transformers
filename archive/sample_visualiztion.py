"""
run_bulk_experiments.py
─────────────────────────────────────────────────────────────
Run pVisionTransformerTrainer on every combination of:
    • dataset            ∈ {mnist, fashionmnist, cifar10, cifar100, tinyimagenet}
    • ising_type         ∈ {LM_saliency_scores, diag_saliency_scores, no_saliency_scores}
    • val_split fraction ∈ {0.89, 0.85, 0.60}
30 independent seeds for each setting are executed.
Results are appended to bulk_results.csv in the working directory.
"""
import os
os.chdir('C:/My files/repos/Likelihood-guided-regularization-for-transformers')

import itertools, random, os, time, importlib, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from argparse import Namespace
from tqdm.auto import tqdm  
from scipy.stats import entropy
from main.VisionTransformer_Trainer import VisionTransformerTrainer
from data_loader.cifar_downloader import *
from data_loader.mnist_fmnist_downloader import *


def ensure_data(dataset):

    """Skip downloading if known file/folder already exists."""
    expected_paths = {
        "mnist":        "./mnist/MNIST/raw/train-images-idx3-ubyte",
        "fashionmnist": "./fashionmnist/FashionMNIST/raw/train-images-idx3-ubyte",
        "cifar10":      "./cifar10/cifar-10-batches-py/data_batch_1",
        "cifar100":     "./cifar100/cifar-100-python/train",
    }
    
    if os.path.exists(expected_paths[dataset]):
        return  # Data is already present

    mod = {
        "mnist":        "data_loader.mnist_fmnist_downloader",
        "fashionmnist": "data_loader.mnist_fmnist_downloader",
        "cifar10":      "data_loader.cifar_downloader",
        "cifar100":     "data_loader.cifar_downloader",
    }[dataset]

    importlib.import_module(mod)  # triggers download on import


import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100

def visualize_samples(dataset_name, n_samples=10):
    """Visualize 10 samples from each dataset, preferably from different classes."""
    # Dataset loading logic
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == "mnist":
        dataset = MNIST(root="./mnist", train=True, download=True, transform=transform)
    elif dataset_name == "fashionmnist":
        dataset = FashionMNIST(root="./fashionmnist", train=True, download=True, transform=transform)
    elif dataset_name == "cifar10":
        dataset = CIFAR10(root="./cifar10", train=True, download=True, transform=transform)
    elif dataset_name == "cifar100":
        dataset = CIFAR100(root="./cifar100", train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Organize samples by class
    samples_by_class = {i: [] for i in range(len(dataset.classes))}
    for img, label in dataset:
        if len(samples_by_class[label]) < n_samples:
            samples_by_class[label].append(img)

    # Visualization
    fig, axes = plt.subplots(len(dataset.classes), n_samples, figsize=(15, len(dataset.classes) * 2))
    fig.suptitle(f"Visualization of {n_samples} Samples from {dataset_name.capitalize()} Dataset", fontsize=16)

    for class_idx, class_samples in samples_by_class.items():
        for sample_idx, img_tensor in enumerate(class_samples):
            if sample_idx >= n_samples:  # Only visualize n_samples per class
                break
            ax = axes[class_idx, sample_idx]
            ax.imshow(img_tensor.permute(1, 2, 0).numpy() if img_tensor.ndim == 3 else img_tensor.numpy(), cmap="gray")
            ax.axis("off")
            if sample_idx == 0:
                ax.set_ylabel(dataset.classes[class_idx], fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"visualization_{dataset_name}.png")
    plt.show()

# Call visualization for each dataset
datasets = ["mnist", "fashionmnist", "cifar10", "cifar100"]
for ds in datasets:
    visualize_samples(ds)