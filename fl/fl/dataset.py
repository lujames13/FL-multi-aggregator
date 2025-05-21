import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Default transform for CIFAR10
pytorch_transforms = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def _partition_indices(dataset_size, num_partitions, partition_id, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(dataset_size)
    part_size = dataset_size // num_partitions
    start = partition_id * part_size
    end = start + part_size if partition_id < num_partitions - 1 else dataset_size
    return indices[start:end]

def load_data(partition_id: int, num_partitions: int):
    """Load partitioned CIFAR10 data using PyTorch, with local caching."""
    # Download (if not already) and load CIFAR10
    train_set = datasets.CIFAR10(
        root=CACHE_DIR, train=True, download=True, transform=pytorch_transforms
    )
    test_set = datasets.CIFAR10(
        root=CACHE_DIR, train=False, download=True, transform=pytorch_transforms
    )
    # Partition train set
    train_indices = _partition_indices(len(train_set), num_partitions, partition_id)
    test_indices = _partition_indices(len(test_set), num_partitions, partition_id)
    train_subset = Subset(train_set, train_indices)
    test_subset = Subset(test_set, test_indices)
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=32)
    return trainloader, testloader 