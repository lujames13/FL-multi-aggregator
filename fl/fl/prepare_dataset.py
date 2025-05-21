import os
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

CACHE_DIR = os.path.join(os.path.dirname(__file__), "hf_cache")
NUM_PARTITIONS = 10  # 可根據需求調整

os.makedirs(CACHE_DIR, exist_ok=True)

partitioner = IidPartitioner(num_partitions=NUM_PARTITIONS)
fds = FederatedDataset(
    dataset="uoft-cs/cifar10",
    partitioners={"train": partitioner},
)

for partition_id in range(NUM_PARTITIONS):
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    train_data = partition_train_test["train"]
    test_data = partition_train_test["test"]
    torch.save(train_data, os.path.join(CACHE_DIR, f"partition_{partition_id}_train.pt"))
    torch.save(test_data, os.path.join(CACHE_DIR, f"partition_{partition_id}_test.pt"))

print(f"Saved {NUM_PARTITIONS} partitions to {CACHE_DIR}") 