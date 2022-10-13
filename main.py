import torch

from data import base_dataset

train_features, train_labels = next(iter(DataSets.train_dataloader))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


