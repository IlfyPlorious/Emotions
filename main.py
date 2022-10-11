import matplotlib.pyplot as plt
import torch
import torchvision.transforms

from torchvision.transforms import Lambda

import DataSets

train_features, train_labels = next(iter(DataSets.train_dataloader))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


