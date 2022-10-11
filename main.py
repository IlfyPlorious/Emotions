import matplotlib.pyplot as plt
import torch
import torchvision.transforms

from torchvision.transforms import Lambda

import DataSets

train_features, train_labels = next(iter(DataSets.train_dataloader))

img = torch.permute(train_features[0].squeeze(), (1, 2, 0))
label = train_labels[0]
print(train_features)
print(train_labels)
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

