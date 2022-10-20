import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.transforms import Lambda

from data.data_manager import DataManager
from util import ioUtil
from networks.EmotionsNetwork2LinLayers import EmotionsNetwork2LinLayers

# train_loader, validation_loader, test_loader = DataManager(config=config).get_train_eval_test_dataloaders()

# train_features, train_labels = next(iter(train_loader))
# print(f"Feature batch shape: {len(train_features)}")
# print(f"Labels batch shape: {len(train_labels)}")
#
# for img in train_features:
#     flatten = nn.Flatten()
#     print(f'Image shape: {flatten(img).shape}')

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
#
# model = EmotionsNetwork().to(device)
# # print(model)
# X = torch.rand(1, 369, 496, device=device)
# logits = model(X.cuda())
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class:", list(ioUtil.labels.keys())[list(ioUtil.labels.values()).index(y_pred)])


