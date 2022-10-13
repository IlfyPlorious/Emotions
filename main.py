import os

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Lambda

from data.data_manager import DataManager
from util import ioUtil

config = {
    'spectrogram_dir': os.path.join(ioUtil.parent_dir, 'Spectrograms'),

    'batch_size': 30,

    'device': 'cuda',

    'train_size': 10,

    'valid_size': 10,

    'test_size': 10
}

transform = Lambda(lambda tensor: tensor.div(255)),
target_transform = Lambda(lambda label:
                          torch.zeros(len(ioUtil.labels.values()),
                                      dtype=torch.float)
                          .scatter_(dim=0,
                                    index=torch.tensor(
                                        ioUtil.labels.get(
                                            label)),
                                    value=1))

train_loader, validation_loader, test_loader = DataManager(config=config).get_train_eval_test_dataloaders()

train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features}")
print(f"Labels batch shape: {train_labels}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
