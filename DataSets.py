import os.path

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader

import ioUtil


class SpectrogramsDataset(Dataset):
    def __init__(self, img_dir='Spectrograms', transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = os.path.join(ioUtil.parent_dir, img_dir)

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        label = img_path.split('_')[2]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_dataloader = DataLoader(
    SpectrogramsDataset(
        transform=Lambda(lambda tensor:
                         tensor.div(255)),
        target_transform=Lambda(lambda label: torch.zeros(
            len(ioUtil.labels.values()), dtype=torch.float).scatter_(dim=0,
                                                                     index=torch.tensor(ioUtil.labels.get(label)),
                                                                     value=1))),
    batch_size=10,
    shuffle=True)
test_dataloader = DataLoader(
    SpectrogramsDataset(
        transform=Lambda(lambda tensor:
                         tensor.div(255)),
        target_transform=Lambda(lambda label: torch.zeros(
            len(ioUtil.labels.values()), dtype=torch.float).scatter_(dim=0,
                                                                     index=torch.tensor(ioUtil.labels.get(label)),
                                                                     value=1))),
    batch_size=10,
    shuffle=True)
