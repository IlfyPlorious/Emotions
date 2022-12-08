import os.path

import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramsDataset(Dataset):
    def __init__(self, config, transform=None, transform_target=None):
        self.transform = transform
        self.target_transform = transform_target
        self.img_dir = config['spectrogram_dir']

    def __len__(self):
        length = 0
        for dir in os.listdir(self.img_dir):
            for _ in os.listdir(os.path.join(self.img_dir, dir)):
                length += 1
        return length

    def get_spectrogram_paths_list(self):
        spectrogram_paths = []
        for dir in os.listdir(self.img_dir):
            for spectrogram in os.listdir(os.path.join(self.img_dir, dir)):
                spectrogram_paths.append((os.path.join(self.img_dir, dir, spectrogram), spectrogram.split('_')[2]))

        return spectrogram_paths

    def __getitem__(self, idx):
        spectrograms = self.get_spectrogram_paths_list()
        spectrogram_path = spectrograms[idx]
        image_path = spectrogram_path[0]  # element in tuple at index 0 is the spectrogram path
        label = spectrogram_path[1]  # element in tuple at index 1 is the label
        image = np.load(image_path)
        image = torch.tensor(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # image is initially [channels, width, height], but plt.imshow() needs [width, height, channels]
        # image = torch.permute(image, [1, 2, 0])

        return image, label


class VideosDataset(Dataset):
    def __init__(self, config, transform=None, transform_target=None):
        self.transform = transform
        self.transform_target = transform_target
        self.img_dir = config['video_dir']
