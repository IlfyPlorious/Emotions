import os.path

from torch.utils.data import Dataset
from torchvision.io import read_image


class SpectrogramsDataset(Dataset):
    def __init__(self, config):
        self.transform = config['transform']
        self.target_transform = config['transform_target']
        self.img_dir = config['spectrogram_dir']

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
