import numpy as np
import torch
from torch.utils.data import DataLoader

from data.base_dataset import SpectrogramsDataset


class DataManager:
    """Manager class for data loaders.

Config argument is a dictionary that contains the following:

spectrogram_dir -> path_to_spectograms_dir

batch_size -> size of the batch

device -> cuda if gpu else cpu

train_size -> number of train samples

valid_size -> number of validation samples

test_size -> number of test samples

"""

    def __init__(self, config, transform=None, transform_target=None):
        self.config = config
        self.transform = transform
        self.transform_target = transform_target

    def get_dataloader(self):
        dataset = SpectrogramsDataset(self.config['spectrogram_dir'], self.transform, self.transform_target)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True if self.config['device'] == 'cuda' else False
        )
        return dataloader

    def get_train_eval_dataloaders(self):
        np.random.seed(707)

        dataset = SpectrogramsDataset(self.config['spectrogram_dir'], self.transform, self.transform_target)
        dataset_size = len(dataset)

        ## SPLIT DATASET
        train_split = self.config['train_size']
        train_size = int(train_split * dataset_size)
        validation_size = dataset_size - train_size

        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        temp = int(train_size + validation_size)
        val_indices = indices[train_size:temp]

        ## DATA LOARDER ##
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=self.config['batch_size'],
                                                   sampler=train_sampler,
                                                   pin_memory=True if self.config['device'] == 'cuda' else False)

        validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=self.config['batch_size'],
                                                        sampler=valid_sampler,
                                                        pin_memory=True if self.config['device'] == 'cuda' else False)

        return train_loader, validation_loader

    def get_train_eval_test_dataloaders(self):
        np.random.seed(707)

        dataset = SpectrogramsDataset(self.config, self.transform, self.transform_target)
        dataset_size = len(dataset)

        ## SPLIT DATASET
        train_split = self.config['train_size']
        valid_split = self.config['valid_size']
        test_split = self.config['test_size']

        train_size = int(train_split * dataset_size)
        valid_size = int(valid_split * dataset_size)
        test_size = dataset_size - train_size - valid_size

        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:(train_size + valid_size)]
        test_indices = indices[(train_size + valid_size):]

        ## DATA LOARDER ##
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=self.config['batch_size'],
                                                   sampler=train_sampler,
                                                   pin_memory=True if self.config['device'] == 'cuda' else False)

        validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=self.config['batch_size'],
                                                        sampler=valid_sampler,
                                                        pin_memory=True if self.config['device'] == 'cuda' else False)

        test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.config['batch_size'],
                                                  sampler=test_sampler,
                                                  pin_memory=True if self.config['device'] == 'cuda' else False)

        return train_loader, validation_loader, test_loader
