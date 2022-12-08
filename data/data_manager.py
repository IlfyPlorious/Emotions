import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda

from data.base_dataset import SpectrogramsDataset
from util import ioUtil


class DataManagerSpectrograms:
    """Manager class for data loaders.

Config argument is a dictionary that contains the following:

spectrogram_dir -> path_to_spectograms_dir

batch_size -> size of the batch

train_epochs -> epochs count

device -> cuda if gpu else cpu

train_split -> division part for train size = dataset // train_split

valid_split -> division part for validation size = dataset // valid_split

"""
    def __init__(self, config):
        self.config = config
        self.transform = Lambda(lambda tensor: ioUtil.map_tensor_to_0_1(tensor))
        self.transform_target = Lambda(lambda label:
                                       torch.zeros(len(ioUtil.labels.values()),
                                                   dtype=torch.float)
                                       .scatter_(dim=0,
                                                 index=torch.tensor(
                                                     ioUtil.labels.get(
                                                         label)),
                                                 value=1))

    def get_dataloader(self):
        dataset = SpectrogramsDataset(self.config, self.transform, self.transform_target)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True if self.config['device'] == 'cuda' else False
        )
        return dataloader

    def get_train_eval_dataloaders(self):
        np.random.seed(707)

        dataset = SpectrogramsDataset(self.config, self.transform, self.transform_target)
        dataset_size = len(dataset)

        ## SPLIT DATASET
        train_split = self.config['train_split']
        train_size = dataset_size // train_split
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
        train_split = self.config['train_split']
        valid_split = self.config['valid_split']

        train_size = dataset_size // train_split
        valid_size = dataset_size // valid_split

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