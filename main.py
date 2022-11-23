import json
import os

import librosa
import numpy as np
import torchaudio.transforms
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

import data.data_manager
import torch

import networks.networks
import networks.res_net as res_net
from data import base_dataset
from trainer import Trainer
from util import ioUtil

config = json.load(open('config.json'))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = networks.networks.ResNet(block=res_net.BasicBlock, layers=[1, 1, 1, 1], num_classes=6).to(device)
train_dataloader, eval_dataloader = data.data_manager.DataManager(config).get_train_eval_dataloaders()
optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])

trainer = Trainer(model=model, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                  loss_fn=nn.CrossEntropyLoss(), criterion=None, optimizer=optimizer, config=config)

trainer.run()
