import json

from torch import nn

import data.data_manager
import torch

import networks.networks
from data import base_dataset
from trainer import Trainer

config = json.load(open('config.json'))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = networks.networks.EmotionsNetworkV3().to(device)
train_dataloader, eval_dataloader = data.data_manager.DataManager(config).get_train_eval_dataloaders()
optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])

trainer = Trainer(model=model, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                  loss_fn=nn.CrossEntropyLoss(), criterion=None, optimizer=optimizer, config=config)

trainer.run()
