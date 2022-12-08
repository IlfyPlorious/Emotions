import json
import os

import librosa
import numpy as np
import torchaudio.transforms
import Torchvision.vision.torchvision as torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

import data.data_manager
import torch

import networks_files.networks
import networks_files.res_net as res_net
import util.VideoFileModel
from data import base_dataset
from trainer import Trainer
from util import ioUtil

config = json.load(open('config.json'))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = networks_files.networks.ResNet(block=res_net.BasicBlock, layers=[1, 1, 1, 1], num_classes=6).to(device)
train_dataloader, eval_dataloader = data.data_manager.DataManagerSpectrograms(config).get_train_eval_dataloaders()
optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])

trainer = Trainer(model=model, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                  loss_fn=nn.CrossEntropyLoss(), criterion=None, optimizer=optimizer, config=config)

trainer.run()

# stream = 'video'
# video_dir_path = config['video_dir_path']
# videos = os.listdir(video_dir_path)
# video_file_path = videos[0]
# video = torchvision.io.VideoReader(os.path.join(video_dir_path, video_file_path), stream)
# actor, sample, emotion, emotion_level = video_file_path.split('_')
# video_file = util.VideoFileModel.VideoFile(sample, actor, emotion, emotion_level, video)
#
# print(str(video_file))
