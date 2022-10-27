import json
import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.transforms import Lambda

import trainer
from data.data_manager import DataManager
from util import ioUtil
from networks.networks import EmotionsNetwork2LinLayers
from networks.networks import EmotionsNetwork2Conv2Layers

config = json.load(open('config.json'))

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

trainer.run(EmotionsNetwork2Conv2Layers().to("cuda" if torch.cuda.is_available() else "cpu"))

