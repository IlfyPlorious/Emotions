import os

import torch
from torch import nn

from data.data_manager import DataManager
from networks.EmotionsNetwork import EmotionsNetwork
from util import ioUtil

learning_rate = 1e-3
batch_size = 64
epochs = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = EmotionsNetwork().to('cpu')

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

config = {
    'spectrogram_dir': os.path.join(ioUtil.parent_dir, 'Spectrograms'),

    'batch_size': 30,

    'device': 'cuda',

    'train_size': 10,

    'valid_size': 10,

    'test_size': 10
}


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.cuda())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


train_loader, validation_loader, test_loader = DataManager(config=config).get_train_eval_test_dataloaders()

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(validation_loader, model, loss_fn)
print("Done!")
