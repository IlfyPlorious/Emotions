import os

import torch
from torch import nn

from data.data_manager import DataManager
from networks.EmotionsNetwork2LinLayers import EmotionsNetwork2LinLayers
from util import ioUtil

learning_rate = 1e-5
batch_size = 5
epochs = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

torch.cuda.empty_cache()

model = EmotionsNetwork2LinLayers().to(device)

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

config = {
    'spectrogram_dir': os.path.join(ioUtil.parent_dir, 'Spectrograms'),

    'batch_size': batch_size,

    'device': 'cuda',

    'train_split': 2,

    'valid_split': 5,
}

epoch_loss_data = []

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.cuda()
        y = y.cuda()
        pred = model(X).cuda()
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        x_length = len(X)
        loss, current = loss.item(), batch * x_length
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader.dataset) // dataloader.batch_size
    test_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.cuda()
        y = y.cuda()
        pred = model(X).cuda()
        loss = loss_fn(pred, y)
        test_loss += loss.item()
        correct += (pred.argmax(axis=1) == y.argmax(axis=1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


train_loader, validation_loader, test_loader = DataManager(config=config).get_train_eval_test_dataloaders()

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    accuracy, loss = test_loop(test_loader, model, loss_fn)
    epoch_loss_data.append(loss)

print(f"Loss data: {sum(epoch_loss_data) / len(epoch_loss_data)}")
print("Done!")
