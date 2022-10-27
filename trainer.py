import json
import os

import torch
from torch import nn

from datetime import date

from data.data_manager import DataManager
from util import ioUtil

config = json.load(open('config.json'))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

torch.cuda.empty_cache()

# model = EmotionsNetwork2LinLayers().to(device)

# Initialize the loss function

epoch_loss_data = []


def train_loop(dataloader, model, loss_fn, optimizer, save_file):
    size = len(dataloader.dataset)
    for batch, (input, emotion_prediction) in enumerate(dataloader, 0):
        # Compute prediction and loss
        input = input.cuda()
        emotion_prediction = emotion_prediction.cuda()
        pred = model(input).cuda()
        loss = loss_fn(pred, emotion_prediction)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 2 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        x_length = len(input)
        loss, current = loss.item(), batch * x_length
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        save_file.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")


def test_loop(dataloader, model, loss_fn, save_file):
    size = len(dataloader.dataset)
    num_batches = len(dataloader.dataset) // dataloader.batch_size
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (input, emotion_prediction) in enumerate(dataloader, 0):
            input = input.cuda()
            emotion_prediction = emotion_prediction.cuda()
            pred = model(input).cuda()
            loss = loss_fn(pred, emotion_prediction)
            test_loss += loss.item()
            correct += (pred.argmax(axis=1) == emotion_prediction.argmax(axis=1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    save_file.write(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


def run(model, loss_fn=nn.CrossEntropyLoss()):
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    train_loader, validation_loader, test_loader = DataManager(config=config).get_train_eval_test_dataloaders()
    save_file = open(os.path.join(config['save_file_path'], str(date.today()) + '.txt'), 'a')
    save_file.write('\n\nRunning new training session...\nLogs from {date.today()}\n\n')

    for t in range(config['train_epochs']):
        print(f"Epoch {t + 1}\n-------------------------------")
        save_file.write(f"Epoch {t + 1}\n-------------------------------\n")
        train_loop(train_loader, model, loss_fn, optimizer, save_file)
        accuracy, loss = test_loop(test_loader, model, loss_fn, save_file)
        epoch_loss_data.append(loss)

    print(f"Loss data: {sum(epoch_loss_data) / len(epoch_loss_data)}")
    save_file.write(f"Loss data: {sum(epoch_loss_data) / len(epoch_loss_data)}\n")
    print("Done!")
    save_file.write("Done!\n")
    save_file.close()
