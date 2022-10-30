import os
from datetime import date

import torch

torch.cuda.empty_cache()


class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, criterion, optimizer, loss_fn, config):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.best_metric = 0.0
        self.loss_fn = loss_fn
        self.log_file = open(os.path.join(config['save_file_path'], str(date.today()) + '.txt'), 'a')
        self.epoch_loss_data = []

    def train_loop(self):
        size = len(self.train_dataloader.dataset)

        if self.config['resume_training'] is True:
            checkpoint = torch.load(
                os.path.join(self.config['exp_path'], self.config['exp_name'], 'latest_checkpoint.pkl'),
                map_location=self.config['device'])
            self.model.load_state_dict(checkpoint['model_weights'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        for batch, (input, emotion_prediction) in enumerate(self.train_dataloader, 0):
            # Compute prediction and loss
            input = input.cuda()
            emotion_prediction = emotion_prediction.cuda()
            pred = self.model(input).cuda()
            loss = self.loss_fn(pred, emotion_prediction)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if batch % 2 == 0:
            #     loss, current = loss.item(), batch * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            x_length = len(input)
            loss, current = loss.item(), batch * x_length
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            self.log_file.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")

    def test_loop(self):
        size = len(self.eval_dataloader.dataset)
        num_batches = len(self.eval_dataloader.dataset) // self.eval_dataloader.batch_size
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch, (input, emotion_prediction) in enumerate(self.eval_dataloader, 0):
                input = input.cuda()
                emotion_prediction = emotion_prediction.cuda()
                pred = self.model(input).cuda()
                loss = self.loss_fn(pred, emotion_prediction)
                test_loss += loss.item()
                correct += (pred.argmax(axis=1) == emotion_prediction.argmax(axis=1)).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        self.log_file.write(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return correct, test_loss

    def save_net_state(self, epoch, latest=False, best=False):
        if latest is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'latest_checkpoint.pkl')
            to_save = {
                'epoch': epoch,
                'model_weights': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(to_save, path_to_save)
        elif best is True:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'best_model.pkl')
            to_save = {
                'epoch': epoch,
                'stats': self.best_metric,
                'model_weights': self.model.state_dict()
            }
            torch.save(to_save, path_to_save)
        else:
            path_to_save = os.path.join(self.config['exp_path'], self.config['exp_name'], f'model_epoch_{epoch}.pkl')
            torch.save(self.model, path_to_save)

    def run(self):
        self.log_file.write(f'\n\nRunning new training session...\nLogs from {date.today()}\n\n')

        for t in range(self.config['train_epochs']):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.log_file.write(f"Epoch {t + 1}\n-------------------------------\n")
            self.train_loop()
            self.save_net_state(epoch=t + 1, latest=True)
            accuracy, loss = self.test_loop()
            self.epoch_loss_data.append(loss)

        print(f"Loss data: {sum(self.epoch_loss_data) / len(self.epoch_loss_data)}")
        self.log_file.write(f"Loss data: {sum(self.epoch_loss_data) / len(self.epoch_loss_data)}\n")
        print("Done!")
        self.log_file.write("Done!\n")
        self.log_file.close()
