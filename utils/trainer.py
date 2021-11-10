import os
import torch
import torch.nn as nn
from torch.autograd import Variable

class Trainer():
    def __init__(self, model, train_loader, val_loader, optimizer, epochs, use_cuda, log_intervals, path_out):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.use_cuda = use_cuda
        self.epochs = epochs
        self.log_intervals = log_intervals
        self.path_out = path_out

    def train(self, epoch):
        val_loss = 1000

        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_intervals == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.data.item()))
    def validation(self):
        self.model.eval()
        correct = 0
        validation_loss = 0
        for data, target in self.val_loader:
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(self.val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            validation_loss, correct, len(self.val_loader.dataset),
            100. * correct / len(self.val_loader.dataset)))
        return validation_loss


    def run(self):
        val_loss = 1000
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            new_val_loss = self.validation()
            model_file = os.path.join(self.path_out, self.model.__class__.__name__ + '.pth')
            if new_val_loss < val_loss:
                val_loss = new_val_loss
                torch.save(self.model.state_dict(), model_file)
                print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
