import os
import torch
import wandb
import itertools

import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from torch.autograd import Variable
import torchvision.transforms as transforms

from utils.gan import VGGPerceptualLoss



class Trainer():
    def __init__(self, model, train_loader, val_loader, optimizer, epochs, use_cuda, log_intervals, path_out, class_names=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.use_cuda = use_cuda
        self.epochs = epochs
        self.log_intervals = log_intervals
        self.path_out = path_out
        self.class_names = class_names

        

    def train(self, epoch):
        wandb.init(project="birds-classification", entity="younesbelkada", 
            config={
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "epochs": self.epochs,
            "batch_size": self.train_loader.batch_size,
            "optimizer_name": self.optimizer.__class__.__name__
        })


        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            if self.model.__class__.__name__ == 'BirdsGAN':
                output, generated_im = self.model(target.long())
            else:
                output = self.model(data)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(output, target)
            if self.model.__class__.__name__ == 'BirdsGAN':
                perc_loss = VGGPerceptualLoss()(data, generated_im)
                loss += perc_loss
            loss.backward()
            wandb.log({"loss": loss.item()})
            self.optimizer.step()
            if batch_idx % self.log_intervals == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.data.item()))
    def validation(self):
        self.model.eval()
        correct = 0
        validation_loss = 0
        predicted_labels = []
        gt_labels = []
        with torch.no_grad():
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

                predicted_labels.extend(pred.detach().cpu().numpy().tolist())
                gt_labels.extend(target.detach().cpu().numpy().tolist())

            validation_loss /= len(self.val_loader.dataset)
            val_acc = 100. * correct / len(self.val_loader.dataset)
        f1_table = f1_score(gt_labels, predicted_labels, average=None)
        wandb.log({"val_loss": validation_loss, "val_acc":val_acc})
        predicted_labels = list(itertools.chain.from_iterable(predicted_labels))
        cm = wandb.plot.confusion_matrix(
            probs=None,
            y_true = gt_labels,
            preds = predicted_labels,
            class_names = self.class_names
        )
        wandb.log({"conf_mat":cm})
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            validation_loss, correct, len(self.val_loader.dataset),
            val_acc))
        print("f1 score {}".format(f1_table))
        return validation_loss
    
    def validation_gan(self):
        self.model.eval()
        correct = 0
        validation_loss = 0
        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])
        with torch.no_grad():
            for data, target in self.val_loader:
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()
                    MEAN, STD = MEAN.cuda(), STD.cuda()
                output = self.model.generate(target)
                # sum up batch loss
                #criterion = torch.nn.CrossEntropyLoss(reduction='mean')
                #validation_loss += criterion(output, target).data.item()
                # get the index of the max log-probability
                #pred = output.data.max(1, keepdim=True)[1]
                #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                generated_images = ()
                labels = []
                for i in range(4):
                    #print(output.shape)
                    generated_images = (*generated_images, transforms.ToPILImage()(output[i, :, :, :] * STD[:, None, None] + MEAN[:, None, None]))
                    labels.append(target[i].item())
                break
            #wandb.log({"gen image": wandb.Image(list(generated_images), caption=labels)}) 
            wandb.log({"gen image": [wandb.Image(list(generated_images)[i], caption=labels[i]) for i in range(4)]}) 


    def run(self, wandb=None):
        #val_acc = 0
        val_loss = 10000
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            model_file = os.path.join(self.path_out, self.model.__class__.__name__ + '.pth')
            if self.model.__class__.__name__ != 'BirdsGAN':
                new_val_loss = self.validation()
                if new_val_loss <= val_loss:
                    val_loss = new_val_loss
                    torch.save(self.model.state_dict(), model_file)
                    print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
            else:
                self.validation_gan()