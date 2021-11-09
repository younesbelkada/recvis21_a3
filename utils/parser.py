import os
import torch
import torch.nn as nn

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from utils.data import data_transforms
from utils.model import Net
from utils.trainer import Trainer

class Parser():
    def __init__(self, config):
        self.config = config
        self.use_cuda = torch.cuda.is_available()
        torch.manual_seed(self.config['Training']['seed'])

        # Create experiment folder
        if not os.path.isdir(self.config['Training']['experiment']):
            os.makedirs(self.config['Training']['experiment'])
    def get_model(self):
        model_name = self.config['Model']['name']
        if model_name == 'Baseline':
            model = Net()
        else:
            model = None

        if self.use_cuda:
            model = model.cuda()
        return model
    def parse(self):
        self.train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(self.config['Dataset']['path_data'],'train_images'), transform=data_transforms), batch_size=int(self.config['Training']['batch_size']), shuffle=True, num_workers=1)
        self.val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(self.config['Dataset']['path_data'],'val_images'), transform=data_transforms), batch_size=int(self.config['Training']['batch_size']), shuffle=False, num_workers=1)
        self.model = self.get_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=float(self.config['Training']['learning_rate']), momentum=float(self.config['Training']['momentum']))
    
    def run(self):
        trainer = Trainer(self.model, self.train_loader, self.val_loader, self.optimizer, int(self.config['Training']['epochs']), self.use_cuda, int(self.config['Training']['log_intervals']))
        trainer.run()