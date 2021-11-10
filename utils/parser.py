import os
import torch
from tqdm import tqdm
import torch.nn as nn

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from utils.data import data_transforms, pil_loader
from utils.model import Net, VGG16_birds, Resnet18
from utils.trainer import Trainer

class Parser():
    def __init__(self, config):
        self.config = config
        self.path_out = self.config['Model']['path_out']
        self.use_cuda = torch.cuda.is_available()
        torch.manual_seed(self.config['Training']['seed'])

        # Create experiment folder
        if not os.path.isdir(os.path.join(self.path_out, self.config['Training']['experiment'])):
            os.makedirs(os.path.join(self.path_out, self.config['Training']['experiment']))
        self.path_out = os.path.join(self.path_out, self.config['Training']['experiment'])
    def get_model(self):
        model_name = self.config['Model']['name']
        if model_name == 'Baseline':
            model = Net()
        else:
            model = Resnet18()

        if self.use_cuda:
            model = model.cuda()
        return model
    def parse(self):
        self.train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(self.config['Dataset']['path_data'],'train_images'), transform=data_transforms), batch_size=int(self.config['Training']['batch_size']), shuffle=True, num_workers=1)
        self.val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(self.config['Dataset']['path_data'],'val_images'), transform=data_transforms), batch_size=int(self.config['Training']['batch_size']), shuffle=False, num_workers=1)
        self.model = self.get_model()
    
    def run(self):
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=float(self.config['Training']['learning_rate']), momentum=float(self.config['Training']['momentum']))
        trainer = Trainer(self.model, self.train_loader, self.val_loader, self.optimizer, int(self.config['Training']['epochs']), self.use_cuda, int(self.config['Training']['log_intervals']), self.path_out)
        trainer.run()
    
    def run_eval(self):
        path_model = os.path.join(self.path_out, self.model.__class__.__name__ + '.pth')
        self.model.load_state_dict(torch.load(path_model))
        self.model.eval()

        output_file = self.config['Evaluation']['output_file']
        test_dir = os.path.join(self.config['Dataset']['path_data'], 'test_images', 'mistery_category')

        output_file = open(output_file, "w")
        output_file.write("Id,Category\n")
        for f in tqdm(os.listdir(test_dir)):
            if 'jpg' in f:
                data = data_transforms(pil_loader(test_dir + '/' + f))
                data = data.view(1, data.size(0), data.size(1), data.size(2))
                if use_cuda:
                    data = data.cuda()
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                output_file.write("%s,%d\n" % (f[:-4], pred))

        output_file.close()

        print("Succesfully wrote " + self.config['Evaluation']['output_file'] + ', you can upload this file to the kaggle competition website')