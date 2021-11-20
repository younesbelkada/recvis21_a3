import os
import torch
import operator
from tqdm import tqdm
import torch.nn as nn

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from utils.data import data_transforms, data_transforms_val, data_transforms_yolo, data_transforms_yolo_val, pil_loader
from utils.model import Net, VGG16_birds, Resnet34, AlexNet_birds, BirdNet, Resnet50, ViT_, EfficientNetB7, BirdNet2
from utils.vit import TransforBirds
from utils.trainer import Trainer
from utils.gan import BirdsGAN

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
        self.model_name = self.config['Model']['name']
        if self.model_name == 'Baseline':
            model = Net()
        elif self.model_name == 'Resnet34':
            model = Resnet34()
        elif self.model_name == 'Resnet50':
            model = Resnet50()
        elif self.model_name == 'Alexnet':
            model = AlexNet_birds()
        elif self.model_name == 'BirdNet':
            model = BirdNet()
        elif self.model_name == 'TransforBirds':
            model = TransforBirds(256, 9, 'conv', self.use_cuda)
        elif self.model_name == 'BirdsGAN':
            model = BirdsGAN(2048, './models/Resnet50/Resnet50.pth', self.use_cuda)
        elif self.model_name == 'EfficientNetB7':
            model = EfficientNetB7()
        elif self.model_name == 'ViT':
            model = ViT_()
        elif self.model_name == 'BirdNet2':
            model = BirdNet2()
            
            #exit(0)

        if self.use_cuda:
            model = model.cuda()
        return model
    def parse(self):

        self.model = self.get_model()
        self.learning_rate = float(self.config['Training']['learning_rate'])
        self.epochs = int(self.config['Training']['epochs'])
        self.batch_size = int(self.config['Training']['batch_size'])
        self.optimizer_name = self.config['Training']['optimizer_name']
        self.augment = self.config['Dataset'].getboolean('augment')
        if self.augment:
            self.train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([datasets.ImageFolder(os.path.join(self.config['Dataset']['path_data'],'train_images'), transform=data_transforms), datasets.ImageFolder(os.path.join(self.config['Dataset']['path_data']+'_yolo','train_images'), transform=data_transforms_yolo)]), batch_size=self.batch_size, shuffle=True, num_workers=1)
            self.val_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([datasets.ImageFolder(os.path.join(self.config['Dataset']['path_data'],'val_images'), transform=data_transforms_val), datasets.ImageFolder(os.path.join(self.config['Dataset']['path_data']+'_yolo','val_images'), transform=data_transforms_yolo_val)]), batch_size=self.batch_size, shuffle=True, num_workers=1)
        else:
            self.train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(self.config['Dataset']['path_data'],'train_images'), transform=data_transforms), batch_size=self.batch_size, shuffle=True, num_workers=1)
            self.val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(self.config['Dataset']['path_data'],'val_images'), transform=data_transforms_val), batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.class_names = datasets.ImageFolder(os.path.join(self.config['Dataset']['path_data'],'train_images')).class_to_idx
        self.class_names = sorted(self.class_names.items(), key=lambda kv: kv[1])
        self.class_names = [item[0] for item in self.class_names]
    def run(self):
        if self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=float(self.config['Training']['learning_rate']), momentum=float(self.config['Training']['momentum']))
        else:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        trainer = Trainer(self.model, self.train_loader, self.val_loader, self.optimizer, self.epochs, self.use_cuda, int(self.config['Training']['log_intervals']), self.path_out, self.class_names)
        trainer.run()
    
    def run_eval(self):
        path_model = os.path.join(self.path_out, self.model.__class__.__name__ + '.pth')
        self.model.load_state_dict(torch.load(path_model))
        self.model.eval()

        output_file = self.config['Evaluation']['output_file']
        test_dir = os.path.join(self.config['Dataset']['path_data'], 'test_images', 'mistery_category')

        output_file = open(output_file, "w")
        output_file.write("Id,Category\n")
        with torch.no_grad():
            for f in tqdm(os.listdir(test_dir)):
                if 'jpg' in f:
                    data = data_transforms_val(pil_loader(test_dir + '/' + f))
                    data = data.view(1, data.size(0), data.size(1), data.size(2))
                    if self.use_cuda:
                        data = data.cuda()
                    output = self.model(data)
                    pred = output.data.max(1, keepdim=True)[1]
                    output_file.write("%s,%d\n" % (f[:-4], pred))

        output_file.close()

        print("Succesfully wrote " + self.config['Evaluation']['output_file'] + ', you can upload this file to the kaggle competition website')