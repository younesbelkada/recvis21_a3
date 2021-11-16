import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

torch.manual_seed(0)
nclasses = 20 

class Patches(nn.Module):
    def __init__(self):
        super(Patches, self).__init__()
        self.patch_size = 56
        self.stride = 56
        self.nb_patches = (224//self.patch_size)**2
    def forward(self, x):
        #x = torch.randn(10, 128, 227, 227)
        x = x.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride).permute(0, 2, 3, 1, 4, 5)
        x = torch.reshape(x, (x.shape[0], self.nb_patches, 3, self.patch_size, self.patch_size))
        return x
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class VGG16_birds(nn.Module):
    def __init__(self):
        super(VGG16_birds, self).__init__()
        self.backbone = models.vgg16(pretrained=True)
        #for param in self.backbone.parameters():
        #    param.requires_grad = False
        self.backbone.classifier = nn.Sequential( 
            nn.Linear(4098, nclasses)
        )
    def forward(self, x):
        return self.backbone(x)

class AlexNet_birds(nn.Module):
  def __init__(self):
        super(AlexNet_birds, self).__init__()
        self.backbone = models.alexnet(pretrained=True)
        self.backbone.classifier = nn.Sequential( 
            nn.Linear(9216, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.4),
            nn.Linear(4096, nclasses)
        )
        #print(self.backbone)
  def forward(self, x):
    return self.backbone(x)

class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        self.backbone = models.resnet34(pretrained=True)
        #for param in self.backbone.parameters():
        #    param.requires_grad = False
        #self.backbone.requires_grad = True
        self.backbone.fc = nn.Sequential(
          nn.Linear(512, nclasses)
        )
        
    def forward(self, x):
        return self.backbone(x)
  
class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        #for param in self.backbone.parameters():
        #    param.requires_grad = False
        #self.backbone.requires_grad = True
        self.backbone.fc = nn.Sequential(
          nn.Linear(2048, nclasses)
        )
        
    def forward(self, x):
        return self.backbone(x)

class BirdNet(nn.Module):
    def __init__(self):
        super(BirdNet, self).__init__()
        self.patch_converter = Patches()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(3200, 1028),
            nn.BatchNorm1d(1028),
            nn.Linear(1028, 64)
        )

        self.fc_classification = nn.Sequential(
            nn.Linear(1024, nclasses)
        )
    def forward(self, x):
        patches = self.patch_converter(x)
        final_vector = ()
        for i in range(patches.shape[1]):
            feat_maps = self.encoder(patches[:, i, :, :, :])
            flattened_feat_map = torch.flatten(feat_maps, start_dim=1)
            encoded_feat_map = self.fc(flattened_feat_map)
            final_vector = (*final_vector, encoded_feat_map)
        final_vector = torch.cat(final_vector, dim=1)
        return self.fc_classification(final_vector)