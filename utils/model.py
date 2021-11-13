import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

nclasses = 20 

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

