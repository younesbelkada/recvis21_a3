import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pytorch_pretrained_vit import ViT

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

class ViT_(nn.Module):
    def __init__(self):
        super(ViT_, self).__init__()
        self.model_name = 'B_16_imagenet1k'
        self.model = ViT(self.model_name, pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(768, nclasses)
        )
    def forward(self, x):
        return self.model(x)


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
        n_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
          nn.Linear(n_feat, nclasses)
        )
        
    def forward(self, x):
        return self.backbone(x)
  
class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        #for param in self.backbone.parameters():
        #    param.requires_grad = False
        n_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
          nn.Linear(n_feat, nclasses)
        )
        
    def forward(self, x):
        return self.backbone(x)

class EfficientNetB7(nn.Module):
    def __init__(self):
        super(EfficientNetB7, self).__init__()
        self.backbone = models.efficientnetb7(pretrained=True)
        #for param in self.backbone.parameters():
        #    param.requires_grad = False
        n_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
          nn.Linear(n_feat, nclasses)
        )
        
    def forward(self, x):
        return self.backbone(x)


class BirdNet(nn.Module):
    def __init__(self, path_resnet, path_vit):
        super(BirdNet, self).__init__()
        print(path_resnet, path_vit)
        self.resnet50 = Resnet50()
        self.resnet50.load_state_dict(torch.load(path_resnet))
        

        #for p in self.resnet50.backbone.parameters():
        #    p.requires_grad = False

        self.vit = ViT_()
        #print(self.vit)
        self.vit.load_state_dict(torch.load(path_vit))



        for p in self.vit.model.transformer.parameters():
            p.requires_grad = False

        self.softmax = nn.Softmax(dim=-1)
        self.predictor = nn.Sequential(
            nn.Linear(40, nclasses, bias=False)
        )

    def forward(self, x):
        #pred_resnet = self.softmax(self.resnet50(x))
        pred_resnet = self.resnet50(x)
        #pred_resnet = self.softmax(self.resnet50(x))
        #pred_vit = self.softmax(self.vit(x))
        pred_vit = self.vit(x)

        out_tensor = torch.cat((pred_resnet, pred_vit), dim=-1)
        return self.predictor(out_tensor)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class BirdNet2(nn.Module):
    def __init__(self):
        super(BirdNet2, self).__init__()
        
        self.vit = ViT_()
        self.vit.model.fc = Identity()
        #print(self.vit.model.transformer)
        #exit(0)

        self.context_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 7),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(5),
        )
        self.context_encoder_fc = nn.Sequential(
            nn.Linear(5*5*128, 768),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(768*2, eps=1e-06, elementwise_affine=True),
            nn.Linear(768*2, 256),
            nn.GELU(),
            nn.Linear(256, nclasses)
        )

    def forward(self, x):
        pred_vit = self.vit.model(x)
        pred_encoder = torch.flatten(self.context_encoder(x), start_dim=1)
        pred_encoder = self.context_encoder_fc(pred_encoder)
        out_tensor = torch.cat((pred_encoder, pred_vit), dim=-1)
        return self.fc(out_tensor)