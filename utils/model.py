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



class BirdNet(nn.Module):
    def __init__(self, path_resnet, path_vit):
        super(BirdNet, self).__init__()
        print(path_resnet, path_vit)
        self.resnet50 = Resnet50()
        self.resnet50.load_state_dict(torch.load(path_resnet))
    

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

class TransforBirds(nn.Module):
    def __init__(self):
        super(TransforBirds, self).__init__()
        
        self.vit = ViT_()
        self.vit.model.fc = Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 11, dilation=7),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.down_sample1 = nn.AdaptiveMaxPool2d(53)
        self.down_sample2 = nn.AdaptiveMaxPool2d(10)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 11, dilation=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 11, dilation=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(10),
        )


        self.context_encoder_fc = nn.Sequential(
            nn.Linear(10*10*32, 768)
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(768, eps=1e-06, elementwise_affine=True),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, nclasses)
        )

    def forward(self, x):
        pred_vit = self.vit.model(x)
        #print(x.shape)
        out_encoder = self.conv1(x)
        out_encoder = self.conv2(out_encoder)
        out_encoder = out_encoder + self.down_sample1(out_encoder)
        #print(out_encoder.shape)
        out_encoder = self.conv3(out_encoder)
        out_encoder = out_encoder + self.down_sample2(out_encoder)
        #print(out_encoder.shape)
        #exit(0)
        pred_encoder = torch.flatten(out_encoder, start_dim=1)
        pred_encoder = self.context_encoder_fc(pred_encoder)
        out_tensor = pred_encoder + pred_vit
        return self.fc(out_tensor)