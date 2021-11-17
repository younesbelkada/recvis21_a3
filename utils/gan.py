import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn import LayerNorm
from PIL import Image

from utils.model import Resnet34

torch.manual_seed(0)
nclasses = 20

class BirdsGenerator(nn.Module):
    def __init__(self, emb_size, use_cuda=None):
        super(BirdsGenerator, self).__init__()
        self.embedding = nn.Embedding(nclasses, emb_size)
        self.use_cuda = use_cuda
        ngf = 64
        nc = 3
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 128, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            #nn.Upsample(2),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # state size. (ngf*4) x 8 x 8
            #nn.Upsample(2),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            #nn.Upsample(2),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )
    def forward(self, x, seed=random.randint(0,5000)):
        #print(seed)
        torch.manual_seed(seed)
        #x = torch.randint(0, nclasses, (batch_size,1))
        #x = torch.LongTensor(labels)
        #if self.use_cuda:
        #    x = x.cuda()
        emb = self.embedding(x).squeeze(1)
        emb = emb.view(emb.shape[0], 128, 4, 4)
        return self.main(emb)

class BirdsGAN(nn.Module):
    def __init__(self, emb_size, path_resnet, use_cuda=None):
        super(BirdsGAN, self).__init__()
        self.gen = BirdsGenerator(emb_size, use_cuda)
        self.discriminator = Resnet34()
        self.discriminator.load_state_dict(torch.load(path_resnet))
        for p in self.discriminator.parameters():
            p.requires_grad = False
    def forward(self, x):
        generated_im = self.gen(x)
        return self.discriminator(generated_im)
    def generate(self, x):
        return self.gen(x)
