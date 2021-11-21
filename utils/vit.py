import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as transforms
import torchvision.models as models
from torch.nn import LayerNorm
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
