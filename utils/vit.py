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

class ConvPatchEncoder(nn.Module):
    def __init__(self, emb_size):
        super(ConvPatchEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, dilation=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=1),
            nn.GELU(),
            nn.AdaptiveMaxPool2d(5),
            nn.BatchNorm2d(128)
        )
        self.fc = nn.Sequential(
            nn.Linear(3200, emb_size)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

class LinearPatchEncoder(nn.Module):
    def __init__(self, input_size, emb_size):
        super(LinearPatchEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, emb_size, biais=False)
        )
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

class TransformersEncoder(nn.Module):
    def __init__(self, emb_size, nb_heads, nb_patches, batch_first=True):
        super(TransformersEncoder, self).__init__()
        self.MHA = nn.MultiheadAttention(emb_size, nb_heads, batch_first)
    def forward(self, x):
        output, att_weights = self.MHA(x, x, x)
        return output, att_weights

class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, hidden_size, mlp_dim, nb_heads, nb_patches):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim)
        self.attn = TransformersEncoder(hidden_size, nb_heads, nb_patches)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class TransforBirds(nn.Module):
    def __init__(self, emb_size, nb_layers, encoder_type='linear', use_cuda=None):
        super(TransforBirds, self).__init__()
        self.patches = Patches()
        nb_patches = self.patches.nb_patches
        self.patch_embedding = nn.Embedding(nb_patches+1, emb_size)
        input_size = self.patches.patch_size
        self.use_cuda = use_cuda
        if encoder_type == 'linear':
            self.encoder = LinearPatchEncoder(input_size*input_size*3, emb_size)
        else:
            self.encoder = ConvPatchEncoder(emb_size)
        self.nb_layers = nb_layers
        #self.transformers = nn.ModuleList([TransformersEncoder(emb_size, 4, nb_patches) for i in range(nb_layers)])
        self.transformers = nn.ModuleList([Block(emb_size, emb_size//2, 4, nb_patches) for i in range(nb_layers)])
        self.classifier = nn.Linear(emb_size, nclasses)

    def forward(self, x):
        patches = self.patches(x)
        first_layers_embeddings = ()
        for i in range(patches.shape[1]):
            indices = torch.LongTensor([i+1 for j in range(x.shape[0])])
            if self.use_cuda:
                indices = indices.cuda()
            pos_emb = self.patch_embedding(indices)
            patch_emb = self.encoder(patches[:, i, :, :, :])
            #final_emb = torch.cat((pos_emb, patch_emb), dim=-1)
            final_emb = pos_emb + patch_emb
            first_layers_embeddings = (*first_layers_embeddings, final_emb)
        layers_embeddings = torch.stack(first_layers_embeddings, dim=1)
        for i in range(self.nb_layers):
            #layers_embeddings = self.transformers[i](layers_embeddings)+layers_embeddings
            layers_embeddings, _ = self.transformers[i](layers_embeddings)
        return self.classifier(layers_embeddings[:, 0, :])