import argparse
import os
import configparser
import wandb
import torch
from tqdm import tqdm

from torch.autograd import Variable
from utils.model import Net
from utils.parser import Parser

parser = argparse.ArgumentParser(description='RecVis A3 training script - Belkada Younes')
parser.add_argument('--file', type=str, default='config.ini', metavar='D', help="Path to the config file")
parser.add_argument('--mode', type=str, default='train', metavar='D', help="Path to the config file")
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.file)

# earnest-mountain-236
# ViT + ResNet = 0.8670

parser = Parser(config)
parser.parse()

if args.mode == 'train':
    parser.run()
else:
    parser.run_eval()