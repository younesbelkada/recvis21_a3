import argparse
import os
import configparser
import wandb
import torch
from tqdm import tqdm

from torch.autograd import Variable
from utils.model import Net
from utils.parser import Parser

wandb.init(project="birds-classification", entity="younesbelkada")
parser = argparse.ArgumentParser(description='RecVis A3 training script - Belkada Younes')
parser.add_argument('--file', type=str, default='config.ini', metavar='D', help="Path to the config file")
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.file)

parser = Parser(config)
parser.parse()

parser.run()