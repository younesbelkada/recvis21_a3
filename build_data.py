import argparse
import torch
import cv2
import os, errno

import torchvision.transforms as transforms
from torchvision import datasets
from glob import glob

parser = argparse.ArgumentParser(description='RecVis A3 data augmentation script')
parser.add_argument('--path_data', type=str, default='../bird_dataset/train_images', metavar='D', help="Path to the dataset")
parser.add_argument('--path_out', type=str, default='../bird_dataset_yolo/train_images', metavar='D', help="Path to the output dataset")
args = parser.parse_args()

path_dataset = args.path_data
path_out = args.path_out

def create_dir(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

dataset = glob(path_dataset+'/**/*.jpg', recursive=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

for img in dataset:
    results = model([img], size=640)
    if ('bird' in results.pandas().xyxy[0].name.values):
        x1, y1, x2, y2 = results.pandas().xyxy[0].xmin.values[0], results.pandas().xyxy[0].ymin.values[0], results.pandas().xyxy[0].xmax.values[0], results.pandas().xyxy[0].ymax.values[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_bird = cv2.imread(img)[y1:y2, x1:x2]
        path_out_image = os.path.join(path_out, os.path.dirname(img).split('/')[-1], os.path.basename(img))
        create_dir(os.path.dirname(path_out_image))
        cv2.imwrite(path_out_image, cropped_bird)