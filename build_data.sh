#!/bin/sh
PATH_DOWNLOAD="."
PATH_OUTPUT_DATA="./bird_dataset_yolo"
PATH_BIRD_DATASET="./bird_dataset"

wget https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip
unzip bird_dataset.zip

python build_data.py --path_data "${PATH_BIRD_DATASET}/train_images" --path_out "${PATH_OUTPUT_DATA}/train_images"
python build_data.py --path_data "${PATH_BIRD_DATASET}/val_images" --path_out "${PATH_OUTPUT_DATA}/val_images"
python build_data.py --path_data "${PATH_BIRD_DATASET}/test_images" --path_out "${PATH_OUTPUT_DATA}/test_images"