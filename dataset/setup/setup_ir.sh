#!/bin/bash

./deep-aerial/dataset/setup/dld_kaggle.sh pandrii000/hituav-a-highaltitude-infrared-thermal-dataset

mkdir data/hit-uav/train
mkdir data/hit-uav/test
mkdir data/hit-uav/val
mv data/hit-uav/images/train data/hit-uav/train/images
mv data/hit-uav/images/test data/hit-uav/test/images
mv data/hit-uav/images/val data/hit-uav/val/images
rmv data/hit-uav/images/

printf "0\n1\n2\n3\n4" > data/swimmingPool/training/list_label.txt
python3 deep-aerial/dataset/setup/to_coco_format.py -p data/swimmingPool/training/labels -l data/swimmingPool/training/list_label.txt

printf "0\n1\n2\n3\n4" > data/swimmingPool/testing/list_label.txt
python3 deep-aerial/dataset/setup/to_coco_format.py -p data/swimmingPool/testing/labels -l data/swimmingPool/testing/list_label.txt