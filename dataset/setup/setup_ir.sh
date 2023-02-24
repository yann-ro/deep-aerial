#!/bin/bash

./deep-aerial/dataset/setup/dld_kaggle.sh pandrii000/hituav-a-highaltitude-infrared-thermal-dataset

mkdir data/hit-uav/train
mkdir data/hit-uav/test
mkdir data/hit-uav/val
mv data/hit-uav/images/train data/hit-uav/train/images
mv data/hit-uav/images/test data/hit-uav/test/images
mv data/hit-uav/images/val data/hit-uav/val/images
mv data/hit-uav/labels/train data/hit-uav/train/labels
mv data/hit-uav/labels/test data/hit-uav/test/labels
mv data/hit-uav/labels/val data/hit-uav/val/labels
rm -rf data/hit-uav/images/
rm -rf data/hit-uav/labels/

printf "0\n1\n2\n3\n4" > data/hit-uav/train/list_label.txt
python3 deep-aerial/dataset/setup/to_coco_format.py -p data/hit-uav/train/labels -l data/hit-uav/train/list_label.txt -d txt

printf "0\n1\n2\n3\n4" > data/hit-uav/test/list_label.txt
python3 deep-aerial/dataset/setup/to_coco_format.py -p data/hit-uav/test/labels -l data/hit-uav/test/list_label.txt -d txt

printf "0\n1\n2\n3\n4" > data/hit-uav/val/list_label.txt
python3 deep-aerial/dataset/setup/to_coco_format.py -p data/hit-uav/val/labels -l data/hit-uav/val/list_label.txt -d txt