#!/bin/bash

./deep-sat/dataset/setup/dld_kaggle.sh cici118/swimming-pool-detection-in-satellite-images

printf "2" > data/swimmingPool/training/list_label.txt
python3 deep-sat/dataset/setup/to_coco_format.py -p data/swimmingPool/training/labels/ -l data/swimmingPool/training/list_label.txt

printf "pool" > data/swimmingPool/testing/list_label.txt
python3 deep-sat/dataset/setup/to_coco_format.py -p data/swimmingPool/testing/labels/ -l data/swimmingPool/training/list_label.txt