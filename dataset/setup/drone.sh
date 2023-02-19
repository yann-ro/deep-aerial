#!/bin/bash

source $(dirname $0)/login_kaggle.sh
kaggle datasets download bulentsiyah/semantic-drone-dataset
unzip semantic-drone-dataset.zip -d data
rm semantic-drone-dataset.zip