#!/bin/bash

start=`date +%s`

source $(dirname $0)/login_kaggle.sh
kaggle datasets download bulentsiyah/semantic-drone-dataset
unzip -q semantic-drone-dataset.zip -d data
rm semantic-drone-dataset.zip

end=`date +%s`
echo setup time: $((end-start))s