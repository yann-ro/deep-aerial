#!/bin/bash

start=`date +%s`

source $(dirname $0)/login_kaggle.sh
kaggle datasets download usharengaraju/isaid-dataset
unzip -q isaid-dataset.zip -d data
rm isaid-dataset.zip

end=`date +%s`
echo setup time: $((end-start))s