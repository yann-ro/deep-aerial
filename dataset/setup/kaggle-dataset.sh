#!/bin/bash

start=`date +%s`

source $(dirname $0)/login_kaggle.sh
zipname = $(basename $1)
kaggle datasets download $1
unzip -q $(zipname).zip -d data
rm $(zipname).zip

end=`date +%s`
echo setup time: $((end-start))s