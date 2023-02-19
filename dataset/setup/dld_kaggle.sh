#!/bin/bash

start=`date +%s`

source $(dirname $0)/login_kaggle.sh
zipname="$(basename $1).zip"
kaggle datasets download $1
unzip -q "$zipname" -d data
rm "$zipname"

end=`date +%s`
echo setup time: $((end-start))s