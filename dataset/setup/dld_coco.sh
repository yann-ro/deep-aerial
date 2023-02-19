#!/bin/bash

start=`date +%s`

wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip -q val2017.zip -d data
unzip -q test2017.zip -d data
unzip -q train2017.zip -d data
unzip -q annotations_trainval2017.zip -d data

rm val2017.zip
rm test2017.zip
rm train2017.zip
rm annotations_trainval2017.zip

end=`date +%s`
echo setup time: $((end-start))s