#!/bin/bash
 
cwd_folder=/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research
cd $cwd_folder
filename=$cwd_folder/input.csv

# split -d -a 4 --additional-suffix=.csv -l 10000 $filename $cwd_folder/data/file_
split -d -a 4 --additional-suffix=.csv -l 250 $filename $cwd_folder/data/file_
