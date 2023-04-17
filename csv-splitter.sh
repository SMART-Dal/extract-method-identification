#!/bin/bash
 
cwd=/home/ip1102/projects/def-tusharma/ip1102/Ref-Res
cd $cwd
filename=$cwd/input.csv

split -d -a 4 --additional-suffix=.csv -l 10000 $filename $cwd/data/file_