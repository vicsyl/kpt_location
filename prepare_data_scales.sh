#!/bin/bash

seq=`seq 1 1 9`

# method="log_stats"
method="log_min_distance"

#echo "scales:"
#for i in $seq; do
#  echo 0.$i
#done

conda activate kpt_loc

for i in $seq; do
  cmd="/Users/vaclav/miniconda3/envs/kpt_loc/bin/python ./patch_dataset.py --scale 0.${i} --method ${method}"
  echo "cmd: ${cmd}"
  `${cmd}`
done
