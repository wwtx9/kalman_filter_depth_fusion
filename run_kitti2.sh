#!/bin/bash

# Data paths
DATA_PATH=/home/wangweihan/Documents/my_project/cvpr2023/dataset/Scene06
OUTPUT_PATH=/home/wangweihan/Documents/my_project/cvpr2023/output

# seq_name=("15-deg-left" "15-deg-right" "30-deg-left" "30-deg-right" "clone" "fog" "morning" "overcast" "rain" "sunset")
seq_name=("15-deg-right")

for seq_i in "${seq_name[@]}"
do
   echo "Start to run sequence ${seq_i}......"
   mkdir "${OUTPUT_PATH}/deep_uncer/${seq_i}"
   mkdir "${OUTPUT_PATH}/deep_uncer/${seq_i}/statistics"
   python depth_fusion_acc.py -r ${DATA_PATH}/depth/${seq_i}/ -p ${DATA_PATH}/cam/${seq_i}/extrinsic.txt -k ${DATA_PATH}/cam/${seq_i}/intrinsic.txt -u ${DATA_PATH}/uncert/${seq_i}/ -m ${DATA_PATH}/mask/${seq_i}/ -b ${DATA_PATH}/cam/${seq_i}/bbox.txt -t ${DATA_PATH}/gt_depth/${seq_i}/ -o ${OUTPUT_PATH}/deep_uncer/${seq_i}/
done
