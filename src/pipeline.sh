#!/bin/bash

echo "Starting soybean nodule pipeline..."

# Start total timer
TOTAL_START=$(date +%s)

python src/pipeline_predict.py --img_folder ./Root_nodules_images --pred_folder ./pred_Root_nodules_images_v01 # nms and conf are 0.2
# python src/pipeline_predict.py --img_folder ./image_label_52 --pred_folder ./pred_image_label_52_v01 # nms and conf are 0.2

# Calculate total time
TOTAL_TIME=$(( $(date +%s) - TOTAL_START ))
TOTAL_MINUTES=$(( TOTAL_TIME / 60 ))
TOTAL_SECONDS=$(( TOTAL_TIME % 60 ))

echo "Total execution time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Finished script"