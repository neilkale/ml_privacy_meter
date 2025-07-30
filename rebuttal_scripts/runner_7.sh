#!/bin/bash

# Script to run all configs assigned to GPU 7
# Based on the device assignment logic: device_id = i % 8
# GPU 7 gets configs: 7, 15 (clsdrop_7, clsdrop_0_1_2_3_4)

echo "Starting GPU 7 jobs at $(date)"

# Run the configs assigned to GPU 7
# Config 7: clsdrop_7
echo "Running config: cifar20_clsdrop_7.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_7.yaml

# Config 15: clsdrop_0_1_2_3_4
echo "Running config: cifar20_clsdrop_0_1_2_3_4.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_0_1_2_3_4.yaml

echo "GPU 7 jobs completed at $(date)" 