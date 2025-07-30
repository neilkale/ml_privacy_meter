#!/bin/bash

# Script to run all configs assigned to GPU 1
# Based on the device assignment logic: device_id = i % 8
# GPU 1 gets configs: 1, 9, 17 (clsdrop_1, clsdrop_9, clsdrop_0_1_2_3_4_5_6_7_8_9)

echo "Starting GPU 1 jobs at $(date)"

# Run the configs assigned to GPU 1
# Config 1: clsdrop_1
echo "Running config: cifar20_clsdrop_1.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_1.yaml

# Config 9: clsdrop_9
echo "Running config: cifar20_clsdrop_9.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_9.yaml

# Config 17: clsdrop_0_1_2_3_4_5_6_7_8_9
echo "Running config: cifar20_clsdrop_0_1_2_3_4_5_6_7_8_9.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_0_1_2_3_4_5_6_7_8_9.yaml

echo "GPU 1 jobs completed at $(date)"