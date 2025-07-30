#!/bin/bash

# Script to run all configs assigned to GPU 0
# Based on the device assignment logic: device_id = i % 8
# GPU 0 gets configs: 0, 8, 16 (clsdrop_0, clsdrop_8, clsdrop_5_6_7_8_9)

echo "Starting GPU 0 jobs at $(date)"

# Run the configs assigned to GPU 0
# Config 0: clsdrop_0
echo "Running config: cifar20_clsdrop_0.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_0.yaml

# Config 8: clsdrop_8  
echo "Running config: cifar20_clsdrop_8.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_8.yaml

# Config 16: clsdrop_5_6_7_8_9
echo "Running config: cifar20_clsdrop_5_6_7_8_9.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_5_6_7_8_9.yaml

echo "GPU 0 jobs completed at $(date)" 