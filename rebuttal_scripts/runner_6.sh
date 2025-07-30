#!/bin/bash

# Script to run all configs assigned to GPU 6
# Based on the device assignment logic: device_id = i % 8
# GPU 6 gets configs: 6, 14 (clsdrop_6, clsdrop_8_9)

echo "Starting GPU 6 jobs at $(date)"

# Run the configs assigned to GPU 6
# Config 6: clsdrop_6
echo "Running config: cifar20_clsdrop_6.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_6.yaml

# Config 14: clsdrop_8_9
echo "Running config: cifar20_clsdrop_8_9.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_8_9.yaml

echo "GPU 6 jobs completed at $(date)" 