#!/bin/bash

# Script to run all configs assigned to GPU 5
# Based on the device assignment logic: device_id = i % 8
# GPU 5 gets configs: 5, 13 (clsdrop_5, clsdrop_6_7)

echo "Starting GPU 5 jobs at $(date)"

# Run the configs assigned to GPU 5
# Config 5: clsdrop_5
echo "Running config: cifar20_clsdrop_5.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_5.yaml

# Config 13: clsdrop_6_7
echo "Running config: cifar20_clsdrop_6_7.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_6_7.yaml

echo "GPU 5 jobs completed at $(date)" 