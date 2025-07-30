#!/bin/bash

# Script to run all configs assigned to GPU 4
# Based on the device assignment logic: device_id = i % 8
# GPU 4 gets configs: 4, 12 (clsdrop_4, clsdrop_4_5)

echo "Starting GPU 4 jobs at $(date)"

# Run the configs assigned to GPU 4
# Config 4: clsdrop_4
echo "Running config: cifar20_clsdrop_4.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_4.yaml

# Config 12: clsdrop_4_5
echo "Running config: cifar20_clsdrop_4_5.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_4_5.yaml

echo "GPU 4 jobs completed at $(date)" 