#!/bin/bash

# Script to run all configs assigned to GPU 3
# Based on the device assignment logic: device_id = i % 8
# GPU 3 gets configs: 3, 11 (clsdrop_3, clsdrop_2_3)

echo "Starting GPU 3 jobs at $(date)"

# Run the configs assigned to GPU 3
# Config 3: clsdrop_3
echo "Running config: cifar20_clsdrop_3.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_3.yaml

# Config 11: clsdrop_2_3
echo "Running config: cifar20_clsdrop_2_3.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_2_3.yaml

echo "GPU 3 jobs completed at $(date)" 