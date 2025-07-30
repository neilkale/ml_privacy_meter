#!/bin/bash

# Script to run all configs assigned to GPU 2
# Based on the device assignment logic: device_id = i % 8
# GPU 2 gets configs: 2, 10, 18 (clsdrop_2, clsdrop_0_1, clsdrop_10_11_12_13_14_15_16_17_18_19)

echo "Starting GPU 2 jobs at $(date)"

# Run the configs assigned to GPU 2
# Config 2: clsdrop_2
echo "Running config: cifar20_clsdrop_2.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_2.yaml

# Config 10: clsdrop_0_1
echo "Running config: cifar20_clsdrop_0_1.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_0_1.yaml

# Config 18: clsdrop_10_11_12_13_14_15_16_17_18_19
echo "Running config: cifar20_clsdrop_10_11_12_13_14_15_16_17_18_19.yaml"
python run_clsdrop_mia.py --cf configs/rebuttal/cifar20_clsdrop_10_11_12_13_14_15_16_17_18_19.yaml

echo "GPU 2 jobs completed at $(date)" 