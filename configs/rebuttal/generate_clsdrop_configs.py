#!/usr/bin/env python3
"""
Script to generate config files for different class dropout scenarios.
Rotates devices from 0 to 7 across different configs.
"""

import os

# Define all the class dropout scenarios
dropout_scenarios = [
    # Single class drops
    ([0], "clsdrop_0"),
    ([1], "clsdrop_1"),
    ([2], "clsdrop_2"),
    ([3], "clsdrop_3"),
    ([4], "clsdrop_4"),
    ([5], "clsdrop_5"),
    ([6], "clsdrop_6"),
    ([7], "clsdrop_7"),
    ([8], "clsdrop_8"),
    ([9], "clsdrop_9"),
    
    # Two class drops
    ([0, 1], "clsdrop_0_1"),
    ([2, 3], "clsdrop_2_3"),
    ([4, 5], "clsdrop_4_5"),
    ([6, 7], "clsdrop_6_7"),
    ([8, 9], "clsdrop_8_9"),
    
    # Five class drops
    ([0, 1, 2, 3, 4], "clsdrop_0_1_2_3_4"),
    ([5, 6, 7, 8, 9], "clsdrop_5_6_7_8_9"),
    
    # Ten class drops
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "clsdrop_0_1_2_3_4_5_6_7_8_9"),
    ([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "clsdrop_10_11_12_13_14_15_16_17_18_19")
]

def format_list_for_yaml(lst):
    """Format a list for YAML with brackets."""
    if len(lst) == 1:
        return f"[{lst[0]}]"
    else:
        return f"[{', '.join(map(str, lst))}]"

def generate_config_content(drop_classes, log_suffix, device_id):
    """Generate the YAML content manually to ensure correct format."""
    drop_classes_str = format_list_for_yaml(drop_classes)
    
    content = f"""run: # Configurations for a specific run
  random_seed: 12345 # Integer number of specifying random seed
  log_dir: logs/demo_cifar20_{log_suffix} # String for indicating where to save all the information, including models and computed signals. We can reuse the models saved in the same log_dir.
  time_log: True # Indicate whether to log the time for each step
  num_experiments: 1 # Number of total experiments

audit: # Configurations for auditing
  privacy_game: privacy_loss_model # Indicate the privacy game from privacy_loss_model
  algorithm: RMIA # String for indicating the membership inference attack. We currently support the RMIA introduced by Zarifzadeh et al. 2024(https://openreview.net/pdf?id=sT7UJh5CTc)) and the LOSS attacks
  num_ref_models: 1 # Number of reference models used to audit each target model
  device: cuda:{device_id} # String for indicating the device we want to use for inferring signals and auditing models
  report_log: report_rmia # String that indicates the folder where we save the log and auditing report
  batch_size: 5000 # Integer number for indicating batch size for evaluating models and inferring signals.
  data_size: 10000 # Integer number for indicating the size of the dataset in auditing. If not specified, the entire dataset is used.

train: # Configuration for training
  model_name: wrn28-2 # String for indicating the model type. We support CNN, wrn28-1, wrn28-2, wrn28-10, vgg16, mlp and speedyresnet (requires cuda). More model types can be added in model.py.
  device: [cuda:{device_id}] # String for indicating the device we want to use for training models.
  batch_size: 32
  optimizer: SGD
  learning_rate: 0.1
  weight_decay: 0.0005
  epochs: 75
  # scheduler: step
  # scheduler_step_fraction: 0.3
  # scheduler_step_gamma: 0.2
  # momentum: 0.9
  # differential_weight_decay: True

data: # Configuration for data
  dataset: cifar20 # String indicates the name of the dataset. We support cifar10, cifar100, purchase100 and texas100 by default.
  data_dir: data
  drop_classes: {drop_classes_str}
"""
    return content

def main():
    """Generate all config files."""
    # Create the rebuttal directory if it doesn't exist
    os.makedirs('configs/rebuttal', exist_ok=True)
    
    # Generate configs for each scenario
    for i, (drop_classes, log_suffix) in enumerate(dropout_scenarios):
        # Rotate device from 0 to 7
        device_id = i % 8
        
        content = generate_config_content(drop_classes, log_suffix, device_id)
        
        # Create filename
        filename = f'configs/rebuttal/cifar20_{log_suffix}.yaml'
        
        # Write config to file
        with open(filename, 'w') as f:
            f.write(content)
        
        print(f"Generated {filename} with device cuda:{device_id} and drop_classes {drop_classes}")

if __name__ == "__main__":
    main() 