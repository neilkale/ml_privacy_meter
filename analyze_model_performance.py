#!/usr/bin/env python3
"""
Temporary script to analyze model performance class-by-class.
This helps verify that models are correctly trained with/without dropped classes.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml
from collections import defaultdict
import argparse
import logging

from dataset.utils import get_dataset
from models.utils import get_model
from util import load_dataset, split_dataset_for_clsdrop


def create_simple_logger():
    """Create a simple logger for the analysis."""
    logger = logging.getLogger('analyze_model')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def load_model_from_pickle(model_path, model_name, dataset_name, configs):
    """Load a model from pickle file."""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Loaded data type: {type(model_data)}")
    
    # Check if it's a state_dict or a full model
    if isinstance(model_data, dict) and ('state_dict' in model_data or all(isinstance(k, str) for k in model_data.keys())):
        # It's a state_dict, need to create model first
        print("Detected state_dict format, creating model architecture...")
        model = get_model(model_name, dataset_name, configs)
        
        if 'state_dict' in model_data:
            model.load_state_dict(model_data['state_dict'])
        else:
            model.load_state_dict(model_data)
        
        print(f"Model created and weights loaded: {type(model)}")
    else:
        # It's already a model object
        model = model_data
        print(f"Direct model object loaded: {type(model)}")
    
    model.eval()
    return model


def evaluate_model_per_class(model, dataset, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size=128):
    """Evaluate model performance per class."""
    model = model.to(device)
    model.eval()
    
    # Get all targets to determine classes
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        # For subset datasets, extract targets
        targets = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            targets.append(target)
        targets = np.array(targets)
    
    unique_classes = np.unique(targets)
    print(f"Found classes: {unique_classes}")
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Track predictions per class
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_predictions = defaultdict(list)
    class_confidences = defaultdict(list)
    
    all_predictions = []
    all_targets = []
    all_confidences = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Get model predictions
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            predicted = output.argmax(dim=1)
            confidence = probabilities.max(dim=1)[0]
            
            # Store all results
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            
            # Per-class statistics
            for i in range(len(target)):
                true_class = target[i].item()
                pred_class = predicted[i].item()
                conf = confidence[i].item()
                
                class_total[true_class] += 1
                class_predictions[true_class].append(pred_class)
                class_confidences[true_class].append(conf)
                
                if true_class == pred_class:
                    class_correct[true_class] += 1
    
    # Calculate overall accuracy
    overall_accuracy = sum(class_correct.values()) / sum(class_total.values()) if sum(class_total.values()) > 0 else 0
    
    # Calculate per-class accuracies
    class_accuracies = {}
    for class_id in unique_classes:
        if class_total[class_id] > 0:
            class_accuracies[class_id] = class_correct[class_id] / class_total[class_id]
        else:
            class_accuracies[class_id] = 0.0
    
    return {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': class_accuracies,
        'class_total': dict(class_total),
        'class_correct': dict(class_correct),
        'class_confidences': dict(class_confidences),
        'all_predictions': all_predictions,
        'all_targets': all_targets,
        'all_confidences': all_confidences
    }


def analyze_dropped_classes_effect(model_path, config_path, model_idx=0):
    """Analyze how dropped classes affect model performance."""
    
    # Create a simple logger
    logger = create_simple_logger()
    
    # Load configuration
    with open(config_path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    
    print(f"Configuration loaded from: {config_path}")
    print(f"Drop classes: {configs['data'].get('drop_classes', 'None')}")
    
    # Load dataset
    dataset, population = load_dataset(configs, configs['data']['data_dir'], logger)
    print(f"Full dataset size: {len(dataset)}")
    
    # Split dataset
    drop_classes = configs['data'].get('drop_classes', [])
    if drop_classes:
        dataset_id, dataset_ood = split_dataset_for_clsdrop(dataset, drop_classes)
        print(f"ID dataset size: {len(dataset_id)}")
        print(f"OOD dataset size: {len(dataset_ood)}")
    else:
        dataset_id = dataset
        dataset_ood = None
        print("No classes dropped - using full dataset")
    
    # Load model
    model = load_model_from_pickle(
        model_path, 
        configs['train']['model_name'],
        configs['data']['dataset'],
        configs
    )
    
    print("\n" + "="*80)
    print(f"ANALYZING MODEL {model_idx}")
    print("="*80)
    
    # Evaluate on full dataset
    print("\n--- FULL DATASET EVALUATION ---")
    full_results = evaluate_model_per_class(model, dataset)
    print(f"Overall accuracy: {full_results['overall_accuracy']:.4f}")
    print("\nPer-class accuracies:")
    for class_id, acc in sorted(full_results['class_accuracies'].items()):
        count = full_results['class_total'][class_id]
        is_dropped = class_id in drop_classes if drop_classes else False
        status = " (DROPPED)" if is_dropped else " (KEPT)"
        print(f"  Class {class_id:2d}: {acc:.4f} ({count:4d} samples){status}")
    
    # Evaluate on ID dataset
    if dataset_id:
        print("\n--- ID DATASET EVALUATION ---")
        id_results = evaluate_model_per_class(model, dataset_id)
        print(f"Overall accuracy on ID data: {id_results['overall_accuracy']:.4f}")
        print("\nPer-class accuracies (ID only):")
        for class_id, acc in sorted(id_results['class_accuracies'].items()):
            count = id_results['class_total'][class_id]
            print(f"  Class {class_id:2d}: {acc:.4f} ({count:4d} samples)")
    
    # Evaluate on OOD dataset
    if dataset_ood:
        print("\n--- OOD DATASET EVALUATION ---")
        ood_results = evaluate_model_per_class(model, dataset_ood)
        print(f"Overall accuracy on OOD data: {ood_results['overall_accuracy']:.4f}")
        print("\nPer-class accuracies (OOD only):")
        for class_id, acc in sorted(ood_results['class_accuracies'].items()):
            count = ood_results['class_total'][class_id]
            print(f"  Class {class_id:2d}: {acc:.4f} ({count:4d} samples)")
        
        # Analyze confidence on OOD data
        print("\nOOD Confidence Analysis:")
        avg_confidences = {}
        for class_id, confidences in ood_results['class_confidences'].items():
            avg_confidences[class_id] = np.mean(confidences)
        
        for class_id, avg_conf in sorted(avg_confidences.items()):
            print(f"  Class {class_id:2d}: avg confidence = {avg_conf:.4f}")
    
    # Check if model predicts dropped classes
    if drop_classes and 'all_predictions' in full_results:
        print(f"\n--- DROPPED CLASS PREDICTION ANALYSIS ---")
        predictions = np.array(full_results['all_predictions'])
        targets = np.array(full_results['all_targets'])
        
        for dropped_class in drop_classes:
            # How often does the model predict the dropped class?
            pred_count = np.sum(predictions == dropped_class)
            print(f"Model predicts dropped class {dropped_class}: {pred_count} times")
            
            # What does the model predict for dropped class samples?
            dropped_mask = targets == dropped_class
            if np.sum(dropped_mask) > 0:
                dropped_predictions = predictions[dropped_mask]
                unique_preds, counts = np.unique(dropped_predictions, return_counts=True)
                print(f"  For actual class {dropped_class} samples, model predicts:")
                for pred, count in zip(unique_preds, counts):
                    print(f"    Class {pred}: {count} times ({count/len(dropped_predictions)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze model performance class-by-class")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the model pickle file")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to the configuration YAML file")
    parser.add_argument("--model_idx", type=int, default=0,
                       help="Model index for logging purposes")
    
    args = parser.parse_args()
    
    analyze_dropped_classes_effect(args.model_path, args.config_path, args.model_idx)


if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python analyze_model_performance.py \\")
        print("  --model_path ml_privacy_meter/logs/demo_cifar20_clsdrop_0/models/model_0.pkl \\")
        print("  --config_path configs/cifar20_clsdrop.yaml \\")
        print("  --model_idx 0")
        print("\nOr run with specific arguments...")
        
        # Try to find and analyze automatically
        default_model = "logs/demo_cifar20_clsdrop_0/models/model_0.pkl"
        default_config = "configs/cifar20_clsdrop.yaml"
        
        if os.path.exists(default_model) and os.path.exists(default_config):
            print(f"\nFound default files, analyzing...")
            analyze_dropped_classes_effect(default_model, default_config, 0)
        else:
            print(f"\nDefault files not found:")
            print(f"  Model: {default_model} - {'✓' if os.path.exists(default_model) else '✗'}")
            print(f"  Config: {default_config} - {'✓' if os.path.exists(default_config) else '✗'}")
    else:
        main() 