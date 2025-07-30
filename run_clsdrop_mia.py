"""This file is the main entry point for running the privacy auditing tool."""

import argparse
import math
import pdb
import time

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from audit import get_average_audit_results, audit_models, sample_auditing_dataset
from get_signals import get_model_signals
from models.utils import load_models, train_models, split_dataset_for_training
from util import (
    check_configs,
    setup_log,
    initialize_seeds,
    create_directories,
    load_dataset,
    split_dataset_for_clsdrop,
)

# Enable benchmark mode in cudnn to improve performance when input sizes are consistent
torch.backends.cudnn.benchmark = True


def main():
    print(20 * "-")
    print("Privacy Meter Tool!")
    print(20 * "-")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run privacy auditing tool.")
    parser.add_argument(
        "--cf",
        type=str,
        default="configs/cifar10.yaml",
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    # Load configuration file
    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    # Validate configurations
    check_configs(configs)

    # Initialize seeds for reproducibility
    initialize_seeds(configs["run"]["random_seed"])

    # Create necessary directories
    log_dir = configs["run"]["log_dir"]
    directories = {
        "log_dir": log_dir,
        "report_dir": f"{log_dir}/report/{configs['audit']['algorithm']}",
        "signal_dir": f"{log_dir}/signals",
        "data_dir": configs["data"]["data_dir"],
    }
    create_directories(directories)

    # Set up logger
    logger = setup_log(
        directories["report_dir"], "time_analysis", configs["run"]["time_log"]
    )

    start_time = time.time()

    # Load the dataset
    baseline_time = time.time()
    dataset, population = load_dataset(configs, directories["data_dir"], logger)
    dataset_id, dataset_ood = split_dataset_for_clsdrop(dataset, configs["data"]["drop_classes"])
    logger.info(f"Dataset split into {len(dataset_id)} ID and {len(dataset_ood)} OOD samples")
    population_id, population_ood = split_dataset_for_clsdrop(population, configs["data"]["drop_classes"])
    logger.info(f"Population split into {len(population_id)} ID and {len(population_ood)} OOD samples")
    logger.info("Loading dataset took %0.5f seconds", time.time() - baseline_time)

    # Define experiment parameters
    num_experiments = configs["run"]["num_experiments"]
    num_reference_models = configs["audit"]["num_ref_models"]
    num_model_pairs = max(math.ceil(num_experiments / 2.0), num_reference_models + 1)        

    # Load or train models
    baseline_time = time.time()
    models_list, memberships = load_models(
        log_dir, dataset, num_model_pairs * 2, configs, logger
    )
    if models_list is None:
        # Split dataset for training two models per pair
        data_splits, memberships = split_dataset_for_training(
            len(dataset), num_model_pairs
        )
        models_list = train_models(
            log_dir, dataset, data_splits, memberships, configs, logger
        )
    logger.info(
        "Model loading/training took %0.1f seconds", time.time() - baseline_time
    )

    # In-distribution (ID) and out-of-distribution (OOD) auditing
    dsets = [dataset_id, dataset_ood]
    pops = [population_id, population_id]
    labels = ["ID", "OOD"]
    memberships_id = memberships[:, dataset_id.indices]
    memberships_ood = memberships[:, dataset_ood.indices]
    memberships_ood[num_experiments:, :] = False

    mems = [memberships_id, memberships_ood]
    exp_dirs = ["exp_id", "exp_ood"]
    # Set OOD membership to False for all reference models since they did not see OOD samples
    offline_a_tuned = None

    for dset, pop, label, mems, exp_dir in zip(dsets, pops, labels, mems, exp_dirs):
        auditing_dataset, auditing_membership = sample_auditing_dataset(
            configs, dset, logger, mems
        )

        pop = Subset(
            pop,
            np.random.choice(
                len(pop),
                min(configs["audit"].get("population_size", len(pop)), len(pop)),
                replace=False,
            ),
        )

        # Generate signals (softmax outputs) for all models
        baseline_time = time.time()
        signals = get_model_signals(models_list, auditing_dataset, configs, logger)
        population_signals = get_model_signals(
            models_list, pop, configs, logger, is_population=True
        )
        logger.info(f"Preparing {label} signals took %0.5f seconds", time.time() - baseline_time)

        # Perform the privacy audit
        baseline_time = time.time()
        target_model_indices = list(range(num_experiments))
        
        mia_score_list, membership_list, offline_a_tuned = audit_models(
            f"{directories['report_dir']}/{exp_dir}",
            target_model_indices,
            signals,
            population_signals,
            auditing_membership,
            num_reference_models,
            logger,
            configs,
            offline_a_tuned if label == "OOD" else None
        )

        # if label == "OOD":
        #     print(f"Average indicator value for members: {np.mean(signals[membership_list[0],0])}")
        #     print(f"Average indicator value for non-members: {np.mean(~signals[membership_list[0],0])}")
        #     import pdb; pdb.set_trace()

        if len(target_model_indices) > 1:
            logger.info(
                f"Auditing {label} privacy risk took %0.1f seconds", time.time() - baseline_time
            )

        # Get average audit results across all experiments
        if len(target_model_indices) > 1:
            get_average_audit_results(
                directories["report_dir"], mia_score_list, membership_list, logger
            )

    logger.info("Total runtime: %0.5f seconds", time.time() - start_time)


if __name__ == "__main__":
    main()
