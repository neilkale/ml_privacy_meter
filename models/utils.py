"""This module defines functions for model handling, including model definition, loading, and training."""

import copy
import json
import logging
import os
import pickle
import time
import multiprocessing as mp
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torchvision
from transformers import AutoModelForCausalLM

from dataset.utils import get_dataloader
from models import AlexNet, CNN, MLP, WideResNet
from trainers.default_trainer import train, inference, dp_train
from trainers.fast_train import (
    load_cifar10_data,
    NetworkEMA,
    make_net,
    print_training_details,
    logging_columns_list,
    fast_train_fun,
)
from trainers.train_transformers import *
from peft import get_peft_model


INPUT_OUTPUT_SHAPE = {
    "cifar10": [3, 10],
    "cifar100": [3, 100],
    "cifar20": [3, 20],
    "purchase100": [600, 100],
    "texas100": [6169, 100],
}


def get_model(model_type: str, dataset_name: str, configs: dict):
    """
    Instantiate and return a model based on the given model type and dataset name.

    Args:
        model_type (str): Type of the model to be instantiated.
        dataset_name (str): Name of the dataset the model will be used for.
        configs (dict): Configuration dictionary containing information about the model.
    Returns:
        torch.nn.Module or PreTrainedModel: An instance of the specified model, ready for training or inference.
    """
    if model_type == "gpt2":
        if configs.get("peft_type", None) is None:
            return AutoModelForCausalLM.from_pretrained(model_type)
        else:
            peft_config = get_peft_model_config(configs)
            return get_peft_model(
                AutoModelForCausalLM.from_pretrained(model_type), peft_config
            )

    num_classes = INPUT_OUTPUT_SHAPE[dataset_name][1]
    in_shape = INPUT_OUTPUT_SHAPE[dataset_name][0]
    if model_type == "CNN":
        return CNN(num_classes=num_classes)
    elif model_type == "alexnet":
        return AlexNet(num_classes=num_classes)
    elif model_type == "wrn28-1":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=1)
    elif model_type == "wrn28-2":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=2)
    elif model_type == "wrn28-10":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=10)
    elif model_type == "mlp":  # for purchase dataset
        return MLP(in_shape=in_shape, num_classes=num_classes)
    elif model_type == "vgg16":
        return torchvision.models.vgg16(pretrained=False)
    else:
        raise NotImplementedError(f"{model_type} is not implemented")


def load_existing_model(
    model_metadata: dict, dataset: torchvision.datasets, device: str, config: dict
):
    """Load an existing model from disk based on the provided metadata.

    Args:
        model_metadata (dict): Metadata dictionary containing information about the model.
        dataset (torchvision.datasets): Dataset object used to instantiate the model.
        device (str): The device on which to load the model, such as 'cpu' or 'cuda'.
        config (dict): Configuration dictionary containing information about the model.
    Returns:
        model (torch.nn.Module): Loaded model object with weights restored from disk.
    """
    model_name = model_metadata["model_name"]
    dataset_name = model_metadata["dataset"]

    if model_name != "speedyresnet":
        model = get_model(model_name, dataset_name, config)
    else:
        data = load_cifar10_data(dataset, [0], [0], device=device)
        model = NetworkEMA(make_net(data, device=device))

    model_checkpoint_extension = os.path.splitext(model_metadata["model_path"])[1]
    if model_checkpoint_extension == ".pkl":
        with open(model_metadata["model_path"], "rb") as file:
            model_weight = pickle.load(file)
        model.load_state_dict(model_weight)
    elif model_checkpoint_extension == ".pt" or model_checkpoint_extension == ".pth":
        model.load_state_dict(torch.load(model_metadata["model_path"]))
    elif model_checkpoint_extension == "":
        if isinstance(model, PreTrainedModel):
            model = model.from_pretrained(model_metadata["model_path"])
        else:
            raise ValueError(f"Model path is invalid.")
    else:
        raise ValueError(f"Model path is invalid.")
    return model


def dp_load_existing_model(
    model_metadata: dict, dataset: torchvision.datasets, device: str, config: dict
):
    """Load an existing model from disk based on the provided metadata.

    Args:
        model_metadata (dict): Metadata dictionary containing information about the model.
        dataset (torchvision.datasets): Dataset object used to instantiate the model.
        device (str): The device on which to load the model, such as 'cpu' or 'cuda'.
        config (dict): Configuration dictionary containing information about the model.
    Returns:
        model (torch.nn.Module): Loaded model object with weights restored from disk.
    """
    from opacus.validators import ModuleValidator

    model_name = model_metadata["model_name"]
    dataset_name = model_metadata["dataset"]

    if model_name != "speedyresnet":
        model = get_model(model_name, dataset_name, config)
    else:
        data = load_cifar10_data(dataset, [0], [0], device=device)
        model = NetworkEMA(make_net(data, device=device))

    model = ModuleValidator.fix(model)

    def remove_module_from_state_dict(old_state_dict):
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            name = k.replace(
                "_module.", ""
            )  # removing ‘.moldule’ from key # remove `module.`
            new_state_dict[name] = v
        return new_state_dict

    model_checkpoint_extension = os.path.splitext(model_metadata["model_path"])[1]
    if model_checkpoint_extension == ".pkl":
        with open(model_metadata["model_path"], "rb") as file:
            model_weight = pickle.load(file)
        model_weight = remove_module_from_state_dict(model_weight)
        model.load_state_dict(model_weight)
    else:
        raise ValueError(f"DP Model path is invalid.")
    return model


def load_models(log_dir, dataset, num_models, configs, logger):
    """
    Load trained models from disk if available.

    Args:
        log_dir (str): Path to the directory containing model logs and metadata.
        dataset (torchvision.datasets): Dataset object used for model training.
        num_models (int): Number of models to be loaded from disk.
        configs (dict): Dictionary of configuration settings, including device information.
        logger (logging.Logger): Logger object for logging the model loading process.

    Returns:
        model_list (list of nn.Module): List of loaded model objects.
        all_memberships (np.array): Membership matrix for all loaded models, indicating training set membership.
    """
    experiment_dir = f"{log_dir}/models"
    if os.path.exists(f"{experiment_dir}/models_metadata.json"):
        with open(f"{experiment_dir}/models_metadata.json", "r") as f:
            model_metadata_dict = json.load(f)
        all_memberships = np.load(f"{experiment_dir}/memberships.npy")
        if len(model_metadata_dict) < num_models:
            return None, None
    else:
        return None, None

    model_list = []
    for model_idx in range(len(model_metadata_dict)):
        logger.info(f"Loading model {model_idx}")
        model_obj = load_existing_model(
            model_metadata_dict[str(model_idx)],
            dataset,
            configs["audit"]["device"],
            configs,
        )
        model_list.append(model_obj)
        if len(model_list) == num_models:
            break
    return model_list, all_memberships


def dp_load_models(log_dir, dataset, num_models, configs, logger):
    """
    Load trained models from disk if available.

    Args:
        log_dir (str): Path to the directory containing model logs and metadata.
        dataset (torchvision.datasets): Dataset object used for model training.
        num_models (int): Number of models to be loaded from disk.
        configs (dict): Dictionary of configuration settings, including device information.
        logger (logging.Logger): Logger object for logging the model loading process.

    Returns:
        model_list (list of nn.Module): List of loaded model objects.
        all_memberships (np.array): Membership matrix for all loaded models, indicating training set membership.
    """
    experiment_dir = f"{log_dir}/models"
    if os.path.exists(f"{experiment_dir}/models_metadata.json"):
        with open(f"{experiment_dir}/models_metadata.json", "r") as f:
            model_metadata_dict = json.load(f)
        all_memberships = np.load(f"{experiment_dir}/memberships.npy")
        if len(model_metadata_dict) < num_models:
            return None, None
    else:
        return None, None

    model_list = []
    for model_idx in range(len(model_metadata_dict)):
        logger.info(f"Loading model {model_idx}")
        model_obj = dp_load_existing_model(
            model_metadata_dict[str(model_idx)],
            dataset,
            configs["audit"]["device"],
            configs,
        )
        model_list.append(model_obj)
        if len(model_list) == num_models:
            break
    return model_list, all_memberships


def train_models(log_dir, dataset, data_split_info, all_memberships, configs, logger):
    """
    Train models based on the dataset split information.

    Args:
        log_dir (str): Path to the directory where models and logs will be saved.
        dataset (torchvision.datasets): Dataset object used for training the models.
        data_split_info (list): List of dictionaries containing training and test split information for each model.
        all_memberships (np.array): Membership matrix indicating which samples were used in training each model.
        configs (dict): Configuration dictionary containing training settings.
        logger (logging.Logger): Logger object for logging the training process.

    Returns:
        model_list (list of nn.Module): List of trained model objects.
    """
    experiment_dir = f"{log_dir}/models"
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Training {len(data_split_info)} models")

    model_list = prepare_models_parallel(
        experiment_dir, dataset, data_split_info, all_memberships, configs, logger
    )
    return model_list


def dp_train_models(
    log_dir, dataset, data_split_info, all_memberships, configs, logger
):
    """
    Train models based on the dataset split information.

    Args:
        log_dir (str): Path to the directory where models and logs will be saved.
        dataset (torchvision.datasets): Dataset object used for training the models.
        data_split_info (list): List of dictionaries containing training and test split information for each model.
        all_memberships (np.array): Membership matrix indicating which samples were used in training each model.
        configs (dict): Configuration dictionary containing training settings.
        logger (logging.Logger): Logger object for logging the training process.

    Returns:
        model_list (list of nn.Module): List of trained model objects.
    """
    experiment_dir = f"{log_dir}/models"
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Training {len(data_split_info)} models")

    model_list = dp_prepare_models(
        experiment_dir, dataset, data_split_info, all_memberships, configs, logger
    )
    return model_list


def split_dataset_for_training(dataset_size, num_model_pairs):
    """
    Split dataset into training and test partitions for model pairs.

    Args:
        dataset_size (int): Total number of samples in the dataset.
        num_model_pairs (int): Number of model pairs to be trained, with each pair trained on different halves of the dataset.

    Returns:
        data_split (list): List of dictionaries containing training and test split indices for each model.
        master_keep (np.array): D boolean array indicating the membership of samples in each model's training set.
    """
    data_splits = []
    indices = np.arange(dataset_size)
    split_index = len(indices) // 2
    master_keep = np.full((2 * num_model_pairs, dataset_size), True, dtype=bool)

    for i in range(num_model_pairs):
        np.random.shuffle(indices)
        master_keep[i * 2, indices[split_index:]] = False
        master_keep[i * 2 + 1, indices[:split_index]] = False
        keep = master_keep[i * 2, :]
        train_indices = np.where(keep)[0]
        test_indices = np.where(~keep)[0]
        data_splits.append(
            {
                "train": train_indices,
                "test": test_indices,
            }
        )
        data_splits.append(
            {
                "train": test_indices,
                "test": train_indices,
            }
        )

    return data_splits, master_keep


def prepare_models(
    log_dir: str,
    dataset: torchvision.datasets,
    data_split_info: list,
    all_memberships: np.array,
    configs: dict,
    logger,
):
    """
    Train models based on the dataset split information and save their metadata.

    Args:
        log_dir (str): Path to the directory where model logs and metadata will be saved.
        dataset (torchvision.datasets): Dataset object used for training.
        data_split_info (list): List of dictionaries containing training and test split indices for each model.
        all_memberships (np.array): Membership matrix indicating which samples were used in training each model.
        configs (dict): Configuration dictionary containing training settings.
        logger (logging.Logger): Logger object for logging the training process.

    Returns:
        list: List of trained model objects.
    """
    np.save(f"{log_dir}/memberships.npy", all_memberships)

    model_metadata_dict = {}
    model_list = []

    # for split, split_info in enumerate(data_split_info):
    for split in range(len(data_split_info)):

        split_info = data_split_info[split]
        # NEW: remove the OOD samples from the training data for the auditing reference models
        if split < configs["run"]["num_experiments"]:
            is_target_model = True
        else:
            is_target_model = False
        if not is_target_model and configs["data"].get("drop_classes", None) is not None:
            drop_classes = configs["data"].get("drop_classes", None)
            dataset_targets = np.array(dataset.targets)
            # Find which indices in train/test sets correspond to OOD classes
            train_ood_mask = np.isin(dataset_targets[split_info["train"]], drop_classes)
            test_ood_mask = np.isin(dataset_targets[split_info["test"]], drop_classes)
            # Remove those indices
            split_info["train"] = split_info["train"][~train_ood_mask]
            split_info["test"] = split_info["test"][~test_ood_mask]
            logger.info(f"Removed {train_ood_mask.sum() + test_ood_mask.sum()} OOD samples from reference model {split}")
        
        baseline_time = time.time()
        logger.info(50 * "-")
        logger.info(
            f"Training model {split}: Train size {len(split_info['train'])}, Test size {len(split_info['test'])}"
        )

        model_name, dataset_name, batch_size, device = (
            configs["train"]["model_name"],
            configs["data"]["dataset"],
            configs["train"]["batch_size"],
            configs["train"]["device"],
        )

        if model_name == "gpt2":
            hf_dataset = dataset.hf_dataset
            if configs.get("peft_type", None) is None:
                model, train_loss, test_loss = train_transformer(
                    hf_dataset.select(split_info["train"]),
                    get_model(model_name, dataset_name, configs),
                    configs,
                    hf_dataset.select(split_info["test"]),
                )
            else:
                # Fine-tuning with PEFT
                model, train_loss, test_loss = train_transformer_with_peft(
                    hf_dataset.select(split_info["train"]),
                    get_peft_model(
                        get_model(model_name, dataset_name, configs),
                        get_peft_model_config(configs),
                    ),
                    configs,
                    hf_dataset.select(split_info["test"]),
                )
            train_acc, test_acc = None, None
        elif model_name != "speedyresnet":
            train_loader = get_dataloader(
                torch.utils.data.Subset(dataset, split_info["train"]),
                batch_size=batch_size,
                shuffle=True,
            )
            test_loader = get_dataloader(
                torch.utils.data.Subset(dataset, split_info["test"]),
                batch_size=batch_size,
            )
            model = train(
                get_model(model_name, dataset_name, configs),
                train_loader,
                configs["train"],
                test_loader,
            )
            test_loss, test_acc = inference(model, test_loader, device)
            train_loss, train_acc = inference(model, train_loader, device)
            logger.info(f"Train accuracy {train_acc}, Train Loss {train_loss}")
            logger.info(f"Test accuracy {test_acc}, Test Loss {test_loss}")
        elif model_name == "speedyresnet" and dataset_name == "cifar10":
            data = load_cifar10_data(
                dataset,
                split_info["train"],
                split_info["test"],
                device=device,
            )
            eval_batch_size, test_size = batch_size, len(split_info["test"])
            divisors = [
                factor
                for i in range(1, int(np.sqrt(test_size)) + 1)
                if test_size % i == 0
                for factor in (i, test_size // i)
                if factor <= eval_batch_size
            ]
            eval_batch_size = max(divisors)  # to support smaller GPUs
            print_training_details(logging_columns_list, column_heads_only=True)
            model, train_acc, train_loss, test_acc, test_loss = fast_train_fun(
                data,
                make_net(data, device=device),
                eval_batchsize=eval_batch_size,
                device=device,
            )
        else:
            raise ValueError(
                f"The {model_name} is not supported for the {dataset_name}"
            )

        model_list.append(copy.deepcopy(model))
        logger.info(
            "Training model %s took %s seconds",
            split,
            time.time() - baseline_time,
        )

        model_idx = split

        with open(f"{log_dir}/model_{model_idx}.pkl", "wb") as f:
            pickle.dump(model.state_dict(), f)

        model_metadata_dict[model_idx] = {
            "num_train": len(split_info["train"]),
            "optimizer": configs["train"]["optimizer"],
            "batch_size": batch_size,
            "epochs": configs["train"]["epochs"],
            "model_name": model_name,
            "learning_rate": configs["train"]["learning_rate"],
            "weight_decay": configs["train"]["weight_decay"],
            "model_path": f"{log_dir}/model_{model_idx}.pkl",
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "dataset": dataset_name,
        }

    with open(f"{log_dir}/models_metadata.json", "w") as f:
        json.dump(model_metadata_dict, f, indent=4)

    return model_list


def _train_and_save_worker(
    split: int,
    split_info: dict,
    dataset: torchvision.datasets,
    configs: dict,
    log_dir: str,
    device: str,
    results_queue: mp.Queue,
):
    """
    Worker function to train a single model on a specific GPU.
    This function is called by each process in the multiprocessing pool.
    """
    # Create a worker-specific copy of train configs to set the device
    worker_train_configs = copy.deepcopy(configs["train"])
    worker_train_configs["device"] = device

    # Pre-process split_info to remove OOD samples for reference models
    if split >= configs["run"]["num_experiments"] and configs["data"].get("drop_classes"):
        drop_classes = configs["data"]["drop_classes"]
        dataset_targets = np.array(dataset.targets)
        train_ood_mask = np.isin(dataset_targets[split_info["train"]], drop_classes)
        test_ood_mask = np.isin(dataset_targets[split_info["test"]], drop_classes)
        split_info["train"] = split_info["train"][~train_ood_mask]
        split_info["test"] = split_info["test"][~test_ood_mask]

    baseline_time = time.time()
    print(50 * "-")
    print(
        f"Worker {split}: Starting training on {device}. "
        f"Train size {len(split_info['train'])}, Test size {len(split_info['test'])}"
    )

    model_name, dataset_name, batch_size = (
        configs["train"]["model_name"],
        configs["data"]["dataset"],
        configs["train"]["batch_size"],
    )

    # --- Training logic, adapted to use the assigned `device` ---
    if model_name == "gpt2":
        # GPT-2 training logic... (assuming it uses device from configs)
        hf_dataset = dataset.hf_dataset
        worker_configs = copy.deepcopy(configs)
        worker_configs['train']['device'] = device
        # ... (original gpt2 training calls using worker_configs)
        train_acc, test_acc = None, None

    elif model_name != "speedyresnet":
        train_loader = get_dataloader(
            torch.utils.data.Subset(dataset, split_info["train"]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = get_dataloader(
            torch.utils.data.Subset(dataset, split_info["test"]),
            batch_size=batch_size,
            num_workers=0,
        )
        # Explicitly move model to the assigned device
        model = get_model(model_name, dataset_name, configs).to(device)
        model = train(model, train_loader, worker_train_configs, test_loader)
        test_loss, test_acc = inference(model, test_loader, device)
        train_loss, train_acc = inference(model, train_loader, device)
        print(f"Worker {split}: Test accuracy {test_acc:.4f}, Test Loss {test_loss:.4f}")

    elif model_name == "speedyresnet" and dataset_name == "cifar10":
        # SpeedyResNet training logic...
        # ... (original speedyresnet calls using the assigned `device`)
        pass # Placeholder for your speedyresnet logic

    else:
        raise ValueError(f"The model {model_name} is not supported for {dataset_name}")

    print(f"Worker {split}: Training took {time.time() - baseline_time:.2f} seconds.")

    # Save model state_dict to file
    model_path = f"{log_dir}/model_{split}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model.to("cpu").state_dict(), f)

    # Prepare metadata for this model
    metadata = {
        "num_train": len(split_info["train"]),
        "optimizer": configs["train"]["optimizer"],
        "batch_size": batch_size,
        "epochs": configs["train"]["epochs"],
        "model_name": model_name,
        "learning_rate": configs["train"]["learning_rate"],
        "weight_decay": configs["train"]["weight_decay"],
        "model_path": model_path,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "dataset": dataset_name,
    }

    # Return results, moving the model object to CPU to free GPU memory
    results_queue.put((split, model.to("cpu").state_dict(), metadata))
    print(f"Worker {split}: Results placed in queue.")

def prepare_models_parallel(
    log_dir: str,
    dataset: torchvision.datasets,
    data_split_info: list,
    all_memberships: np.array,
    configs: dict,
    logger,
):
    """
    Trains models in parallel using manually managed, non-daemonic processes,
    allowing DataLoader to use its own workers.
    """
    np.save(f"{log_dir}/memberships.npy", all_memberships)

    devices = configs["train"].get("device", "cpu")
    if isinstance(devices, str):
        devices = [devices]
    
    # 'spawn' is still the recommended start method for CUDA
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass # Already set

    # Use the Manager as a context manager
    with mp.Manager() as manager:
        # 1. Create the queue from the manager
        results_queue = manager.Queue()

        device_cycle = cycle(devices)
        processes = []
        num_models = len(data_split_info)

        # 2. Create and start a process for each model
        for split in range(num_models):
            process_args = (
                split,
                data_split_info[split],
                dataset,
                configs,
                log_dir,
                next(device_cycle),
                results_queue,
            )
            p = mp.Process(target=_train_and_save_worker, args=process_args)
            processes.append(p)
            p.start()
            logger.info(f"Launched process for model {split}.")

        # 3. Collect results from the queue
        results = []
        for _ in range(num_models):
            results.append(results_queue.get())
        
        # 4. Wait for all processes to complete
        for p in processes:
            p.join()

    # The manager and its queue are automatically shut down here upon exiting the 'with' block
    logger.info("All training processes and the manager have completed.")

    # --- Process and save the collected results ---
    results.sort(key=lambda x: x[0]) # Sort by model index

    model_metadata_dict = {}
    model_list = []

    # Recreate model objects from the state_dicts
    for split_idx, state_dict, metadata in results:
        model_metadata_dict[split_idx] = metadata
        # Re-create the model and load its trained state
        model_obj = get_model(
            configs["train"]["model_name"], 
            configs["data"]["dataset"], 
            configs
        )
        model_obj.load_state_dict(state_dict)
        model_list.append(model_obj)

    with open(f"{log_dir}/models_metadata.json", "w") as f:
        json.dump(model_metadata_dict, f, indent=4)
    logger.info(f"Saved aggregated model metadata.")

    return model_list

def dp_prepare_models(
    log_dir: str,
    dataset: torchvision.datasets,
    data_split_info: list,
    all_memberships: np.array,
    configs: dict,
    logger,
):
    """
    Train models based on the dataset split information and save their metadata.

    Args:
        log_dir (str): Path to the directory where model logs and metadata will be saved.
        dataset (torchvision.datasets): Dataset object used for training.
        data_split_info (list): List of dictionaries containing training and test split indices for each model.
        all_memberships (np.array): Membership matrix indicating which samples were used in training each model.
        configs (dict): Configuration dictionary containing training settings.
        logger (logging.Logger): Logger object for logging the training process.

    Returns:
        list: List of trained model objects.
    """
    np.save(f"{log_dir}/memberships.npy", all_memberships)

    model_metadata_dict = {}
    model_list = []

    # for split, split_info in enumerate(data_split_info):
    for split in range(len(data_split_info)):
        split_info = data_split_info[split]
        baseline_time = time.time()
        logger.info(50 * "-")
        logger.info(
            f"Training model {split}: Train size {len(split_info['train'])}, Test size {len(split_info['test'])}"
        )

        model_name, dataset_name, batch_size, device = (
            configs["train"]["model_name"],
            configs["data"]["dataset"],
            configs["train"]["batch_size"],
            configs["train"]["device"],
        )

        if model_name == "gpt2":
            raise ValueError(
                f"DP training is not supported for model {model_name} and dataset {dataset_name}"
            )
        elif model_name != "speedyresnet":
            train_loader = get_dataloader(
                torch.utils.data.Subset(dataset, split_info["train"]),
                batch_size=batch_size,
                shuffle=True,
            )
            test_loader = get_dataloader(
                torch.utils.data.Subset(dataset, split_info["test"]),
                batch_size=batch_size,
            )
            model, epsilon = dp_train(
                get_model(model_name, dataset_name, configs),
                train_loader,
                configs["train"],
                test_loader,
            )
            test_loss, test_acc = inference(model, test_loader, device)
            train_loss, train_acc = inference(model, train_loader, device)
            logger.info(
                f"Train accuracy {train_acc}, Train Loss {train_loss} (epsilon = {epsilon}, delta = 1e-5)"
            )
            logger.info(f"Test accuracy {test_acc}, Test Loss {test_loss}")
        elif model_name == "speedyresnet" and dataset_name == "cifar10":
            raise ValueError(
                f"DP training is not supported for model {model_name} and dataset {dataset_name}"
            )
        else:
            raise ValueError(
                f"The {model_name} is not supported for the {dataset_name}"
            )

        model_list.append(copy.deepcopy(model))
        logger.info(
            "Training model %s took %s seconds",
            split,
            time.time() - baseline_time,
        )

        model_idx = split

        with open(f"{log_dir}/model_{model_idx}.pkl", "wb") as f:
            pickle.dump(model.state_dict(), f)

        model_metadata_dict[model_idx] = {
            "num_train": len(split_info["train"]),
            "optimizer": configs["train"]["optimizer"],
            "batch_size": batch_size,
            "epochs": configs["train"]["epochs"],
            "model_name": model_name,
            "learning_rate": configs["train"]["learning_rate"],
            "weight_decay": configs["train"]["weight_decay"],
            "model_path": f"{log_dir}/model_{model_idx}.pkl",
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "dataset": dataset_name,
            "epsilon": epsilon,
            "delta": 1e-5,
        }

    with open(f"{log_dir}/models_metadata.json", "w") as f:
        json.dump(model_metadata_dict, f, indent=4)

    return model_list
