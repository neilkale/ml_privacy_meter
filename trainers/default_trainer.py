"""This file contains functions for training and testing the model."""

import pdb
import time
from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler


def lr_update(step: int, total_epoch: int, train_size: int, initial_lr: float) -> float:
    """
    Updates learning rate using cosine decay schedule,
    from https://github.com/tensorflow/privacy/blob/4e1fc252e4c64132ad6fcd838e93f071f38dedd7/research/mi_lira_2021/train.py#L58

    Args:
        step (int): Current step number.
        total_epoch (int): Total number of epochs.
        train_size (int): Size of the training dataset.
        initial_lr (float): Initial learning rate.

    Returns:
        float: Updated learning rate.
    """
    progress = step / (total_epoch * train_size)
    lr = initial_lr * np.cos(progress * (7 * np.pi) / (2 * 8))
    lr *= np.clip(progress * 100, 0, 1)
    return lr


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    configs: Dict,
    test_loader: torch.utils.data.DataLoader = None,
) -> torch.nn.Module:
    """
    Trains the model using the provided training data.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        configs (dict): Configuration dictionary for training.
        test_loader (torch.utils.data.DataLoader, optional): DataLoader for test data. Defaults to None.

    Returns:
        torch.nn.Module: Trained model.
    """
    # Ensure the model is moved to the correct device (e.g., cuda:1 or cpu)
    device = configs.get("device", "cpu")
    model = model.to(device)  # Make sure the model is on the correct device

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, configs)

    epochs = configs.get("epochs", 1)

    if configs.get("scheduler", None) == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=int(epochs * configs.get("scheduler_step_fraction", 0.3)),
            gamma=configs.get("scheduler_step_gamma", 0.2),
        )
    else:
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step * 256, epochs, len(train_loader) * 256, 0.1
            ),
        )

    for epoch_idx in range(epochs):
        start_time = time.time()
        total_loss, correct_predictions = 0, 0

        # Set model to training mode
        model.train()

        for data, target in train_loader:
            # Ensure that both data and target are moved to the same device as the model
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True).long(),
            )

            optimizer.zero_grad(set_to_none=True)

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_predictions += pred.eq(target).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct_predictions / len(train_loader.dataset)

        print(
            f"Epoch [{epoch_idx + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        )

        if test_loader:
            test_loss, test_acc = inference(model, test_loader, device)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        print(f"Epoch {epoch_idx + 1} took {time.time() - start_time:.2f} seconds")

    # Move the model back to CPU if needed (this is optional)
    model.to("cpu")
    return model


def dp_train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    configs: Dict,
    test_loader: torch.utils.data.DataLoader = None,
) -> torch.nn.Module:
    """
    Trains the model using the provided training data.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        configs (dict): Configuration dictionary for training.
        test_loader (torch.utils.data.DataLoader, optional): DataLoader for test data. Defaults to None.

    Returns:
        torch.nn.Module: Trained model.
    """
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    # Ensure the model is moved to the correct device (e.g., cuda:1 or cpu)
    device = configs.get("device", "cpu")
    model = model.to(device)  # Make sure the model is on the correct device
    model = ModuleValidator.fix(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, configs)

    epochs = configs.get("epochs", 1)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_update(
            step * 256, epochs, len(train_loader) * 256, 0.1
        ),
    )

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
    )

    for epoch_idx in range(epochs):
        start_time = time.time()
        total_loss, correct_predictions = 0, 0

        # Set model to training mode
        model.train()

        for data, target in train_loader:
            # Ensure that both data and target are moved to the same device as the model
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True).long(),
            )

            optimizer.zero_grad(set_to_none=True)

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_predictions += pred.eq(target).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct_predictions / len(train_loader.dataset)

        epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)

        print(
            f"Epoch [{epoch_idx + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | DP guarantee: (ε = {epsilon:.2f}, δ = {1e-5})"
        )

        if test_loader:
            test_loss, test_acc = inference(model, test_loader, device)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        print(f"Epoch {epoch_idx + 1} took {time.time() - start_time:.2f} seconds")

    # Move the model back to CPU if needed (this is optional)
    model.to("cpu")
    epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
    return model, epsilon


def inference(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> Tuple[float, float]:
    """
    Evaluates the model on the provided data.

    Args:
        model (torch.nn.Module): Model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (str): Device to use for computation ('cpu' or 'cuda').

    Returns:
        Tuple[float, float]: Loss and accuracy on the evaluation data.
    """
    model.eval().to(device)  # Make sure the model is on the correct device
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct_predictions = 0, 0

    with torch.no_grad():
        for data, target in loader:
            # Ensure data and target are moved to the same device as the model
            data, target = data.to(device), target.to(device).long()

            output = model(data)
            loss = loss_fn(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_predictions += pred.eq(target).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / len(loader.dataset)

    return avg_loss, accuracy

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    """
    Set differential weight decay for different parameter groups.
    
    Args:
        model: PyTorch model
        skip_list: List of parameter names to skip weight decay
        skip_keywords: List of keywords to identify parameters to skip weight decay
        
    Returns:
        List of parameter groups with different weight decay values
    """
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1  # bias terms
            or name.endswith(".bias")  # explicit bias parameters
            or (name in skip_list)  # manually specified parameters
            or check_keywords_in_name(name, skip_keywords)  # parameters with specific keywords
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)
    
    return [
        {"params": has_decay, "weight_decay": 5e-4},  # Apply weight decay to weights
        {"params": no_decay, "weight_decay": 0.0}     # No weight decay for bias/norm layers
    ]


def check_keywords_in_name(name, keywords=()):
    """
    Check if any keyword is present in the parameter name.
    
    Args:
        name: Parameter name
        keywords: List of keywords to check for
        
    Returns:
        Boolean indicating if any keyword is found
    """
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def get_optimizer(model: torch.nn.Module, configs: Dict) -> torch.optim.Optimizer:
    """
    Returns the optimizer based on the configuration.

    Args:
        model (torch.nn.Module): Model for which to create the optimizer.
        configs (dict): Configuration dictionary.

    Raises:
        NotImplementedError: If the specified optimizer is not supported.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """
    optimizer_name = configs.get("optimizer", "SGD")
    learning_rate = configs.get("learning_rate", 0.001)
    weight_decay = configs.get("weight_decay", 0.0)
    momentum = configs.get("momentum", 0.0)

    # Check if differential weight decay is enabled
    use_differential_wd = configs.get("differential_weight_decay", False)
    
    if use_differential_wd:
        # Use differential weight decay
        parameters = set_weight_decay(model)
        print(f"Using differential weight decay: 5e-4 for weights, 0.0 for bias/norm layers")
    else:
        # Use uniform weight decay
        parameters = model.parameters()
        print(f"Using uniform weight decay: {weight_decay}")

    print(
        f"Using optimizer: {optimizer_name} | Learning Rate: {learning_rate} | Weight Decay: {weight_decay} | Momentum: {momentum}"
    )

    if optimizer_name == "SGD":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optimizer_name == "SGD_Nesterov":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
        )
    elif optimizer_name == "Adam":
        return torch.optim.Adam(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise NotImplementedError(
            f"Optimizer '{optimizer_name}' is not implemented. Choose 'SGD', 'SGD_Nesterov', 'Adam', or 'AdamW'."
        )
