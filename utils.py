import torch                            # Core pytorch for tensor operator
import torch.nn as nn                   # Neural network module (layers, loss_function,..)
import torch.optim as optim             # optimization algorithms (SDG, RMSprop, Adam,...)
import torch.nn.functional as F         # Functional operations (activations, pooling,..)
import torch.utils.data as data         # Dataset handling and batching

import torchvision                                  # Computer vision utilities and pre-trained models
from torchvision import models
from torchvision import transforms                  # Image processing and augmentation
from torchvision import datasets                    # Image processing and augmentation
from torchvision.datasets import CIFAR100           # common CV datasets
from torchvision.models import resnet50             # Pre-trained model supported in PyTorch
from torchvision.models import ViT_B_16_Weights
from torchvision.models import vit_b_16
from torchinfo import summary

f"Torch version: {torch.__version__}", "^",\
f"Torchvision version: {torchvision.__version__}"

"""
A series of helper functions used throughout the course.
If a function gets defined once and could be used over and over, it'll go in here.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
from pathlib import Path
import requests
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import time
import sys
import warnings
import random
from PIL import Image 
from datetime import datetime
warnings.filterwarnings("ignore")

# "///////////////////////////////// <Helper Func> /////////////////////////////////////////////////"

# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()






# "///////////////////////////////// <Engine> /////////////////////////////////////////////////"

# This code was for training step (hidden recommended)
def format_time(seconds: float) -> str:                          # Time formatting helper
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

def train_step(model: nn.Module,
               dataloader: data.DataLoader,
               loss_fn: nn.Module,
               optimizer: optim.Optimizer,
               device: torch.device,
               epoch: int,
               total_epochs: int,
               start_time: float) -> Tuple[float, float]:
    """"""
    model.train()                                            # Set model to training mode
    train_loss, train_acc = 0, 0                             # Initialize training metrics
    total_steps = len(dataloader)                            # Total batches in epoch
    
    for batch, (X, y) in enumerate(dataloader):              # Training loop
        X, y = X.to(device), y.to(device)                    # Move data to device
        y_pred = model(X)                                    # Forward pass
        loss = loss_fn(y_pred, y)                            # Calculate loss
        train_loss += loss.item()                            # Accumulate loss
        
        optimizer.zero_grad()                                # Clear gradients
        loss.backward()                                      # Backward pass
        optimizer.step()                                     # Update weights
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)  # Calculate predictions
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)         # Accumulate accuracy
        
        progress = (batch + 1) / total_steps                 # Calculate progress
        elapsed = time.time() - start_time                   # Time elapsed
        eta = elapsed * (total_steps - (batch + 1)) / max(batch + 1, 1)  # Estimated time remaining
        bar_length = 30                                      # Progress bar length
        filled = int(bar_length * progress)                  # Filled portion of bar
        bar = '=' * filled + '-' * (bar_length - filled)    # Create bar string
        
        metrics = {                                          # Current metrics
            'loss': train_loss/(batch + 1),
            'acc': train_acc/(batch + 1),
            'lr': optimizer.param_groups[0]['lr']
        }
        metrics_str = ' - '.join(f'{k}: {v:.4f}' for k, v in metrics.items())  # Format metrics
        
        status = f'Epoch {epoch+1}/{total_epochs} [{bar}] {progress:.0%} '  # Build status string
        status += f'[{format_time(elapsed)}<{format_time(eta)}] {metrics_str}'
        sys.stdout.write(f'\r{status}')                      # Update progress bar
        sys.stdout.flush()
    
    train_loss = train_loss / total_steps                    # Average training metrics
    train_acc = train_acc / total_steps
    sys.stdout.write('\n')                                   # New line after epoch
    sys.stdout.flush()
    
    return train_loss, train_acc

def test_step(model: nn.Module,
              dataloader: data.DataLoader,
              loss_fn: nn.Module,
              device: torch.device) -> Tuple[float, float]:
    model.eval()                                             # Set model to evaluation mode
    test_loss, test_acc = 0, 0                               # Initialize testing metrics
    
    with torch.no_grad():                                    # Inference mode for testing
        for X, y in dataloader:                              # Testing loop
            X, y = X.to(device), y.to(device)                # Move data to device
            test_pred_logits = model(X)                      # Forward pass
            test_loss += loss_fn(test_pred_logits, y).item() # Accumulate loss
            test_pred_labels = test_pred_logits.argmax(dim=1)  # Calculate predictions
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)  # Accumulate accuracy
    
    test_loss = test_loss / len(dataloader)                  # Average testing metrics
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train_model(model: nn.Module,
                train_dataloader: data.DataLoader,
                test_dataloader: data.DataLoader,
                optimizer: optim.Optimizer,
                loss_fn: nn.Module,
                epochs: int,
                device: torch.device,
                seed: int = 16) -> Dict[str, List]:
    
    torch.manual_seed(seed)                                  # Set seed for reproducibility
    torch.cuda.manual_seed(seed)                             # Set CUDA seed for GPU operations
    
    results = {"train_loss": [],                             # Initialize results dictionary
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    
    start_time = time.time()                                 # Track total training time
    
    for epoch in range(epochs):                              # Main epoch loop
        train_loss, train_acc = train_step(                  # Perform training step
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            start_time=start_time
        )
        
        test_loss, test_acc = test_step(                     # Perform testing step
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        
        results["train_loss"].append(train_loss)             # Store results
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    return results


# "///////////////////////////////// <Data> /////////////////////////////////////////////////"



def create_dataloaders(train_dir: str, 
                        valid_dir: str, 
                        transform: transforms.Compose, 
                        batch_size: int, 
                        num_workers: int):
    """
    Creates training and testing DataLoaders.
    
    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=transform)
    
    # Get class names
    class_names = train_data.classes
    
    # NUM_WORKERS = 

    print(f"Maximum number of cpu core available: {num_workers}")
    # Turn images into data loaders
    train_dataloader = data.DataLoader(train_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=True,)
    
    valid_dataloader = data.DataLoader(valid_data,
                                      batch_size=batch_size,
                                      shuffle=False, # don't need to shuffle test data
                                      num_workers=num_workers,
                                      pin_memory=True,)

    return train_dataloader, valid_dataloader, class_names

# "///////////////////////////////// <Data> /////////////////////////////////////////////////"


def save_model(model: nn.Module,
               target_dir: str = "saved_models",
               model_name: str = None) -> None:

    model_name = model._get_name() if model_name == None else model_name
    timestamp = datetime.now().strftime("%d_%m_%Y")         # Format date as day_month_year
    full_model_name = f"{model_name}_{timestamp}_{torch.cuda.get_device_name()}.pth"       # Combine name with timestamp
    target_dir_path = Path(target_dir)                      # Convert to Path object
    target_dir_path.mkdir(parents=True, exist_ok=True)      # Create directory if needed
    model_save_path = target_dir_path / full_model_name     # Complete file path
    # Save the model
    print(f"[INFO] Saving model to: {model_save_path}")     # Log save operation
    torch.save(obj=model.state_dict(), f=model_save_path)   # Save model state_dict