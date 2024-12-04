import matplotlib.pyplot as plt
from itertools import product
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import glob
import torch
import os

def visualize_predictions(images, masks, outputs, save_path, epoch, batch_idx):
    '''
    Visualizes and saves sample predictions for a given batch of images, masks, and model outputs.

    Args:
    - images (torch.Tensor): Input images (batch of tensors).
    - masks (torch.Tensor): Ground truth segmentation masks (batch of tensors).
    - outputs (torch.Tensor): Model outputs (batch of tensors).
    - save_path (str): Directory path where the visualization will be saved.
    - epoch (int): Current epoch number (for labeling the file).
    - batch_idx (int): Index of the current batch (for labeling the file).
    
    Functionality:
    - Displays and saves the first few samples from the batch, showing the input images, ground truth masks, and predicted masks.
    - Applies a sigmoid function to the outputs and uses a threshold of 0.5 to convert them to binary masks.
    '''
    os.makedirs(save_path, exist_ok=True)
    
    # Apply sigmoid to outputs and threshold to get binary predictions
    preds = torch.sigmoid(outputs)
    preds = (preds > 0.5).float()
    
    # Detach and convert tensors to numpy for visualization
    images = images.cpu().detach().numpy()
    masks = masks.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()

    num_samples = min(4, images.shape[0])  # Display first few samples in the batch
    fig, axs = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))

    for i in range(num_samples):
        # Display the input image
        axs[i, 0].imshow(images[i].squeeze(), cmap='gray')
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis("off")
        
        # Display the ground truth mask
        axs[i, 1].imshow(masks[i].squeeze(), cmap='gray')
        axs[i, 1].set_title("Ground Truth Mask")
        axs[i, 1].axis("off")
        
        # Display the predicted mask
        axs[i, 2].imshow(preds[i].squeeze(), cmap='gray')
        axs[i, 2].set_title("Predicted Mask")
        axs[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"predictions_epoch_{epoch}_batch_{batch_idx}.jpg"))
    plt.close()

def plot_train_val_history(train_loss_history, val_loss_history, plot_dir, args):
    '''
    Plots and saves the training and validation loss curves.

    Args:
    - train_loss_history (list): List of training loss values over epochs.
    - val_loss_history (list): List of validation loss values over epochs.
    - plot_dir (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    
    Functionality:
    - Plots the train and validation loss curves.
    - Saves the plot as a JPG file in the specified directory.
    '''
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Training and Validation Loss - {args.exp_id}")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(plot_dir, f"train_val_loss_{args.exp_id}.jpg"))
    plt.close()

def plot_metric(x, label, plot_dir, args, metric):
    '''
    Plots and saves a metric curve over epochs.

    Args:
    - x (list): List of metric values over epochs.
    - label (str): Label for the y-axis (name of the metric).
    - plot_dir (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    - metric (str): Name of the metric (used for naming the saved file).
    
    Functionality:
    - Plots the given metric curve.
    - Saves the plot as a JPEG file in the specified directory.
    '''
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.title(f"{label} over Epochs - {args.exp_id}")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(plot_dir, f"{metric}_{args.exp_id}.jpg"))
    plt.close()