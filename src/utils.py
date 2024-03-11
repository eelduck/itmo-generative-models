import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve


def inference_and_visualize(dataset, autoencoder, device, num_samples=10):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    
    for i in range(num_samples):
        # Get a sample from the dataset
        x, _ = dataset[i]
        x = x.to(device)  # Move the tensor to the device
        
        # Add batch dimension and perform inference
        x_reconstructed = autoencoder(x.unsqueeze(0)).squeeze(0).cpu().detach()  # Move the tensor back to CPU for visualization
        
        # Display the original and reconstructed images
        ax = axes[i, 0]
        ax.imshow(x.cpu().permute(1, 2, 0))  # Convert from CxHxW to HxWxC for matplotlib and move tensor to CPU
        ax.set_title('Original')
        ax.axis('off')
        
        ax = axes[i, 1]
        ax.imshow(x_reconstructed.permute(1, 2, 0))  # Convert from CxHxW to HxWxC for matplotlib
        ax.set_title('Reconstructed')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()


def calculate_individual_losses(dataset, autoencoder, device):
    autoencoder.eval()  # Ensure the model is in evaluation mode
    losses = []  # List to store the MSE for each image

    with torch.no_grad():  # No need to keep track of gradients
        for i in range(len(dataset)):
            # Get a sample from the dataset
            x, _ = dataset[i]
            x = x.to(device)  # Move the tensor to the device

            # Add batch dimension and perform inference
            x_reconstructed = autoencoder(x.unsqueeze(0))

            # Calculate the loss between the original and the reconstructed image
            loss = F.mse_loss(x_reconstructed, x.unsqueeze(0))

            # Append the individual loss to the list
            losses.append(loss.item())

    return losses


def find_best_threshold(probabilities_pos, probabilities_neg):
    # Combine the probabilities into one array and the true labels into another array
    probabilities = np.concatenate([probabilities_pos, probabilities_neg])
    labels = np.concatenate([np.ones(len(probabilities_pos)), np.zeros(len(probabilities_neg))])
    
    # Calculate the False Positive Rate, True Positive Rate, and thresholds
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    
    # Calculate the distances to the top-left corner for each threshold
    distances = np.sqrt((1-tpr)**2 + fpr**2)
    
    # Find the index of the smallest distance
    min_distance_idx = np.argmin(distances)
    
    # The best threshold is the one with the smallest distance to the top-left corner
    best_threshold = thresholds[min_distance_idx]
    
    return best_threshold


def calculate_tpr_tnr(true_labels, predicted_labels):
    """
    Calculate the True Positive Rate (TPR) and True Negative Rate (TNR).

    Parameters:
    - true_labels: array-like, true binary labels in range {0, 1} or {-1, 1}
    - predicted_labels: array-like, predicted binary labels in range {0, 1} or {-1, 1}

    Returns:
    - tpr: True Positive Rate
    - tnr: True Negative Rate
    """
    # Ensure inputs are numpy arrays for easy element-wise operations
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # True Positives (TP): We predicted positive, and it's true.
    TP = np.sum((predicted_labels == 1) & (true_labels == 1))

    # True Negatives (TN): We predicted negative, and it's true.
    TN = np.sum((predicted_labels == 0) & (true_labels == 0))

    # False Negatives (FN): We predicted negative, but it's false.
    FN = np.sum((predicted_labels == 0) & (true_labels == 1))

    # False Positives (FP): We predicted positive, but it's false.
    FP = np.sum((predicted_labels == 1) & (true_labels == 0))

    # True Positive Rate (TPR) = TP / (TP + FN)
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0

    # True Negative Rate (TNR) = TN / (TN + FP)
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0

    return {"TPR": tpr, "TNR": tnr}


def calculate_labels(losses, threshold):
    return [int(loss >= threshold) for loss in losses]
