import torch
import numpy as np
from sklearn.metrics import jaccard_score

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class distribution and inverse weighting (e.g., for weighted loss)
CLASS_DISTRIBUTION = [0.9621471811176255, 0.012111862189784502, 0.013016226246835367, 0.01272473044575458]
CLASS_WEIGHTS = torch.tensor([1.0 / x for x in CLASS_DISTRIBUTION], dtype=torch.float32).to(device)


def compute_metrics(config, y_true: torch.Tensor, y_pred: torch.Tensor, argmax_axis: int = 1) -> np.ndarray:
    """
    Compute classification and segmentation metrics.

    Args:
        config: Configuration object with boolean flags for which metrics to compute.
        y_true (torch.Tensor): Ground truth tensor of shape (N, C, H, W) or (N, H, W) with one-hot encoding.
        y_pred (torch.Tensor): Predicted tensor of shape (N, C, H, W).
        argmax_axis (int): Axis over which to perform argmax for class prediction.

    Returns:
        np.ndarray: Computed metric values in the order defined by config flags.
    """
    results = []

    # Cross-entropy losses
    if config.metrics.crossentropy:
        ce_loss = torch.nn.CrossEntropyLoss()
        results.append(ce_loss(y_pred, y_true).item())

    if config.metrics.crossentropy_weighted:
        ce_loss_weighted = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
        results.append(ce_loss_weighted(y_pred, y_true).item())

    # Prepare predictions and labels for scikit-learn metrics
    # Move channels to last dim if one-hot encoded (N, C, H, W) -> (N, H, W, C)
    y_true = torch.movedim(y_true, 1, -1)
    y_pred = torch.movedim(y_pred, 1, -1)

    # Convert to class indices
    y_true_flat = torch.argmax(y_true, dim=-1).flatten().cpu()
    y_pred_flat = torch.argmax(y_pred, dim=-1).flatten().cpu()

    # Accuracy
    if config.metrics.accuracy:
        acc = (y_pred_flat == y_true_flat).float().mean().item()
        results.append(acc)

    # IoU metrics
    if config.metrics.iou:
        results.append(jaccard_score(y_true_flat, y_pred_flat, average='micro'))

    if config.metrics.iou_avg:
        results.append(jaccard_score(y_true_flat, y_pred_flat, average='macro'))

    if config.metrics.iou_weighted:
        results.append(jaccard_score(y_true_flat, y_pred_flat, average='weighted'))

    return np.array(results)