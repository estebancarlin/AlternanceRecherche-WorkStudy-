import torch
from torch import nn


def iou(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1e-3) -> torch.Tensor:
    """
    Compute the Intersection over Union (IoU) between prediction and ground truth.

    Args:
        y_true (torch.Tensor): Ground truth tensor.
        y_pred (torch.Tensor): Predicted tensor.
        smooth (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: IoU score.
    """
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true + y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


class IoULoss(nn.Module):
    """
    IoU Loss for binary segmentation tasks.
    """

    def __init__(self, smooth: float = 1e-3):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss.

        Args:
            y_pred (torch.Tensor): Predicted tensor of shape (N, 1, H, W).
            y_true (torch.Tensor): Ground truth tensor of the same shape.

        Returns:
            torch.Tensor: IoU loss value.
        """
        return 1.0 - iou(y_true, y_pred, self.smooth)


class IoUClassesLoss(nn.Module):
    """
    IoU Loss averaged over multiple classes (for multi-class segmentation).
    """

    def __init__(self, num_classes: int, smooth: float = 1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute average IoU loss over all classes.

        Args:
            y_pred (torch.Tensor): Predicted tensor of shape (N, C, H, W).
            y_true (torch.Tensor): Ground truth tensor of the same shape.

        Returns:
            torch.Tensor: Averaged IoU loss across all classes.
        """
        iou_per_class = [
            iou(y_true[:, c], y_pred[:, c], self.smooth)
            for c in range(self.num_classes)
        ]
        mean_iou = sum(iou_per_class) / self.num_classes
        return 1.0 - mean_iou