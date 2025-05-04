import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model import UNet
from src.data import create_generator
from src.metrics import compute_metrics
from src.utils import test_logger
from src.loss import IoULoss, IoUClassesLoss

torch.manual_seed(0)


def test(logging_path, config):
    """
    Run evaluation on the test set.

    Args:
        logging_path (str): Directory where logs and model checkpoints are stored.
        config: Configuration object with data, model, and test settings.
    """
    # Load test set
    test_dataset = create_generator(config)[2]
    test_loader = DataLoader(test_dataset, batch_size=config.test.batch_size)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = UNet(
        input_channels=config.data.number_of_channels,
        output_classes=config.data.number_classes,
        hidden_channels=config.model.hidden_channels,
        dropout_probability=config.model.dropout
    ).to(device)

    checkpoint_path = get_checkpoint_path(config, logging_path)
    print("Checkpoint loaded from:", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Select loss function
    if "crossentropy" in config.model.loss.lower():
        criterion = torch.nn.CrossEntropyLoss()
        print("Using loss: CrossEntropy")
    elif "iou" in config.model.loss.lower():
        if "avg" in config.model.loss.lower():
            criterion = IoUClassesLoss(nb_classes=config.data.number_classes)
            print("Using loss: IoU (per class average)")
        else:
            criterion = IoULoss()
            print("Using loss: IoU")

    # Initialize metric tracking
    metric_names = [name for name in config.metrics if config.metrics[name]]
    test_loss_values = []
    test_metric_sums = np.zeros(len(metric_names), dtype=float)

    # Evaluation loop
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)
            targets = torch.movedim(targets, 4, 1)  # Move channels to dim=1

            predictions = model(images)

            loss = criterion(predictions, targets).item()
            test_loss_values.append(loss)

            metrics = compute_metrics(config, targets, predictions, argmax_axis=1)
            test_metric_sums += metrics

    avg_loss = np.mean(test_loss_values)
    avg_metrics = test_metric_sums / len(test_loader)

    print(f"Test Loss: {avg_loss:.4f}")
    test_logger(logging_path, metric_names, avg_metrics)


def get_checkpoint_path(config, base_path):
    """
    Resolve path to the correct checkpoint file.

    Args:
        config: Configuration object containing test.checkpoint.
        base_path (str): Base directory to search in.

    Returns:
        str: Full path to the selected .pth checkpoint file.
    """
    # List all .pth files in base directory
    pth_files = [f for f in os.listdir(base_path) if f.endswith(".pth")]
    if len(pth_files) == 1:
        return os.path.join(base_path, pth_files[0])

    # Check in `checkpoint_path` subdirectory if available
    checkpoint_dir = os.path.join(base_path, "checkpoint_path")
    if os.path.isdir(checkpoint_dir):
        if config.test.checkpoint in os.listdir(checkpoint_dir):
            return os.path.join(checkpoint_dir, config.test.checkpoint)

        if config.test.checkpoint == "last":
            all_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
            if all_checkpoints:
                last_idx = len(all_checkpoints) - 1
                return os.path.join(checkpoint_dir, f"model{last_idx}.pth")

        checkpoint_name = f"model{config.test.checkpoint}.pth"
        if checkpoint_name in os.listdir(checkpoint_dir):
            return os.path.join(checkpoint_dir, checkpoint_name)

    # Fallback: check for checkpoint in base_path
    checkpoint_name = f"model{config.test.checkpoint}.pth"
    if checkpoint_name in os.listdir(base_path):
        return os.path.join(base_path, checkpoint_name)

    raise FileNotFoundError("Checkpoint file could not be resolved.")