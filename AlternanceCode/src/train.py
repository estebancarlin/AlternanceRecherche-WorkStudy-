import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model import UNet
from src.modelmoredepth import UNet as UNetDeep
from src.data import create_generator
from src.loss import IoULoss, IoUClassesLoss
from src.metrics import compute_metrics
from src.utils import train_logger, train_step_logger, save_learning_curves

torch.manual_seed(0)

CLASS_DISTRIBUTION = [0.9621471811176255, 0.012111862189784502, 0.013016226246835367, 0.01272473044575458]


def get_n_params(model):
    """Return the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(config):
    """Train a U-Net model on the given dataset using the specified configuration."""
    # Create dataset and loaders
    dataset_train, dataset_val, _ = create_generator(config)
    train_loader = DataLoader(dataset_train, batch_size=config.train.batch_size)
    val_loader = DataLoader(dataset_val, batch_size=config.val.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    # Initialize model
    model = UNet(
        input_channels=config.data.number_of_channels,
        output_classes=config.data.number_classes,
        hidden_channels=config.model.hidden_channels,
        dropout_probability=config.model.dropout
    ).to(device)

    print('Model parameters:', get_n_params(model))

    # Select loss function
    if "crossentropy" in config.model.loss.lower():
        if "weight" in config.model.loss.lower():
            class_weights = torch.tensor([1 / x for x in CLASS_DISTRIBUTION], dtype=torch.float).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            print('Loss: CrossEntropy (weighted)')
        else:
            criterion = torch.nn.CrossEntropyLoss()
            print('Loss: CrossEntropy')
    elif "iou" in config.model.loss.lower():
        if "avg" in config.model.loss.lower():
            criterion = IoUClassesLoss(nb_classes=config.data.number_classes)
            print('Loss: IoU (per class)')
        else:
            criterion = IoULoss()
            print('Loss: IoU')

    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)

    # Prepare logging
    metrics_names = [name for name, enabled in config.metrics.items() if enabled]
    logging_path = train_logger(config, metrics_names)

    best_epoch = 0
    best_val_loss = float('inf')

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(1, config.train.epochs + 1):
        print(f'\nEpoch {epoch}/{config.train.epochs}')
        model.train()

        epoch_train_loss = []
        train_metrics_sum = np.zeros(len(metrics_names))
        virtual_batch_counter = 0

        train_iter = tqdm(train_loader, desc='Training', leave=False)
        for images, targets in train_iter:
            images = images.to(device)
            targets = targets.to(device)
            targets = torch.movedim(targets, 4, 1)

            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()

            epoch_train_loss.append(loss.item())
            train_metrics_sum += compute_metrics(config, targets, predictions, argmax_axis=-1)

            if virtual_batch_counter % config.train.virtual_batch_size == 0 or virtual_batch_counter + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            virtual_batch_counter += 1

            train_iter.set_description(f"TRAIN | Epoch {epoch} | Loss: {np.mean(epoch_train_loss):.4f}")

        train_metrics_avg = train_metrics_sum / len(train_loader)

        # ---------------------------
        # Validation Loop
        # ---------------------------
        model.eval()
        epoch_val_loss = []
        val_metrics_sum = np.zeros(len(metrics_names))

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation', leave=False):
                images = images.to(device)
                targets = targets.to(device)
                targets = torch.movedim(targets, 4, 1)

                predictions = model(images)
                loss = criterion(predictions, targets)

                epoch_val_loss.append(loss.item())
                val_metrics_sum += compute_metrics(config, targets, predictions, argmax_axis=-1)

        val_loss_avg = np.mean(epoch_val_loss)
        val_metrics_avg = val_metrics_sum / len(val_loader)

        # ---------------------------
        # Logging & Checkpointing
        # ---------------------------
        train_step_logger(logging_path, epoch, np.mean(epoch_train_loss), val_loss_avg, train_metrics_avg, val_metrics_avg)

        if val_loss_avg < best_val_loss:
            print("Saving new best checkpoint...")
            best_val_loss = val_loss_avg
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(logging_path, 'model.pth'))

    # Rename best model
    best_model_path = os.path.join(logging_path, f'model{best_epoch}.pth')
    os.rename(os.path.join(logging_path, 'model.pth'), best_model_path)

    # Save loss/metric curves
    save_learning_curves(logging_path)