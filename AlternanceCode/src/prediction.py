import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from src.model import UNet
from src.utils import load_nii
from src.data import resize_for_unet


def prediction(config, weights_path, irm_path):
    """
    Perform inference on a 3D IRM volume using a trained U-Net model and visualize slice-wise predictions.

    Args:
        config: Configuration object containing model and data settings.
        weights_path (str): Path to the saved model weights (.pth).
        irm_path (str): Path to the input IRM .nii.gz file.
    """
    # Load and initialize the model
    model = UNet(
        input_channels=config.data.number_of_channels,
        output_classes=config.data.number_classes,
        hidden_channels=config.model.hidden_channels,
        dropout_probability=config.model.dropout
    )
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Load and resize IRM image
    irm_data, irm_affine, irm_header = load_nii(irm_path)
    image_size = irm_data.shape

    irm_resized = resize_for_unet(irm_data, image_size, config.model.depth_unet)
    aligned_size = (
        image_size[0] - image_size[0] % (2 ** config.model.depth_unet),
        image_size[1] - image_size[1] % (2 ** config.model.depth_unet),
        image_size[2]
    )

    # Prepare input tensor
    X = np.empty((config.data.number_of_channels, *aligned_size), dtype=np.float32)
    X[0] = irm_resized
    X = torch.tensor(X).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        predictions = model(X)

    # Load ground truth
    directory, filename = os.path.split(irm_path)
    gt_path = os.path.join(directory, filename.replace(".nii.gz", "_gt.nii.gz"))
    gt_data, _, _ = load_nii(gt_path)

    # Sanity check
    print("IRM shape:", irm_data.shape)
    print("Prediction shape:", predictions.shape)
    print("Ground truth shape:", gt_data.shape)

    # Set up colormap and labels
    class_labels = ['Autre', 'RV', 'Myo', 'LV']
    cmap = plt.get_cmap('jet', len(class_labels))

    # Process each slice
    prediction_map = predictions[0].argmax(dim=0).cpu().numpy()

    for slice_idx in range(irm_data.shape[2]):
        plt.figure(figsize=(18, 6))

        # Original IRM
        plt.subplot(1, 3, 1)
        plt.imshow(irm_data[:, :, slice_idx], cmap='gray', origin='lower')
        plt.title(f'Original IRM - Slice {slice_idx}')

        # Prediction
        plt.subplot(1, 3, 2)
        plt.imshow(prediction_map[:, :, slice_idx], cmap=cmap, origin='lower',
                   vmin=0, vmax=len(class_labels) - 1, interpolation='none')
        plt.title(f'Segmentation Result - Slice {slice_idx}')

        # Ground Truth
        plt.subplot(1, 3, 3)
        plt.imshow(gt_data[:, :, slice_idx], cmap=cmap, origin='lower',
                   vmin=0, vmax=len(class_labels) - 1, interpolation='none')
        plt.title(f'Ground Truth - Slice {slice_idx}')

        # Shared colorbar for all subplots
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(class_labels) - 1)),
            ax=plt.gcf().get_axes(), ticks=range(len(class_labels))
        )
        cbar.set_ticklabels(class_labels)

        plt.tight_layout()
        plt.show()