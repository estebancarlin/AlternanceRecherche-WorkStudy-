import os
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import load_nii


class ACDCGenerator(Dataset):
    """PyTorch Dataset for the ACDC segmentation task."""

    def __init__(self, config, list_IDs):
        self.classes = os.listdir(config.data.data_path)
        self.list_IDs = list_IDs
        self.n_channels = config.data.number_of_channels
        self.shuffle = config.data.shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.data_path = config.data.data_path
        self.deep_unet = config.model.depth_unet
        self.number_classes = config.data.number_classes
        self.on_epoch_end()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        X, y = self.__data_generation(self.list_IDs[self.indexes[index]])
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        path_img, path_gt = list_IDs_temp
        image_size = np.shape(load_nii(path_img)[0])

        X_raw = resize_for_unet(load_nii(path_img)[0], image_size, self.deep_unet)
        y_raw = resize_for_unet(load_nii(path_gt)[0], image_size, self.deep_unet)

        aligned_size = (
            image_size[0] - image_size[0] % (2 ** self.deep_unet),
            image_size[1] - image_size[1] % (2 ** self.deep_unet),
            image_size[2]
        )

        X = np.empty((self.n_channels, *aligned_size), dtype=np.float32)
        y = np.empty(aligned_size, dtype=np.int64)

        X[0] = X_raw
        y = y_raw

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.int64)
        y_tensor = torch.nn.functional.one_hot(y_tensor, num_classes=self.number_classes).permute(2, 0, 1)
        return X_tensor, y_tensor.float()


def get_data(data_path):
    """Load and match image/label pairs from structured ACDC folder."""
    image_paths = []

    for patient in os.listdir(data_path):
        if patient in ['.DS_Store', 'MANDATORY_CITATION.md'] or patient.endswith(('x', 'y')):
            continue

        frames = [None] * 4  # [frame1_img, frame1_label, frame2_img, frame2_label]
        patient_path = os.path.join(data_path, patient)

        for filename in os.listdir(patient_path):
            full_path = os.path.join(patient_path, filename)
            if filename in ['MANDATORY_CITATION.md', 'Info.cfg', '.DS_Store']:
                continue
            if len(filename) == 25 and filename[17] == '1':
                frames[0] = full_path
            elif len(filename) == 28 and filename[17] == '1':
                frames[1] = full_path
            elif len(filename) == 25 and filename[17] == '2':
                frames[2] = full_path
            elif len(filename) == 28 and filename[17] == '2':
                frames[3] = full_path

        if all(f is not None for f in frames):
            image_paths.append(frames)

    return image_paths


def data_split(config, paths_list):
    """Split data into training, validation, and testing sets."""
    n = len(paths_list)
    split_1 = int(n * config.data.train_split)
    split_2 = split_1 + int(n * config.data.val_split)
    return paths_list[:split_1], paths_list[split_1:split_2], paths_list[split_2:]


def reshape(paired_list):
    """Convert list of (n, 4) to (2n, 2), pairing images and labels."""
    reshaped = []
    for entry in paired_list:
        reshaped.append(entry[:2])
        reshaped.append(entry[2:])
    return np.array(reshaped)


def resize_for_unet(volume, image_size, deep_unet):
    """Trim spatial dimensions to be divisible by 2^depth."""
    resize_x = image_size[0] % (2 ** deep_unet)
    resize_y = image_size[1] % (2 ** deep_unet)

    x_left = resize_x - (resize_x // 2)
    x_right = resize_x // 2
    y_top = resize_y // 2
    y_bottom = resize_y - y_top

    if resize_x == 0 and resize_y == 0:
        return volume

    return volume[
        x_right:image_size[0] - x_left,
        y_top:image_size[1] - y_bottom,
        :
    ]


def create_generator(config):
    """Create PyTorch dataset generators for training, validation, and testing."""
    all_image_paths = get_data(config.data.data_path)

    train_list, val_list, test_list = data_split(config, np.asarray(all_image_paths))
    train_list = reshape(train_list)
    val_list = reshape(val_list)
    test_list = reshape(test_list)

    train_generator = ACDCGenerator(config, train_list)
    val_generator = ACDCGenerator(config, val_list)
    test_generator = ACDCGenerator(config, test_list)

    return train_generator, val_generator, test_generator