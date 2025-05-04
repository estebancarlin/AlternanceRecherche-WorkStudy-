import os
import re
import time
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from datetime import datetime
from easydict import EasyDict
from glob import glob
from medpy.metric.binary import dc


# --------------------------- Z-SCORE NORMALIZATION ---------------------------

def z_score_normalisation_on_files(img):
    data = img.get_fdata()
    mean, std = np.mean(data), np.std(data)
    normalized = (data - mean) / std
    return nib.Nifti1Image(normalized, img.affine, img.header)


def z_score_normalisation_on_directories(path):
    img = nib.load(os.path.join(path, 'example.nii.gz'))
    return z_score_normalisation_on_files(img)


def z_score_normalization_on_tensor(tensor):
    return (tensor - torch.mean(tensor)) / torch.std(tensor)


def file_or_dir(path_or_dir):
    if os.path.isfile(path_or_dir):
        return z_score_normalisation_on_files(nib.load(path_or_dir))
    elif os.path.isdir(path_or_dir):
        return z_score_normalisation_on_directories(path_or_dir)
    else:
        raise ValueError("Input must be a file or directory.")


# --------------------------- TRAINING LOGGING ---------------------------

def number_folder(path, prefix):
    existing = [d for d in os.listdir(path) if d.startswith(prefix)]
    indices = [int(d.replace(prefix, '')) for d in existing if d.replace(prefix, '').isdigit()]
    return prefix + str(max(indices + [-1]) + 1)


def train_logger(config, metrics_name):
    path = config.train.logs_path
    folder_name = number_folder(path, 'experiment_')
    log_path = os.path.join(path, folder_name)
    os.makedirs(log_path)

    print(f"Logging to: {log_path}")

    with open(os.path.join(log_path, 'train_log.csv'), 'w') as f:
        header = ['step', config.model.loss, f'val {config.model.loss}']
        for metric in metrics_name:
            header += [metric, f'val {metric}']
        f.write(','.join(header) + '\n')

    with open(os.path.join(log_path, 'config.yaml'), 'w') as f:
        now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        f.write(f"config_metadata: 'Saving time : {now}'\n")
        f.writelines('\n'.join(config_to_yaml(config)) + '\n')

    return log_path


def config_to_yaml(config, space=''):
    lines = []
    for key, value in config.items():
        if isinstance(value, EasyDict):
            lines.append('')
            lines.append(f"# {key} options")
            lines.append(f"{key}:")
            lines.extend(config_to_yaml(value, space + '  '))
        else:
            val_str = "null" if value is None else str(value).lower() if isinstance(value, bool) else repr(value)
            lines.append(f"{space}{key}: {val_str}")
    return lines


def train_step_logger(path, epoch, train_loss, val_loss, train_metrics, val_metrics):
    with open(os.path.join(path, 'train_log.csv'), 'a') as f:
        row = [str(epoch), f"{train_loss:.6f}", f"{val_loss:.6f}"]
        row += [f"{v:.6f}" for pair in zip(train_metrics, val_metrics) for v in pair]
        f.write(','.join(row) + '\n')


def test_logger(path, metrics, values):
    with open(os.path.join(path, 'test_log.txt'), 'w') as f:
        for name, value in zip(metrics, values):
            f.write(f"{name}: {value:.4f}\n")


# --------------------------- CURVE PLOTTING ---------------------------

def save_learning_curves(path):
    results, names = get_result(path)
    save_path = os.path.join(path, 'learning_curves')
    os.makedirs(save_path, exist_ok=True)
    loss_idx, acc_idx, iou_idx, ce_idx = make_groups(names)

    plot_curves(results, names, save_path, names[1], loss_idx)
    plot_curves(results, names, save_path, 'Accuracy', acc_idx)
    plot_curves(results, names, save_path, 'IoU', iou_idx)
    plot_curves(results, names, save_path, 'CrossEntropy', ce_idx)


def get_result(path):
    with open(os.path.join(path, 'train_log.csv'), 'r') as f:
        names = f.readline().strip().split(',')
        data = [line.strip().split(',') for line in f]
    return np.array(data, dtype=float), names


def make_groups(names):
    idx = range(len(names))
    return (
        [i for i in idx if 'loss' in names[i]],
        [i for i in idx if 'acc' in names[i] and 'loss' not in names[i]],
        [i for i in idx if 'iou' in names[i] and 'loss' not in names[i]],
        [i for i in idx if 'entropy' in names[i] and 'loss' not in names[i]]
    )


def plot_curves(data, names, save_path, title, indices):
    if not indices:
        return
    epochs = data[:, 0]
    plt.figure()
    for i in indices:
        plt.plot(epochs, data[:, i], label=names[i])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f"{title}.png"))
    plt.close()


# --------------------------- METRICS + I/O ---------------------------

def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def save_nii(path, data, affine, header):
    nib.save(nib.Nifti1Image(data, affine, header), path)


def metrics(gt, pred, voxel_size):
    if gt.shape != pred.shape:
        raise ValueError("Shape mismatch between ground truth and prediction.")

    res = []
    for c in [3, 1, 2]:  # LV, RV, Myo
        gt_bin = (gt == c).astype(np.uint8)
        pred_bin = (pred == c).astype(np.uint8)

        dice = dc(gt_bin, pred_bin)
        vol_gt = np.sum(gt_bin) * np.prod(voxel_size) / 1000
        vol_pred = np.sum(pred_bin) * np.prod(voxel_size) / 1000

        res.extend([dice, vol_pred, vol_pred - vol_gt])
    return res


def compute_metrics_on_files(path_gt, path_pred):
    gt, _, header = load_nii(path_gt)
    pred, _, _ = load_nii(path_pred)
    metrics_vals = metrics(gt, pred, header.get_zooms())
    name = os.path.splitext(os.path.basename(path_gt))[0]
    print(f"{'Name':>14}, {'Dice LV':>7}, {'Vol LV':>9}, {'Err LV':>10}, "
          f"{'Dice RV':>7}, {'Vol RV':>9}, {'Err RV':>10}, "
          f"{'Dice Myo':>8}, {'Vol Myo':>10}, {'Err Myo':>11}")
    print(f"{name:>14}, " + ', '.join(f"{v:.3f}" for v in metrics_vals))


def compute_metrics_on_directories(dir_gt, dir_pred):
    files_gt = sorted(glob(os.path.join(dir_gt, '*')), key=natural_order)
    files_pred = sorted(glob(os.path.join(dir_pred, '*')), key=natural_order)

    results = []
    for f_gt, f_pred in zip(files_gt, files_pred):
        if os.path.basename(f_gt) != os.path.basename(f_pred):
            raise ValueError(f"Mismatched file names: {f_gt}, {f_pred}")
        gt, _, header = load_nii(f_gt)
        pred, _, _ = load_nii(f_pred)
        results.append(metrics(gt, pred, header.get_zooms()))

    names = [os.path.splitext(os.path.basename(f))[0] for f in files_gt]
    df = pd.DataFrame([[n] + r for n, r in zip(names, results)],
                      columns=["Name", "Dice LV", "Vol LV", "Err LV",
                               "Dice RV", "Vol RV", "Err RV",
                               "Dice Myo", "Vol Myo", "Err Myo"])
    df.to_csv(f"results_{time.strftime('%Y%m%d_%H%M%S')}.csv", index=False)


# --------------------------- STRING SORTING ---------------------------

def conv_int(s):
    return int(s) if s.isdigit() else s


def natural_order(s):
    return [conv_int(c) for c in re.split(r'(\d+)', s[0] if isinstance(s, tuple) else s)]


# --------------------------- MAIN ---------------------------

def main(path_gt, path_pred):
    if os.path.isfile(path_gt) and os.path.isfile(path_pred):
        compute_metrics_on_files(path_gt, path_pred)
    elif os.path.isdir(path_gt) and os.path.isdir(path_pred):
        compute_metrics_on_directories(path_gt, path_pred)
    else:
        raise ValueError("Input paths must both be files or both be directories.")