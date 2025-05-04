"""
author: Clément Zotti (clement.zotti@usherbrooke.ca)
date: April 2017

DESCRIPTION :
The script provide helpers functions to handle nifti image format:
    - load_nii()
    - save_nii()

to generate metrics for two images:
    - metrics()

And it is callable from the command line (see below).
Each function provided in this script has comments to understand
how they works.

HOW-TO:

This script was tested for python 3.4.

First, you need to install the required packages with
    pip install -r requirements.txt

After the installation, you have two ways of running this script:
    1) python metrics.py ground_truth/patient001_ED.nii.gz prediction/patient001_ED.nii.gz
    2) python metrics.py ground_truth/ prediction/

The first option will print in the console the dice and volume of each class for the given image.
The second option wiil ouput a csv file where each images will have the dice and volume of each class.


Link: http://acdc.creatis.insa-lyon.fr

"""

import os
from glob import glob
import time
import re
import argparse
import nibabel as nib
import pandas as pd
from medpy.metric.binary import hd, dc
import numpy as np
import statistics
from scipy.ndimage import zoom
from src.params import TRAINING_DATA_PATH, DATA_PATH, COMMON_SIZE

HEADER = ["Name", "Dice LV", "Volume LV", "Err LV(ml)",
          "Dice RV", "Volume RV", "Err RV(ml)",
          "Dice MYO", "Volume MYO", "Err MYO(ml)"]


#
# Utils functions used to sort strings into a natural order
#
def conv_int(i):
    return int(i) if i.isdigit() else i


def natural_order(sord):
    """
    Sort a (list,tuple) of strings into natural order.

    Ex:

    ['1','10','2'] -> ['1','2','10']

    ['abc1def','ab10d','b2c','ab1d'] -> ['ab1d','ab10d', 'abc1def', 'b2c']

    """
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


#
# Utils function to load and save nifti files with the nibabel package
#
def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everything needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    # print(nimg.shape)
    return nimg, nimg.get_fdata(), nimg.affine, nimg.header


def verif_shape():
    # Locate the appropriate folder
    folder_path = DATA_PATH

    min_00 = 1000
    max_00 = 0
    min_01 = 1000
    max_01 = 0
    min_02 = 1000
    max_02 = 0

    # Collect the list of patient files
    file_list = []
    for i in range(1, 151):
        file_name = 'patient{:03d}'.format(i)
        file_path = os.path.join(folder_path, 'training', file_name)
        for j in range(2, 31):
            ed_file = '{}_frame01.nii.gz'.format(file_name)
            es_file = '{}_frame{:02d}.nii.gz'.format(file_name, j)
            gtd_file = '{}_frame01_gt.nii.gz'.format(file_name, j)
            gts_file = '{}_frame{:02d}_gt.nii.gz'.format(file_name, j)
            ed_path = os.path.join(file_path, ed_file)
            es_path = os.path.join(file_path, es_file)
            gtd_path = os.path.join(file_path, gtd_file)
            gts_path = os.path.join(file_path, gts_file)
            # Check if files exist before appending to list
            if os.path.exists(ed_path) and os.path.exists(es_path) and os.path.exists(gtd_path) and os.path.exists(gts_path):
                file_list.append(ed_path)
                file_list.append(es_path)
                file_list.append(gtd_path)
                file_list.append(gts_path)
    liste = []
    for k in range(len(file_list)):
        nii, truc, machin, bidule = load_nii(file_list[k])
        liste.append([nii.shape[0], nii.shape[1], nii.shape[2]])
        # print(nii.shape)
        # print([nii.shape[0], nii.shape[1], nii.shape[2]])
        if nii.shape[0] <= min_00:
            min_00 = nii.shape[0]
        if nii.shape[0] >= max_00:
            max_00 = nii.shape[0]
            # print('Max_00_intermediaire :', max_00)
        if nii.shape[1] <= min_01:
            min_01 = nii.shape[1]
        if nii.shape[1] >= max_01:
            max_01 = nii.shape[1]
            # print('Max_01_intermediaire :', max_01)
        if nii.shape[2] <= min_02:
            min_02 = nii.shape[2]
        if nii.shape[2] >= max_02:
            max_02 = nii.shape[2]
            # print('Max_02_intermediaire :', max_02)

    print('Minimums :', min_00, min_01, min_02)
    print('Maximums :', max_00, max_01, max_02)
    # print('Liste de valeurs:', liste)
    return liste

#
# L = verif_shape()
# L00 = []
# L01 = []
# L02 = []
# for k in range(len(L)):
#     L00.append(L[k][0])
#     L01.append(L[k][1])
#     L02.append(L[k][2])
# print('L00 :', L00)
# print('Médiane L00 :', statistics.median(L00))
# # print('Moyenne L00 :', sum(L00)/len(L00))
# # print('Moyenne de la moyenne et de la médiane L00:', (statistics.median(L00)+sum(L00)/len(L00))/2)
# # print('Entier du truc précédent L00 :', int((statistics.median(L00)+sum(L00)/len(L00))/2))
# print('L01 :', L01)
# print('Médiane L01 :', statistics.median(L01))
# # print('Moyenne L01 :', sum(L01)/len(L01))
# # print('Moyenne de la moyenne et de la médiane L01:', (statistics.median(L01)+sum(L01)/len(L01))/2)
# # print('Entier du truc précédent L01 :', int((statistics.median(L01)+sum(L01)/len(L01))/2))
# print('L02 :', L02)
# print('Médiane L02 :', statistics.median(L02))
# # print('Moyenne L02 :', sum(L02)/len(L02))
# # print('Moyenne de la moyenne et de la médiane L02:', (statistics.median(L02)+sum(L02)/len(L02))/2)
# # print('Entier du truc précédent L02 :', int((statistics.median(L02)+sum(L02)/len(L02))/2))
#
# print('Coordonnées médianes :', (statistics.median(L00), statistics.median(L01), statistics.median(L02)))


def reshape_nii(array):
    # Get the current shape of the array
    curr_shape = array.shape

    # Calculate the difference between the current and desired shapes
    shape_diff = np.array(COMMON_SIZE) - np.array(curr_shape)

    # Calculate the padding needed on each side of the array
    pad_before = shape_diff // 2
    pad_after = shape_diff - pad_before

    # Pad the array with zeros to the desired shape
    pad_width = [(pad_before[i], pad_after[i]) for i in range(len(COMMON_SIZE))]
    arr_padded = np.pad(array, pad_width, mode='constant')

    # Crop the array to the desired shape if it's larger than the desired shape
    crop_slices = tuple(slice(pad_before[i], pad_before[i] + COMMON_SIZE[i]) for i in range(len(COMMON_SIZE)))
    arr_cropped = arr_padded[crop_slices]

    return arr_cropped


def save_nii(img_path, data, affine, header):
    """
    Function to save a 'nii' or 'nii.gz' file.

    Parameters
    ----------

    img_path: string
    Path to save the image should be ending with '.nii' or '.nii.gz'.

    data: np.array
    Numpy array of the image data.

    affine: list of list or np.array
    The affine transformation to save with the image.

    header: nib.Nifti1Header
    The header that define everything about the data
    (pleasecheck nibabel documentation).
    """
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


#
# Functions to process files, directories and metrics
#
def metrics(img_gt, img_pred, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    for c in [3, 1, 2]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)

        # Compute volume
        volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
        volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice, volpred, volpred - volgt]

    return res


def compute_metrics_on_files(path_gt, path_pred):
    """
    Function to give the metrics for two files

    Parameters
    ----------

    path_gt: string
    Path of the ground truth image.

    path_pred: string
    Path of the predicted image.
    """
    gt, _, header = load_nii(path_gt)
    pred, _, _ = load_nii(path_pred)
    zooms = header.get_zooms()

    name = os.path.basename(path_gt)
    name = name.split('.')[0]
    res = metrics(gt, pred, zooms)
    res = ["{:.3f}".format(r) for r in res]

    formatting = "{:>14}, {:>7}, {:>9}, {:>10}, {:>7}, {:>9}, {:>10}, {:>8}, {:>10}, {:>11}"
    print(formatting.format(*HEADER))
    print(formatting.format(name, *res))


def compute_metrics_on_directories(dir_gt, dir_pred):
    """
    Function to generate a csv file for each images of two directories.

    Parameters
    ----------

    path_gt: string
    Directory of the ground truth segmentation maps.

    path_pred: string
    Directory of the predicted segmentation maps.
    """
    lst_gt = sorted(glob(os.path.join(dir_gt, '*')), key=natural_order)
    lst_pred = sorted(glob(os.path.join(dir_pred, '*')), key=natural_order)

    res = []
    for p_gt, p_pred in zip(lst_gt, lst_pred):
        if os.path.basename(p_gt) != os.path.basename(p_pred):
            raise ValueError("The two files don't have the same name"
                             " {}, {}.".format(os.path.basename(p_gt),
                                               os.path.basename(p_pred)))

        gt, _, header = load_nii(p_gt)
        pred, _, _ = load_nii(p_pred)
        zooms = header.get_zooms()
        res.append(metrics(gt, pred, zooms))

    lst_name_gt = [os.path.basename(gt).split(".")[0] for gt in lst_gt]
    res = [[n, ] + r for r, n in zip(res, lst_name_gt)]
    df = pd.DataFrame(res, columns=HEADER)
    df.to_csv("results_{}.csv".format(time.strftime("%Y%m%d_%H%M%S")), index=False)


def metrics_file_or_dir(path_gt, path_pred):
    """
    Main function to select which method to apply on the input parameters.
    """
    if os.path.isfile(path_gt) and os.path.isfile(path_pred):
        compute_metrics_on_files(path_gt, path_pred)
    elif os.path.isdir(path_gt) and os.path.isdir(path_pred):
        compute_metrics_on_directories(path_gt, path_pred)
    else:
        raise ValueError(
            "The paths given needs to be two directories or two files.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Script to compute ACDC challenge metrics.")
#     parser.add_argument("GT_IMG", type=str, help="Ground Truth image")
#     parser.add_argument("PRED_IMG", type=str, help="Predicted image")
#     args = parser.parse_args()
#     main(args.GT_IMG, args.PRED_IMG)
