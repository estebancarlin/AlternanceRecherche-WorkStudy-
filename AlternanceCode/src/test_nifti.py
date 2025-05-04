import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from src.params import TRAINING_DATA_PATH

# -------------------------------
# Load sample IRM and ground truth images
# -------------------------------
image_path = os.path.join(TRAINING_DATA_PATH, 'patient002', 'patient002_frame01.nii.gz')
gt_path = os.path.join(TRAINING_DATA_PATH, 'patient002', 'patient002_frame01_gt.nii.gz')

image_nii = nib.load(image_path)
gt_nii = nib.load(gt_path)

image_data = image_nii.get_fdata()
gt_data = gt_nii.get_fdata()

# -------------------------------
# Display 8 consecutive slices from both volumes
# -------------------------------
plt.figure(figsize=(12, 8))
for i in range(8):
    # Original IRM slice
    plt.subplot(4, 4, i + 1)
    plt.imshow(image_data[:, :, 2 + i], cmap='gray')
    plt.title(f'IRM Slice {2 + i}')
    plt.axis('off')

    # Ground truth mask slice
    plt.subplot(4, 4, i + 9)
    plt.imshow(gt_data[:, :, 2 + i], cmap='jet', vmin=0, vmax=np.max(gt_data))
    plt.title(f'GT Slice {2 + i}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# -------------------------------
# Compare headers of both images
# -------------------------------
image_header = image_nii.header
gt_header = gt_nii.header

print("Image Header - sform_code:", image_header['sform_code'])
print("GT Header    - sform_code:", gt_header['sform_code'])

print("Image Header Affine Matrix:")
print(image_nii.affine)
print("GT Header Affine Matrix:")
print(gt_nii.affine)

print("Image Header sform:", image_header.get_sform())
print("Image Header qform:", image_header.get_qform())
print("Image Header base_affine:", image_header.get_base_affine())

print("GT Header sform:", gt_header.get_sform())
print("GT Header qform:", gt_header.get_qform())
print("GT Header base_affine:", gt_header.get_base_affine())

# -------------------------------
# Check if headers are identical
# -------------------------------
print("\n--- Header Comparison ---")
for key in image_header.keys():
    if key in gt_header:
        img_val = image_header[key]
        gt_val = gt_header[key]
        if isinstance(img_val, np.ndarray):
            if np.array_equal(img_val, gt_val):
                print(f"[=] '{key}' is equal")
            else:
                print(f"[≠] '{key}' differs")
        else:
            if img_val == gt_val:
                print(f"[=] '{key}' is equal")
            else:
                print(f"[≠] '{key}' differs")
    else:
        print(f"[!] '{key}' not found in ground truth header")

# -------------------------------
# Quick inspection of directory structure
# -------------------------------
DATA_PATH = TRAINING_DATA_PATH  # or specify explicitly
print("\n--- Directory Structure ---")
for patient in os.listdir(DATA_PATH):
    if patient in {'.DS_Store', 'MANDATORY_CITATION.md'}:
        continue
    patient_dir = os.path.join(DATA_PATH, patient)
    for file in os.listdir(patient_dir):
        if file not in {'MANDATORY_CITATION.md', 'Info.cfg', '.DS_Store'}:
            print(f"Patient: {patient} | File: {file}")