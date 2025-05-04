# AI-Assisted Cardiac Diagnosis via 3D U-Net Segmentation

**Author:** Esteban Carlin  
**Institution:** Institut Fresnel, Ecole Centrale Marseille (Centrale Méditerrannée)
**Internship Duration:** February 6, 2023 – January 31, 2024  
**Supervisor:** Mr. Salah Bourennane

## Project Overview

This project focuses on using artificial intelligence (AI), particularly **Deep Learning**, to assist in **diagnosing cardiac pathologies** from cardiac MRI (Magnetic Resonance Imaging) scans. It was developed during a research-oriented work-study program in collaboration with **Institut Fresnel**, Marseille.

The work consists of two major components:
1. **Segmentation** of heart structures from 3D MRI data using a U-Net model.
2. **Classification** of cardiac diseases based on extracted physiological features.

## Objective

To develop an AI pipeline capable of:
- Segmenting key heart structures: *left ventricle (LV)*, *right ventricle (RV)*, and *myocardium (MYO)* from 3D MRI data.
- Using these segmentations to extract clinical indicators such as volume and wall thickness.
- Classifying patients into five categories:  
  - NOR: Normal  
  - MINF: Myocardial infarction  
  - DCM: Dilated cardiomyopathy  
  - HCM: Hypertrophic cardiomyopathy  
  - AR: Right ventricle abnormality

## Dataset

The model was trained and evaluated on the **ACDC Challenge dataset** ([Automated Cardiac Diagnosis Challenge](https://acdc.creatis.insa-lyon.fr/)):
- **150 patients** (2 MRIs per patient: systole and diastole phases)
- Balanced classes: 30 patients per cardiac condition
- Manual expert annotations used as segmentation ground truth

## Methodology

### 1. Image Segmentation – 3D U-Net

A modified 3D U-Net architecture is implemented:

- **Encoder**: Downsampling blocks with convolution + ReLU + max pooling
- **Decoder**: Upsampling blocks with transposed convolutions and skip connections
- **Output**: 4-class voxel-wise prediction (LV, RV, MYO, background)

### 2. Feature Extraction

From the segmentation maps, the following features were extracted:
- Volume of LV and RV
- Myocardial wall thickness
- Volume change between systole and diastole
- Inter-ventricular volume ratio

### 3. Classification (to be completed)

These features will serve as input to a classification model (e.g., **VGG-Net**) to predict the associated cardiac pathology.

## Results

### Segmentation Metrics
- **Accuracy:** Approaches 100%
- **IoU:** Evaluated using micro, macro, and weighted averaging
- **Cross-Entropy Loss:** Shows clear convergence

> Visual comparisons between predictions and ground truth show high-quality segmentations across different patients and heart conditions.

![Data format overview](./images/data_format.png)
![Data format overview (1)](./images/data_format_bis.png)

---

## Poster & Report

- **Poster**: Summary for academic communication  
  → `POSTER_EstebanCARLIN.pdf`

- **Final Report**: Comprehensive technical document  
  → `RAPPORT_EstebanCARLIN.pdf`

- **Slides**: Internship oral defense  
  → `SLIDES_EstebanCARLIN.pdf`

---

## Remaining Work

- Complete and validate the classification pipeline
- Compare multiple classifiers (VGG, SVM, etc.)
- Optimize performance on small data samples

## References

> **If using or citing this project, please reference the original academic sources:**

- Bernard et al., *Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?*, 2018. DOI: [10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502)  
- Khened et al., *Fully convolutional multi-scale residual DenseNets for cardiac segmentation and automated cardiac diagnosis using ensemble of classifiers*, 2019. DOI: [10.1016/j.media.2018.10.004](https://doi.org/10.1016/j.media.2018.10.004)  
- Çiçek et al., *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation*, arXiv: [1606.06650](https://arxiv.org/abs/1606.06650)  
- Simonyan and Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, arXiv: [1409.1556](https://arxiv.org/abs/1409.1556)

See [`MANDATORY_CITATION.md`](./MANDATORY_CITATION.md) for complete citation requirements.

## Technologies Used

- Python (>=3.8)
- PyTorch
- NiBabel (for NIfTI MRI files)
- scikit-learn
- NumPy
- Matplotlib

## Folder Structure
```
Alternance/
├── AlternanceCode/                         # Source code and main Python scripts
│   ├── configs/                            # Configuration files (YAML/JSON)
│   ├── logs/                               # Training logs and metrics
│   ├── src/                                # All source code for data, model, training, evaluation
│   │   ├── __init__.py
│   │   ├── data.py                         # Dataset loading and preprocessing
│   │   ├── loss.py                         # Custom loss functions (e.g. IoU)
│   │   ├── metrics.py                      # Evaluation metrics like Dice, IoU
│   │   ├── model.py                        # UNet model definition
│   │   ├── modelmoredepth.py              # UNet with deeper architecture
│   │   ├── params.py                       # Hyperparameters and configurations
│   │   ├── prediction.py                   # Inference script
│   │   ├── test.py                         # Evaluation script on pre-defined test set
│   │   ├── test_nifti.py                   # Evaluation on 3D NIfTI images
│   │   ├── train.py                        # Training loop
│   │   ├── traitement.py                   # Data processing utilities (e.g. shape analysis)
│   │   └── utils.py                        # Logging, plotting, and general helpers
│   ├── venv/                               # Python virtual environment (excluded from version control)
│   ├── main.py                             # Main script to run the project
│   └── requirements.txt                    # Project dependencies
│
├── data_format.png                         # Illustration of the dataset structure
├── data_format_bis.png                     # Additional visualization of data format
├── MANDATORY_CITATION.md                   # Citation file required by the dataset license
├── POSTER_EstebanCARLIN.pdf                # Research poster summarizing the project
├── RAPPORT_EstebanCARLIN.pdf               # Final detailed research report
├── README.md                               # Project documentation (this file)
└── SLIDES_EstebanCARLIN.pdf                # Internship report presentation
```


## Acknowledgments

> "The most valuable part of this journey has been the opportunity to dive into research and AI for health applications. I am deeply thankful to my advisor Salah Bourennane for his guidance and to Institut Fresnel for hosting this enriching experience."

---

