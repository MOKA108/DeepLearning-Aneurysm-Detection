# 🧠 Cerebral Aneurysm Detection with Deep Learning
## Course info

- **Institution:** DSTI School of Engineering
- **Program:** Applied MSc in Data Science & Artificial Intelligence  
- **Course:** Deep Learning 
- **Project:** Group assessment

---

## Project Overview
This project implements a full end-to-end deep-learning pipeline for the automatic detection of intracranial aneurysm using 3D Magnetic Resonance Angiography (MRA).
The goal is to support radiologists by identifying suspicious regions in volumetric scans, reducing workload and minimizing missed aneurysm, especially small or difficult-to-spot ones.  
The pipeline includes a comprehensive medical-imaging preprocessing workflow (DICOM → NIfTI, skull stripping, N4 bias-field correction, resampling), a patch-based 3D CNN and U-Net detection framework, and a complete scan-level inference system capable of generating 3D probability maps. 
A dedicated Streamlit application allows interactive visualizations.

Model pipeline overview:

![Model overview](schema/Model_pipeline_overview.PNG)

---

## Features

- Comprehensive Preprocessing Pipeline  
    - DICOM to NIfTI conversion with geometry preservation
    - Skull stripping using SynthStrip (state-of-the-art DL-based method)
    - N4 bias field correction to normalize MRI intensity
    - Intensity normalization (min–max)
    - Isotropic 1×1×1 mm resampling 
    - Automatic mapping of DICOM slice annotations into 3D voxel coordinates

- Patch-Based Deep Learning Framework
    - Efficient extraction of 3D patches for training
    - Configurable patch size, jittering, class balance
    - Positive patches centered on aneurysm coordinates 
    - Negative patches drawn from brain tissue only

- Neural Architectures Tested
    - A simple 3D CNN baseline
    - A Tiny 3D U-Net encoder–decoder (best-performing)

- Training Tools
    - Support for Binary Cross-Entropy and Focal Loss
    - Mixed-precision training (AMP)
    - Early stopping based on F2-score
    - Full set of evaluation metrics: accuracy, sensitivity, specificity, AUC, F2

- Whole-Scan Inference Engine
    - Sliding-window prediction with overlapping 3D patches
    - Reconstruction into a unified 3D probability heatmap
    - Threshold calibration based on validation data

- Streamlit Application
    - Run inference on preprocessed volumes
    - 2D slicer for axial / coronal / sagittal visualization
    - 3D viewer showing predicted vs. ground-truth aneurysm locations
    - Interactive probability maps and focal-region inspection

---

## How to install and run
1. Clone the Repository

```bash
git clone https://github.com/yourusername/aneurysm-detection.git
cd aneurysm-detection
```

2. Prepare the Dataset
- Download the raw dataset (≈200 GB) from RSNA sources (https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/overview)
- Place data paths according to the instructions in the repository’s root README

3. Run Preprocessing

```bash
python preprocessing/run_preprocessing.py

```

4. Train the Model
 ```bash
python model/train.py --config configs/model5.json
```

5. Run Scan-Level Inference
```bash
python app/inference.py
```

6. Launch the Streamlit Application
```bash
streamlit run app/main.py
```

7. Explore
Navigate through the 3D volume, inspect aneurysm predictions, visualize probability maps, and interact with the model’s output using intuitive tools designed for medical imaging.

---

## Overview of the tested architectures
Two architectures were tested: 

- A simple 3D CNN
A lightweight baseline composed of three convolutional blocks followed by global pooling and a fully connected classifier.

![3D CNN](schema/3DCNN.png)

- A Tiny 3D U-Net. 
A compact encoder–decoder architecture with skip connections.
It includes:
- Two convolutional blocks (encoder)
- A bottleneck block
- Two upsampling blocks (decoder)
- A global pooling layer for classification

![3D Tiny U-Net](schema/3D_U-Net.png)

---

## Results Summary

The Tiny 3D U-Net with Focal Loss achieved the best validation performance:

- Patch-level:
    - Accuracy: 0.91
    - Sensitivity: 0.94
    - Specificity: 0.89
    - AUC: 0.97
    - F2-score: 0.92

- Scan-level:
    - Accuracy: 0.62
    - Sensitivity: 0.68
    - Specificity: 0.58
    - AUC: 0.67

Although the model demonstrated strong discriminative power at the patch scale, whole-scan performance remains challenging—reflecting known limitations of patch-based inference in 3D medical imaging.

---

## Acknowledgements
This project was developed as part of the DSTI Deep Learning module.   

The dataset was sourced from the RSNA Intracranial Aneurysm Detection competition on Kaggle:
https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/overview  

The approach and methodology were partially inspired by:
Ham, S., Seo, J., Yun, J. et al. Automated detection of intracranial aneurysms using skeleton-based 3D patches, semantic segmentation, and auxiliary classification for overcoming data imbalance in brain TOF-MRA. Sci Rep 13, 12018 (2023). https://doi.org/10.1038/s41598-023-38586-9
