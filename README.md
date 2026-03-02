# Interpretable Probabilistic Medical Image Segmentation via Gaussian Process with Explicit Modelling of Annotation Bias and Variability

This repository is the official implementation for "Interpretable Probabilistic Medical Image Segmentation via Gaussian Process with Explicit Modelling of Annotation Bias and Variability". It introduces a logit-space probabilistic segmentation framework with explicit modelling of inter- and intra-rater variability. Using a stochastic variational Gaussian Process, the model yields closed-form rater-conditioned predictive probabilities. This enables systematic examination of how annotator bias and variability affect calibration and segmentation performance, which is demonstrated on real-word clinical data.


## Install conda environment
``` bash
conda create -n GPS python=3.9.13
conda activate GPS
pip install -r requirements.txt
``` 

## Data structure
The following shows the data structure requirement if you want to train the model on your own data.
``` bash
    data/
    │
    ├── images/
        ├── VOLXX.nii.gz  # US scan, including a number of US slices 
        ├── ... 
    ├── seg_01/  # Annotation of rater 1
        ├── label_VOLXX.nii.gz  # Annotation1 of the corresponding US scan
        ├── ... 
    ├── seg_02/  # Annotation of rater 2
        ├── label_VOLXX.nii.gz  # Annotation2 of the corresponding US scan
        ├── ... 
    ├── seg_03/  # Annotation of rater 3
        ├── label_VOLXX.nii.gz  # Annotation3 of the corresponding US scan
        ├── ... 
    ├── seg_MV/  # Majority vote annotation
        ├── label_VOLXX.nii.gz  # Majority vote annotation of the corresponding US scan
        ├── ... 
    ├── seg_HQ/  # High quality annotation
        ├── label_VOLXX.nii.gz  # High quality annotation of the corresponding US scan
        ├── ... 

```

## Train a SVGP model
``` bash
python3 main.py
``` 

## Test
``` bash
python3 main_test.py
``` 
## Query performance at various bias and variance
``` bash
python3 performance_estimation.py
``` 
