# Influence of Label Bias and Variance on Model Performance: Gaussian Process Formulation

This repository is the official implementation for "Influence of Label Bias and Variance on Model Performance: Gaussian Process Formulation". It contains the algorithm of a Gaussian Process-based framework for the task of medical image segmentation, which parameterises intra-rater and inter-rater variability with bias and variance, enabling quantification of their correlations with model performance. By formulating model performance as a probabilistic function of intra-rater and inter-rater variability, the proposed approach allows for explicit modeling of how label variability propagate through the training process and affect prediction accuracy.

## Install conda environment
``` bash
conda create -n GPS python=3.9.13
conda activate GPS
pip install -r requirements.txt
``` 

## Data structure
The following shows the data structure requirement if you want to train the model on your own data.
```
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

## Train a GP model
``` bash
python3 main_DKL.py
``` 

## Test
``` bash
python3 main_DKL_test.py
``` 
## Query performance at various bias and variance
``` bash
python3 performance_estimation.py
``` 
