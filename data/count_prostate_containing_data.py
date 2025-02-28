
# calculate the number of prostate containing data in the dataset

import SimpleITK as sitk
import numpy as np
import os,re

label_path_majority_vote_vs = './data/generated_majority_vote_and_vs'
label_path_majority_vote = './data/generated_majority_vote'

labels_all_vote = sorted(os.listdir(label_path_majority_vote_vs))
pro_gland = []
for i in range(len(labels_all_vote)):
    itk_mask0 = sitk.ReadImage(os.path.join(label_path_majority_vote_vs,labels_all_vote[i]))
    mask0 = sitk.GetArrayFromImage(itk_mask0)
    if not labels_all_vote[i].endswith('VS.nii.gz') and not labels_all_vote[i].endswith('vs.nii.gz'):
        mask0 = np.transpose(mask0,(2,1,0))
    
    for j in range(mask0.shape[0]):
        if len(np.unique(mask0[j]))==2:
            pro_gland.append(1)
        elif len(np.unique(mask0[j]))==1 and np.unique(mask0[j])[0]==0:
            pro_gland.append(0)
        elif len(np.unique(mask0[j]))==1 and np.unique(mask0[j])[0]!=0:
            raise('only one class and not 0')
        elif len(np.unique(mask0[j]))>3:
            raise('more than 3 classes')
        

# calculate the images which contains prostate, from marjoity vote
        
labels_all_vote = sorted(os.listdir(label_path_majority_vote))
pro_gland_2 = []
for i in range(len(labels_all_vote)):
    itk_mask0 = sitk.ReadImage(os.path.join(label_path_majority_vote,labels_all_vote[i]))
    mask0 = sitk.GetArrayFromImage(itk_mask0)
    mask0 = np.transpose(mask0,(2,1,0))
    
    for j in range(mask0.shape[0]):
        if len(np.unique(mask0[j]))==2:
            pro_gland_2.append(1)
        elif len(np.unique(mask0[j]))==1 and np.unique(mask0[j])[0]==0:
            pro_gland_2.append(0)
        elif len(np.unique(mask0[j]))==1 and np.unique(mask0[j])[0]!=0:
            raise('only one class and not 0')
        elif len(np.unique(mask0[j]))>3:
            raise('more than 3 classes')

print('done')