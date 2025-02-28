
# generate majority vote label according to the three labels

import SimpleITK as sitk
import numpy as np
import os,re
import matplotlib.pyplot as plt
import nibabel as nib

image_path = './data/images'
label_path1 = './data/seg_01'
label_path2 = './data/seg_02'
label_path3 = './data/seg_03'
label_path_vote = './data/vs_reg'
label_path_majority_vote = './data/generated_majority_vote'
label_path_majority_vote_vs = './data/generated_majority_vote_and_vs'



plot_path = './data/plots'
os.makedirs(os.path.join(plot_path,'generated_majority_vote'),exist_ok=True)

images_all = sorted(os.listdir(image_path))
labels_all1 = sorted(os.listdir(label_path1))
labels_all2 = sorted(os.listdir(label_path2))
labels_all3 = sorted(os.listdir(label_path3))
labels_all_vote = sorted(os.listdir(label_path_vote))


for i in range(len(labels_all1)):
    itk_mask0 = sitk.ReadImage(os.path.join(label_path1,labels_all1[i]))
    mask0 = sitk.GetArrayFromImage(itk_mask0)

    itk_mask1 = sitk.ReadImage(os.path.join(label_path2,labels_all1[i]))
    mask1 = sitk.GetArrayFromImage(itk_mask1)

    itk_mask2 = sitk.ReadImage(os.path.join(label_path3,labels_all1[i]))
    mask2 = sitk.GetArrayFromImage(itk_mask2)

    if mask0.shape[0]!=mask1.shape[0] or mask0.shape[0]!=mask2.shape[0]:
        raise('three labels are not the same size')
    # if (mask0 == mask1).all() or (mask0 == mask2).all() or (mask1 == mask2).all():
    #     raise('three labels are the same')
    
   
    mask_vote = np.zeros_like(mask0)
    saved_path = os.path.join(plot_path,'generated_majority_vote',labels_all1[i])
    os.makedirs(saved_path,exist_ok=True)
    

    for iii in range(mask_vote.shape[0]):
        vote = np.stack([mask0[iii], mask1[iii], mask2[iii]], axis=0)
        majority_vote = np.sum(vote, axis=0) > 1
        majority_vote = majority_vote.astype(np.uint8)
        mask_vote[iii] = majority_vote

        plt.imshow(mask_vote[iii])
        plt.savefig(os.path.join(saved_path,f'{iii}.png'))
        plt.close()

    if i == 0 :
        mask_vote_all = np.array(mask_vote)
    else:
        mask_vote_all = np.concatenate((mask_vote_all,mask_vote),axis=0)

    
    # save as .nii.gz
    # mask_vote = sitk.GetImageFromArray(mask_vote)
    # mask_vote.CopyInformation(itk_mask0)
    # sitk.WriteImage(mask_vote, os.path.join(label_path_majority_vote,labels_all1[i]))
    # print(f'{labels_all1[i]} saved')
        

    nifti_img = nib.Nifti1Image(mask_vote, np.eye(4))
    nib.save(nifti_img, os.path.join(label_path_majority_vote,labels_all1[i]))
    nib.save(nifti_img, os.path.join(label_path_majority_vote_vs,labels_all1[i]))


print('Done')


        


