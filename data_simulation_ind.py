
# generate simulated data, based on high quality annotaion

import os
import torch
from utils.utils import *
from utils.data_loader import DataSet
import SimpleITK as sitk
import nibabel as nib
from utils.data_simulation_funs import generate_continuous_annotation

bias_level = 6
variance_level = 100


image_path = './data/images'
seg_path_GT = os.path.join('./data', 'generated_majority_vote_and_vs')
fd_name_save = 'data/data_simulation_ind'
saved_label_path = os.path.join(fd_name_save,"bias_"+'%.04f'%bias_level+"_variance_"+'%.04f'%variance_level)
os.makedirs(saved_label_path,exist_ok=True)

args = {}
args['seed'] = 0
args['dataset'] = './data'
args = argparse.Namespace(**args)

print(f"Seeding with seed: {args.seed}")
seed_all(int(args.seed))
args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cuda_avail, device = torch_init(args.device)
print("pytorch using device", device)

split_train = os.path.join(args.dataset,'data_split','train_list.json')
split_val = os.path.join(args.dataset,'data_split','val_list.json')
# split_test = os.path.join(args.dataset,'data_split','test_list.json')

with open(split_train, 'r') as fp:
    rows = fp.readlines()
image_list_train = [row[:-1] for row in rows]
with open(split_val, 'r') as fp:
    rows = fp.readlines()
image_list_val = [row[:-1] for row in rows]

# with open(split_test, 'r') as fp:
#     rows = fp.readlines()
# image_list_test = [row[:-1] for row in rows]

sub_list_train_val = set([sublist[2:33] for sublist in image_list_train]).union(set([sublist[2:33] for sublist in image_list_val]))
sub_list_train_val = sorted(list(sub_list_train_val))


if True:
    for i_sub in sub_list_train_val:

        # obtain image

        image_name = os.path.join(image_path,i_sub)
        itk_image = sitk.ReadImage(image_name)
        image = sitk.GetArrayFromImage(itk_image)

        # obtain the ground truth mask (high quality annotation)

        try:
            # for the generated and saved .nii.gz file, need to transpose
            mask_name_vote = os.path.join(seg_path_GT,'label_'+i_sub)
            itk_mask_vote = sitk.ReadImage(mask_name_vote)
            mask_vote = np.transpose(sitk.GetArrayFromImage(itk_mask_vote),(2,1,0))

        except:
            try:
                mask_name_vote = os.path.join(seg_path_GT,'label_'+i_sub[:24]+'_VS.nii.gz')
                itk_mask_vote = sitk.ReadImage(mask_name_vote)
                mask_vote = sitk.GetArrayFromImage(itk_mask_vote)
                mask_vote = np.uint8(mask_vote)
            except:
                mask_name_vote = os.path.join(seg_path_GT,'label_'+i_sub[:24]+'_vs.nii.gz')
                itk_mask_vote = sitk.ReadImage(mask_name_vote)
                mask_vote = sitk.GetArrayFromImage(itk_mask_vote)
                mask_vote = np.uint8(mask_vote)

        if mask_vote.shape[0]!= image.shape[0] or mask_vote.shape[1]!= image.shape[1] or mask_vote.shape[2]!= image.shape[2]:
            raise ValueError("image and mask size not match")
        
        masks_simulated=[]
        for i_mask in range(mask_vote.shape[0]):
            
            masks_temp = generate_continuous_annotation(mask_vote[i_mask], bias_level=bias_level, variance_level=variance_level)

            if i_mask == 0:
                masks_simulated = np.expand_dims(masks_temp, axis=0)
            else:
                masks_simulated = np.concatenate((masks_simulated,np.expand_dims(masks_temp, axis=0)),axis=0)
            
        
        
        nifti_mask = nib.Nifti1Image(np.transpose(masks_simulated,(2,1,0)), np.eye(4))
        nib.save(nifti_mask, os.path.join(saved_label_path,"label_"+i_sub))
        plot_simulated_labels_and_GT(image,masks_simulated,mask_vote,i_sub,saved_label_path)

        print("save simulated label for subject", i_sub)





       







    



    

  