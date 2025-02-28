
import SimpleITK as sitk
import numpy as np
import os,re
import matplotlib.pyplot as plt


image_path = './data/images'
label_path1 = './data/seg_01'
label_path2 = './data/seg_02'
label_path3 = './data/seg_03'
label_path_vote = './data/vs_reg'

plot_path = './data/plots'
os.makedirs(os.path.join(plot_path,image_path),exist_ok=True)
os.makedirs(os.path.join(plot_path,label_path1),exist_ok=True)
os.makedirs(os.path.join(plot_path,label_path2),exist_ok=True)
os.makedirs(os.path.join(plot_path,label_path3),exist_ok=True)
os.makedirs(os.path.join(plot_path,label_path_vote),exist_ok=True)

images_all = sorted(os.listdir(image_path))
labels_all1 = sorted(os.listdir(label_path1))
labels_all2 = sorted(os.listdir(label_path2))
labels_all3 = sorted(os.listdir(label_path3))
labels_all_vote = sorted(os.listdir(label_path_vote))
sum_images = []
for i in range(len(images_all)):

    itk_image = sitk.ReadImage(os.path.join(image_path,images_all[i]))
    image = sitk.GetArrayFromImage(itk_image)
    saved_path = os.path.join(plot_path,image_path,images_all[i])
    os.makedirs(saved_path,exist_ok=True)

    sum_images.append(image.shape[0])

    # for ii in range(image.shape[0]):

    #     plt.imshow(image[ii],'gray')
    #     plt.savefig(os.path.join(saved_path,f'{ii}.png'))
    #     plt.close()
sum_images = np.array(sum_images)
# plot histogram
plt.hist(sum_images)
plt.savefig(os.path.join(plot_path,image_path,'histogram.png'))
plt.close()


print('Done')
labels_all = [labels_all1,labels_all2,labels_all3,labels_all_vote]
label_path_all = [label_path1,label_path2,label_path3,label_path_vote]
sum_labels_all = [[],[],[],[]]
masks_all = [[],[],[],[]]
for j in range(4):
    for i in range(len(labels_all[j])):
        
        itk_mask = sitk.ReadImage(os.path.join(label_path_all[j],labels_all[j][i]))
        mask = sitk.GetArrayFromImage(itk_mask)
        sum_labels_all[j].append(mask.shape[0])
        # corresponding image
        itk_image = sitk.ReadImage(os.path.join(image_path,'VOL_'+re.findall(r'\d+\.?\d*', labels_all[j][i])[0]+'_FRAMES.nii.gz'))
        image = sitk.GetArrayFromImage(itk_image)
        if mask.shape[0]!=image.shape[0]:
            raise('image and label is not the same size')

        if i == 0 :
            masks_all[j] = np.array(mask)
        else:
            masks_all[j] = np.concatenate((masks_all[j],mask),axis=0)
        saved_path = os.path.join(plot_path,label_path_all[j],labels_all[j][i])
        os.makedirs(saved_path,exist_ok=True)
        # for ii in range(mask.shape[0]):

        #     # plt.imshow(image[ii],'gray')
        #     # plt.imshow(mask[ii],'gray',alpha = 0.3)
        #     plt.imshow(mask[ii],'gray')
        #     plt.savefig(os.path.join(saved_path,f'{ii}.png'))
        #     plt.close()


# check the number of images with prostate gland
pro_gland = [[],[],[],[]]
for j in range(len(masks_all)): 
    for i in range(masks_all[j].shape[0]):
        if len(np.unique(masks_all[j][i]))==2:
            pro_gland[j].append(1)
        elif len(np.unique(masks_all[j][i]))==1 and np.unique(masks_all[j][i])[0]==0:
            pro_gland[j].append(0)
        elif len(np.unique(masks_all[j][i]))==1 and np.unique(masks_all[j][i])[0]!=0:
            raise('only one class and not 0')
        elif len(np.unique(masks_all[j][i]))>3:
            raise('more than 3 classes')
# slice voted for if the image containing prostat gland or not is not correct
# as the labels from three observers may not overlap,
# such that when looking for slice level, the voted image will be regarded as containing prostate;
# however, the voted image may not contain prostate gland when doing the pixel level voting
aa=[]
for i in range(len(pro_gland[0])):
    if pro_gland[0][i] + pro_gland[1][i]+pro_gland[2][i]>1 and pro_gland[0][i] + pro_gland[1][i]+pro_gland[2][i]<4:
        aa.append(1)
    elif pro_gland[0][i] + pro_gland[1][i]+pro_gland[2][i]>3:
        raise('more than 3 classes')
    else:
        aa.append(0)

# morjority voting
mask_vote_all = []    
for i in range(len(labels_all[0])):
    itk_mask0 = sitk.ReadImage(os.path.join(label_path_all[0],labels_all[0][i]))
    mask0 = sitk.GetArrayFromImage(itk_mask0)

    itk_mask1 = sitk.ReadImage(os.path.join(label_path_all[1],labels_all[0][i]))
    mask1 = sitk.GetArrayFromImage(itk_mask1)

    itk_mask2 = sitk.ReadImage(os.path.join(label_path_all[2],labels_all[0][i]))
    mask2 = sitk.GetArrayFromImage(itk_mask2)

    if mask0.shape[0]!=mask1.shape[0] or mask0.shape[0]!=mask2.shape[0]:
        raise('three labels are not the same size')
    
    itk_image = sitk.ReadImage(os.path.join(image_path,'VOL_'+re.findall(r'\d+\.?\d*', labels_all[0][i])[0]+'_FRAMES.nii.gz'))
    image = sitk.GetArrayFromImage(itk_image)
    if mask2.shape[0]!=image.shape[0] or mask1.shape[0]!=image.shape[0] or mask0.shape[0]!=image.shape[0]:
        raise('image and label is not the same size')


    mask_vote = np.zeros_like(mask0)
    saved_path = os.path.join(plot_path,'marjority_vote',labels_all[0][i])
    os.makedirs(saved_path,exist_ok=True)
    
    saved_path_merge = os.path.join(plot_path,'merged_4labels_majority_voted',labels_all[0][i])
    os.makedirs(saved_path_merge,exist_ok=True)

    for iii in range(mask_vote.shape[0]):
        vote = np.stack([mask0[iii], mask1[iii], mask2[iii]], axis=0)
        majority_vote = np.sum(vote, axis=0) > 1
        majority_vote = majority_vote.astype(np.uint8)
        mask_vote[iii] = majority_vote

        # plt.imshow(mask_vote[iii])
        # plt.savefig(os.path.join(saved_path,f'{iii}.png'))
        # plt.close()

        # plt.imshow(image[iii],'gray')
        # plt.contour(mask0[iii],colors='red')
        # plt.contour(mask1[iii],colors='green')
        # plt.contour(mask2[iii],colors='blue')
        # plt.contour(majority_vote,colors='orange')

        # plt.savefig(os.path.join(saved_path_merge,f'{iii}.png'))
        # plt.close()


    if i == 0 :
        mask_vote_all = np.array(mask_vote)
    else:
        mask_vote_all = np.concatenate((mask_vote_all,mask_vote),axis=0)

   
pro_gland_vote = []
for i in range(mask_vote_all.shape[0]):
    if len(np.unique(mask_vote_all[i]))==2:
        pro_gland_vote.append(1)
    elif len(np.unique(mask_vote_all[i]))==1 and np.unique(mask_vote_all[i])[0]==0:
        pro_gland_vote.append(0)
    elif len(np.unique(mask_vote_all[i]))==1 and np.unique(mask_vote_all[i])[0]!=0:
        raise('only one class and not 0')
    elif len(np.unique(mask_vote_all[i]))>3:
        raise('more than 3 classes')




# plot the four labels in the same figure, together with the images
for i in range(len(labels_all[3])):
    saved_path = os.path.join(plot_path,'merged_4labels_expert',labels_all[3][i])
    os.makedirs(saved_path,exist_ok=True)

    itk_mask3 = sitk.ReadImage(os.path.join(label_path_all[3],labels_all[3][i]))
    mask3 = sitk.GetArrayFromImage(itk_mask3)

    itk_mask2 = sitk.ReadImage(os.path.join(label_path_all[2],'label_VOL_'+re.findall(r'\d+\.?\d*', labels_all[3][i])[0]+'_FRAMES.nii.gz'))
    mask2 = sitk.GetArrayFromImage(itk_mask2)

    itk_mask1 = sitk.ReadImage(os.path.join(label_path_all[1],'label_VOL_'+re.findall(r'\d+\.?\d*', labels_all[3][i])[0]+'_FRAMES.nii.gz'))
    mask1 = sitk.GetArrayFromImage(itk_mask1)

    itk_mask0 = sitk.ReadImage(os.path.join(label_path_all[0],'label_VOL_'+re.findall(r'\d+\.?\d*', labels_all[3][i])[0]+'_FRAMES.nii.gz'))
    mask0 = sitk.GetArrayFromImage(itk_mask0)
    # corresponding image
    itk_image = sitk.ReadImage(os.path.join(image_path,'VOL_'+re.findall(r'\d+\.?\d*', labels_all[3][i])[0]+'_FRAMES.nii.gz'))
    image = sitk.GetArrayFromImage(itk_image)
    if mask3.shape[0]!=image.shape[0] or mask2.shape[0]!=image.shape[0] or mask1.shape[0]!=image.shape[0] or mask0.shape[0]!=image.shape[0]:
        raise('image and label is not the same size')


    # for ii in range(mask3.shape[0]):

    #     # plt.imshow(image[ii],'gray')
    #     # plt.imshow(mask0[ii],'gray',alpha = 0.3)
    #     # plt.imshow(mask1[ii],'gray',alpha = 0.3)
    #     # plt.imshow(mask2[ii],'gray',alpha = 0.3)
    #     # plt.imshow(mask3[ii],'gray',alpha = 0.3)
    #     # _, binary = cv2.threshold(mask0[ii], 127, 255, cv2.THRESH_BINARY)
    #     # contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # contour_image = np.zeros_like(mask0[ii])  # Create a blank canvas of the same size

    #     # contours, _ = cv2.findContours(mask0[ii], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # cv2.drawContours(image[ii], contours, -1, (0, 0, 255), 1)

    #     # contours, _ = cv2.findContours(mask1[ii], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # cv2.drawContours(image[ii], contours, -1, (0, 255, 0), 1)

    #     # contours, _ = cv2.findContours(mask2[ii], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # cv2.drawContours(image[ii], contours, -1, (255, 0, 0), 1)

    #     # contours, _ = cv2.findContours(mask3[ii].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # cv2.drawContours(image[ii], contours, -1, (255, 128, 0) , 1)


    #     plt.imshow(image[ii],'gray')
    #     plt.contour(mask0[ii],colors='red')
    #     plt.contour(mask1[ii],colors='green')
    #     plt.contour(mask2[ii],colors='blue')
    #     plt.contour(mask3[ii],colors='orange')

    #     plt.savefig(os.path.join(saved_path,f'{ii}.png'))
    #     plt.close()


print('Done')
