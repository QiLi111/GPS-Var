import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random,re
from utils.transforms import Binary
import torch
from utils.transforms import z_score, Flip, Affine



class DataSet(Dataset):
    def __init__(self, datapath,num_class,labels,labels_inference,data_aug,img_list = None,train_val = None,simulated_model = None,sigma = None,mu = None):
        self.datapath = datapath
        self.img_list = img_list
        self.num_class = num_class
        self.labels = labels
        self.labels_inference = labels_inference
        self.data_aug = data_aug
        self.simulated_model = simulated_model
        self.sigma = sigma
        self.mu = mu

        self.AffineTransform = Affine(prob = 0.3, scale = (1,1), degrees = 5, shear = 0, translate = 0)
        self.transform_train=[z_score(), Flip(), self.AffineTransform]
        self.transform_val=[z_score()]
        self.bin = Binary()
        if train_val == 'train':
            self.transform = self.transform_train
        elif train_val == 'val':
            self.transform = self.transform_val

        self.image_path = os.path.join(self.datapath, 'images')
        self.seg_path1 = os.path.join(self.datapath, 'seg_01')
        self.seg_path2 = os.path.join(self.datapath, 'seg_02')
        self.seg_path3 = os.path.join(self.datapath, 'seg_03')
        self.seg_path_GT = os.path.join(self.datapath, 'generated_majority_vote_and_vs')
        self.observation_labels = [self.seg_path1, self.seg_path2, self.seg_path3]
        self.observation_labels_4 = [self.seg_path1, self.seg_path2, self.seg_path3, self.seg_path_GT]
        self.seg_path_voted = os.path.join(self.datapath, 'generated_majority_vote')

        # path of simulated labels
        if self.simulated_model:
            self.seg_path_simulated = os.path.join(self.datapath,'data_simulation', self.simulated_model,"sigma_"+'%.04f'%self.sigma+"_mu_"+'%.04f'%self.mu)

        self.seg_path_simulated_ind = os.path.join(self.datapath,'data_simulation_ind',"bias_"+'%.04f'%self.mu+"_variance_"+'%.04f'%self.sigma)
        
    def split_data(self,split_ratio=[3,1,1]):
        # randon split data, and save the name of each case into a file, saved in json format

        self.all_images = []
        self.sum_images = []
        

        all_patients = os.listdir(self.image_path)
        for i in range(len(all_patients)):
            itk_image = sitk.ReadImage(os.path.join(self.image_path,all_patients[i]))
            image = sitk.GetArrayFromImage(itk_image)
            self.sum_images.append(image.shape[0])
            for j in range(image.shape[0]):
                self.all_images.append([all_patients[i],str(j)])

        if sum(self.sum_images) != 6644:
            assert('Warning: the number of cases is not correct')
        
        self.num_images_train = int(len(self.all_images)*split_ratio[0]/sum(split_ratio))
        self.num_images_val = int(len(self.all_images)*split_ratio[1]/sum(split_ratio))
        self.num_images_test = len(self.all_images) - self.num_images_train - self.num_images_val
        
        if self.num_images_train + self.num_images_val + self.num_images_test != 6644:
            assert('Warning: the number of cases is not correct')
        
        self.train_imgs,self.val_imgs,self.test_imgs = [],[],[]
        self.train_train_imgs, self.train_val_imgs,self.val_train_imgs, self.val_val_imgs, self.test_train_imgs, self.test_val_imgs = [],[],[],[],[],[]

        random.shuffle(all_patients)
        for i in range(len(all_patients)):
            itk_image = sitk.ReadImage(os.path.join(self.image_path,all_patients[i]))
            image = sitk.GetArrayFromImage(itk_image)

            if len(self.train_imgs) < self.num_images_train:
                for j in range(image.shape[0]):
                    self.train_imgs.append([all_patients[i],str(j)])

                if (len(self.train_train_imgs) < self.num_images_train/2):
                    for j in range(image.shape[0]):
                        self.train_train_imgs.append([all_patients[i],str(j)])
                elif len(self.train_train_imgs) >= self.num_images_train/2 and (len(self.train_val_imgs) < self.num_images_train/2):
                    for j in range(image.shape[0]):
                        self.train_val_imgs.append([all_patients[i],str(j)])
                else:
                    assert('Warning: the number of cases in the split file is not correct')
            
            elif len(self.train_imgs) >= self.num_images_train and len(self.test_imgs) < self.num_images_test:
                for j in range(image.shape[0]):
                    self.test_imgs.append([all_patients[i],str(j)])

                if len(self.test_train_imgs) < self.num_images_test/2:
                    for j in range(image.shape[0]):
                        self.test_train_imgs.append([all_patients[i],str(j)])
                elif len(self.test_train_imgs) >= self.num_images_test/2 and len(self.test_val_imgs) < self.num_images_test/2:
                    for j in range(image.shape[0]):
                        self.test_val_imgs.append([all_patients[i],str(j)])
                else:
                    assert('Warning: the number of cases in the split file is not correct')

            elif len(self.train_imgs) >= self.num_images_train and len(self.test_imgs) >= self.num_images_test and len(self.val_imgs) < self.num_images_val:
                for j in range(image.shape[0]):
                    self.val_imgs.append([all_patients[i],str(j)])
                
                if len(self.val_train_imgs) < self.num_images_val/2:
                    for j in range(image.shape[0]):
                        self.val_train_imgs.append([all_patients[i],str(j)])
                elif len(self.val_train_imgs) >= self.num_images_val/2 and len(self.val_val_imgs) < self.num_images_val/2:
                    for j in range(image.shape[0]):
                        self.val_val_imgs.append([all_patients[i],str(j)])
                    
        print('train:',len(self.train_imgs),'val:',len(self.val_imgs),'test:',len(self.test_imgs))
        print('train_train:',len(self.train_train_imgs), 'train_val:',len(self.train_val_imgs),'val_train:',len(self.val_train_imgs), 'val_val:',len(self.val_val_imgs), 'test_train:',len(self.test_train_imgs), 'test_val:',len(self.test_val_imgs))
        
        if len(self.train_imgs) + len(self.val_imgs) + len(self.test_imgs) != 6644:
            assert('Warning: the number of cases is not correct')
        if len(self.train_train_imgs) + len(self.train_val_imgs) + len(self.val_train_imgs) + len(self.val_val_imgs) + len(self.test_train_imgs) + len(self.test_val_imgs) != 6644:
            assert('Warning: the number of cases is not correct')
        
        if len(self.train_train_imgs)+len(self.train_val_imgs) != len(self.train_imgs):
            assert('Warning: the number of cases is not correct')
        if len(self.val_train_imgs)+len(self.val_val_imgs) != len(self.val_imgs):
            assert('Warning: the number of cases is not correct')
        if len(self.test_train_imgs)+len(self.test_val_imgs) != len(self.test_imgs):
            assert('Warning: the number of cases is not correct')


        os.makedirs(os.path.join(self.datapath,'data_split'),exist_ok=True)
        os.makedirs(os.path.join(self.datapath,'data_split_few_shot'),exist_ok=True)

        with open(os.path.join(self.datapath, 'data_split', 'train_list.json'), 'w') as fp:
            for case in self.train_imgs:
                fp.write(str(case) + '\n')

        with open(os.path.join(self.datapath, 'data_split', 'val_list.json'), 'w') as fp:
            for case in self.val_imgs:
                fp.write(str(case) + '\n')

        with open(os.path.join(self.datapath, 'data_split', 'test_list.json'), 'w') as fp:
            for case in self.test_imgs:
                fp.write(str(case) + '\n')

        with open(os.path.join(self.datapath, 'data_split_few_shot', 'train_train_list.json'), 'w') as fp:
            for case in self.train_train_imgs:
                fp.write(str(case) + '\n')

        with open(os.path.join(self.datapath, 'data_split_few_shot', 'val_train_list.json'), 'w') as fp:
            for case in self.val_train_imgs:
                fp.write(str(case) + '\n')

        with open(os.path.join(self.datapath, 'data_split_few_shot', 'test_train_list.json'), 'w') as fp:
            for case in self.test_train_imgs:
                fp.write(str(case) + '\n')

        with open(os.path.join(self.datapath, 'data_split_few_shot', 'train_val_list.json'), 'w') as fp:
            for case in self.train_val_imgs:
                fp.write(str(case) + '\n')

        with open(os.path.join(self.datapath, 'data_split_few_shot', 'val_val_list.json'), 'w') as fp:
            for case in self.val_val_imgs:
                fp.write(str(case) + '\n')

        with open(os.path.join(self.datapath, 'data_split_few_shot', 'test_val_list.json'), 'w') as fp:
            for case in self.test_val_imgs:
                fp.write(str(case) + '\n')

        


    def load_data_split(self, split_train, split_val, split_test):
        # load the data split from the file, for each site
        with open(split_train, 'r') as fp:
            rows = fp.readlines()
        image_list_train = [row[:-1] for row in rows]
        with open(split_val, 'r') as fp:
            rows = fp.readlines()
        image_list_val = [row[:-1] for row in rows]
        with open(split_test, 'r') as fp:
            rows = fp.readlines()
        image_list_test = [row[:-1] for row in rows]

        if len(image_list_train) + len(image_list_val) + len(image_list_test) != 6644:
            assert('Warning: the number of cases in the split file is not correct')
        if len(set([sublist[:20] for sublist in image_list_train]).intersection(set([sublist[:20] for sublist in image_list_val]))) != 0:
            assert('Warning: the train and val split has overlap')
        if len(set([sublist[:20] for sublist in image_list_train]).intersection(set([sublist[:20] for sublist in image_list_test]))) != 0:
            assert('Warning: the train and test split has overlap')
        if len(set([sublist[:20] for sublist in image_list_val]).intersection(set([sublist[:20] for sublist in image_list_test]))) != 0:
            assert('Warning: the val and test split has overlap')
        if len(set(image_list_train).union(set(image_list_val)).union(set(image_list_test))) != 6644:
            assert('Warning: the union of train, val and test split is not equal to the total number of cases')
        if len(set([sublist[:20] for sublist in image_list_train]).union(set([sublist[:20] for sublist in image_list_val])).union(set([sublist[:20] for sublist in image_list_test]))) != 249:
            assert('Warning: the union of train, val and test split is not equal to 249')
        # if len(image_list_train) != int(6644*3/5) or len(image_list_val) != int(6644/5) or len(image_list_test) != 6644-int(6644*3/5)-int(6644/5):
        #     assert('Warning: the number of cases in the split file is not correct')



        return [DataSet(datapath = self.datapath, num_class=self.num_class,labels=self.labels,labels_inference=self.labels_inference,data_aug = self.data_aug, img_list = image_list_train,train_val = 'train',simulated_model = self.simulated_model,sigma = self.sigma,mu = self.mu),\
                DataSet(datapath = self.datapath, num_class=self.num_class,labels=self.labels,labels_inference=self.labels_inference,data_aug = self.data_aug,img_list = image_list_val,train_val = 'val',simulated_model = self.simulated_model,sigma = self.sigma,mu = self.mu),\
                DataSet(datapath = self.datapath, num_class=self.num_class,labels=self.labels,labels_inference=self.labels_inference,data_aug = self.data_aug,img_list = image_list_test,train_val = 'val',simulated_model = self.simulated_model,sigma = self.sigma,mu = self.mu)]
    

    def load_data_split_few_shot(self, split_train_train, split_train_val, split_val_train, split_val_val, split_test_train, split_test_val):
        with open(split_train_train, 'r') as fp:
            rows = fp.readlines()
        image_list_train_train = [row[:-1] for row in rows]
        with open(split_val_train, 'r') as fp:
            rows = fp.readlines()
        image_list_val_train = [row[:-1] for row in rows]
        with open(split_test_train, 'r') as fp:
            rows = fp.readlines()
        image_list_test_train = [row[:-1] for row in rows]

        with open(split_train_val, 'r') as fp:
            rows = fp.readlines()
        image_list_train_val = [row[:-1] for row in rows]
        with open(split_val_val, 'r') as fp:
            rows = fp.readlines()
        image_list_val_val = [row[:-1] for row in rows]
        with open(split_test_val, 'r') as fp:
            rows = fp.readlines()
        image_list_test_val = [row[:-1] for row in rows]



        if len(image_list_train_train) + len(image_list_train_val) + len(image_list_val_train) + len(image_list_val_val) + len(image_list_test_train) + len(image_list_test_val)!= 6644:
            assert('Warning: the number of cases in the split file is not correct')
        if len(set([sublist[:20] for sublist in image_list_train_train]).intersection(set([sublist[:20] for sublist in image_list_train_val]))) != 0:
            assert('Warning: the train_train and train_val split has overlap')

        if len(set([sublist[:20] for sublist in image_list_val_train]).intersection(set([sublist[:20] for sublist in image_list_val_val]))) != 0:
            assert('Warning: the val_train and val_val split has overlap')

        if len(set([sublist[:20] for sublist in image_list_test_train]).intersection(set([sublist[:20] for sublist in image_list_test_val]))) != 0:
            assert('Warning: the test_train and test_val split has overlap')

        if len(set([sublist[:20] for sublist in image_list_train_train]).intersection(set([sublist[:20] for sublist in image_list_val_train]))) != 0:
            assert('Warning: the train and val split has overlap')
        
        if len(set([sublist[:20] for sublist in image_list_train_train]).intersection(set([sublist[:20] for sublist in image_list_test_train]))) != 0:
            assert('Warning: the train and test split has overlap')
        
        if len(set([sublist[:20] for sublist in image_list_val_train]).intersection(set([sublist[:20] for sublist in image_list_test_train]))) != 0:
            assert('Warning: the val and test split has overlap')

        if len(set([sublist[:20] for sublist in image_list_train_val]).intersection(set([sublist[:20] for sublist in image_list_val_val]))) != 0:
            assert('Warning: the train and val split has overlap')

        if len(set([sublist[:20] for sublist in image_list_train_val]).intersection(set([sublist[:20] for sublist in image_list_test_val]))) != 0:
            assert('Warning: the train and test split has overlap')
        
        if len(set([sublist[:20] for sublist in image_list_val_val]).intersection(set([sublist[:20] for sublist in image_list_test_val]))) != 0:
            assert('Warning: the val and test split has overlap')

        if len(set(image_list_train_train).union(set(image_list_train_val)).union(set(image_list_val_train)).union(set(image_list_val_val)).union(set(image_list_test_train)).union(set(image_list_test_val))) != 6644:
            assert('Warning: the union of train, val and test split is not equal to the total number of cases')  
        
        if len(set([sublist[:20] for sublist in image_list_train_train]).union(set([sublist[:20] for sublist in image_list_train_val])).union(set([sublist[:20] for sublist in image_list_val_train])).union(set([sublist[:20] for sublist in image_list_val_val])).union(set([sublist[:20] for sublist in image_list_test_train])).union(set([sublist[:20] for sublist in image_list_test_val]))) != 249:
            assert('Warning: the union of train, val and test split is not equal to 249')

        return [DataSet(datapath = self.datapath, num_class=self.num_class,labels=self.labels,img_list = image_list_train_train),\
                DataSet(datapath = self.datapath, num_class=self.num_class,labels=self.labels,img_list = image_list_train_val),\
                DataSet(datapath = self.datapath, num_class=self.num_class,labels=self.labels,img_list = image_list_val_train),\
                DataSet(datapath = self.datapath, num_class=self.num_class,labels=self.labels,img_list = image_list_val_val),\
                DataSet(datapath = self.datapath, num_class=self.num_class,labels=self.labels,img_list = image_list_test_train),\
                DataSet(datapath = self.datapath, num_class=self.num_class,labels=self.labels,img_list = image_list_test_val)]
    


    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):

        image_name = os.path.join(self.image_path,self.img_list[idx][2:33])
        itk_image = sitk.ReadImage(image_name)
        image = sitk.GetArrayFromImage(itk_image)[int(re.findall(r'\d+', self.img_list[idx])[1])]
            
        if self.labels_inference == 'voted_only':
            mask_name_vote = os.path.join(self.seg_path_voted,'label_'+self.img_list[idx][2:33])
            itk_mask_vote = sitk.ReadImage(mask_name_vote)
            mask_vote = np.transpose(sitk.GetArrayFromImage(itk_mask_vote),(2,1,0))[int(re.findall(r'\d+', self.img_list[idx])[1])]


        elif self.labels_inference == 'high_quality':
            try:
                # for the generated and saved .nii.gz file, need to transpose
                mask_name_vote = os.path.join(self.seg_path_GT,'label_'+self.img_list[idx][2:33])
                itk_mask_vote = sitk.ReadImage(mask_name_vote)
                mask_vote = np.transpose(sitk.GetArrayFromImage(itk_mask_vote),(2,1,0))[int(re.findall(r'\d+', self.img_list[idx])[1])]

            except:
                try:
                    mask_name_vote = os.path.join(self.seg_path_GT,'label_'+self.img_list[idx][2:26]+'_VS.nii.gz')
                    itk_mask_vote = sitk.ReadImage(mask_name_vote)
                    mask_vote = sitk.GetArrayFromImage(itk_mask_vote)[int(re.findall(r'\d+', self.img_list[idx])[1])]
                    mask_vote = np.uint8(mask_vote)
                except:
                    mask_name_vote = os.path.join(self.seg_path_GT,'label_'+self.img_list[idx][2:26]+'_vs.nii.gz')
                    itk_mask_vote = sitk.ReadImage(mask_name_vote)
                    mask_vote = sitk.GetArrayFromImage(itk_mask_vote)[int(re.findall(r'\d+', self.img_list[idx])[1])]
                    mask_vote = np.uint8(mask_vote)


        if self.labels == 'all':
            # get all labels: seg1, seg2, seg3, seg_vote

            mask_name1 = os.path.join(self.seg_path1,'label_'+self.img_list[idx][2:33])
            itk_mask1 = sitk.ReadImage(mask_name1)
            mask1 = sitk.GetArrayFromImage(itk_mask1)[int(re.findall(r'\d+', self.img_list[idx])[1])]

            mask_name2 = os.path.join(self.seg_path2,'label_'+self.img_list[idx][2:33])
            itk_mask2 = sitk.ReadImage(mask_name2)
            mask2 = sitk.GetArrayFromImage(itk_mask2)[int(re.findall(r'\d+', self.img_list[idx])[1])]

            mask_name3 = os.path.join(self.seg_path3,'label_'+self.img_list[idx][2:33])
            itk_mask3 = sitk.ReadImage(mask_name3)
            mask3 = sitk.GetArrayFromImage(itk_mask3)[int(re.findall(r'\d+', self.img_list[idx])[1])]

            if mask_vote.shape[0] != mask1.shape[0] or mask_vote.shape[1] != mask1.shape[1]:
                print(mask_vote.shape,mask1.shape)
                raise('voted label is not the same size')
            
            mask_all = np.stack([mask1,mask2,mask3,mask_vote],axis=0)
            # mask_decomp = self._label_decomp(mask_all, num_cls=self.num_class)
            selected_observation = -1

        elif self.labels == 'co':
            # using only consensus label
            mask_all = mask_vote
            # mask_decomp = self._label_decomp(mask_all, num_cls=self.num_class)
            selected_observation = -1

        elif self.labels == 'random3':
            # random select from three labels
            selected_observation = np.random.randint(3)
            selected_folder = self.observation_labels[selected_observation]

            mask_name1 = os.path.join(selected_folder,'label_'+self.img_list[idx][2:33])
            itk_mask1 = sitk.ReadImage(mask_name1)
            mask_all = sitk.GetArrayFromImage(itk_mask1)[int(re.findall(r'\d+', self.img_list[idx])[1])]
        
        # elif self.labels == 'random4':
        #     # random select from four avaliable labels
        #     selected_observation = np.random.randint(4)
        #     selected_folder = self.observation_labels_4[selected_observation]

        #     mask_name1 = os.path.join(selected_folder,'label_'+self.img_list[idx][2:33])
        #     itk_mask1 = sitk.ReadImage(mask_name1)
        #     mask_all = sitk.GetArrayFromImage(itk_mask1)[int(re.findall(r'\d+', self.img_list[idx])[1])]

        elif self.labels == 'inference':

            # get all three labels: seg1, seg2, seg3

            mask_name1 = os.path.join(self.seg_path1,'label_'+self.img_list[idx][2:33])
            itk_mask1 = sitk.ReadImage(mask_name1)
            mask1 = sitk.GetArrayFromImage(itk_mask1)[int(re.findall(r'\d+', self.img_list[idx])[1])]

            mask_name2 = os.path.join(self.seg_path2,'label_'+self.img_list[idx][2:33])
            itk_mask2 = sitk.ReadImage(mask_name2)
            mask2 = sitk.GetArrayFromImage(itk_mask2)[int(re.findall(r'\d+', self.img_list[idx])[1])]

            mask_name3 = os.path.join(self.seg_path3,'label_'+self.img_list[idx][2:33])
            itk_mask3 = sitk.ReadImage(mask_name3)
            mask3 = sitk.GetArrayFromImage(itk_mask3)[int(re.findall(r'\d+', self.img_list[idx])[1])]

            if mask_vote.shape[0] != mask1.shape[0] or mask_vote.shape[1] != mask1.shape[1]:
                print(mask_vote.shape,mask1.shape)
                raise('voted label is not the same size')
            
            mask_all = np.stack([mask1,mask2,mask3],axis=0)
            selected_observation = -1

        elif self.labels == 'seg1':
            mask_name1 = os.path.join(self.seg_path1,'label_'+self.img_list[idx][2:33])
            itk_mask1 = sitk.ReadImage(mask_name1)
            mask1 = sitk.GetArrayFromImage(itk_mask1)[int(re.findall(r'\d+', self.img_list[idx])[1])]

            mask_all = mask1
            selected_observation = -1

        elif self.labels == 'seg2':
            mask_name2 = os.path.join(self.seg_path2,'label_'+self.img_list[idx][2:33])
            itk_mask2 = sitk.ReadImage(mask_name2)
            mask2 = sitk.GetArrayFromImage(itk_mask2)[int(re.findall(r'\d+', self.img_list[idx])[1])]

            mask_all = mask2
            selected_observation = -1
        
        elif self.labels == 'seg3':

            mask_name3 = os.path.join(self.seg_path3,'label_'+self.img_list[idx][2:33])
            itk_mask3 = sitk.ReadImage(mask_name3)
            mask3 = sitk.GetArrayFromImage(itk_mask3)[int(re.findall(r'\d+', self.img_list[idx])[1])]

            mask_all = mask3
            selected_observation = -1

        elif self.labels == 'seg_simulated':
            mask_name3 = os.path.join(self.seg_path_simulated,'label_'+self.img_list[idx][2:33])
            itk_mask3 = sitk.ReadImage(mask_name3)
            mask3 = sitk.GetArrayFromImage(itk_mask3)[int(re.findall(r'\d+', self.img_list[idx])[1])]

            mask_all = mask3
            selected_observation = -1

        elif self.labels == 'simulated_ind':
            mask_name3 = os.path.join(self.seg_path_simulated_ind,'label_'+self.img_list[idx][2:33])
            itk_mask3 = sitk.ReadImage(mask_name3)
            mask3 = sitk.GetArrayFromImage(itk_mask3)[int(re.findall(r'\d+', self.img_list[idx])[1])]

            mask_all = mask3
            selected_observation = -1
            
        else:
            raise('labels not recognized')
        
        # data augmentation
        image = torch.from_numpy(image).float()
        mask_all = torch.from_numpy(mask_all).float()
        mask_vote = torch.from_numpy(mask_vote).float()
        if self.data_aug == 'add':
            for _transform in self.transform:
                if _transform.__class__.__name__ == 'Flip' or _transform.__class__.__name__ == 'Affine':

                    input_ = torch.stack([image, mask_all,mask_vote])
                    input_ = _transform(input_)
                    image, mask_all,mask_vote = torch.chunk(input_, 3)
                    mask_all = self.bin(mask_all.squeeze(0))
                    mask_vote = self.bin(mask_vote.squeeze(0))
                    image = image.squeeze(0)

                else:
                    image = _transform(image)

            
        return image, mask_all, mask_vote,selected_observation # image; labels with noise; labels without noise; which observation #, mask_decomp
   
       
    def _label_decomp(self,label_vol, num_cls):
        # adapted from https://github.com/liuquande/MS-Net 
        """
        decompose label for softmax classifier
        original labels are batchsize * W * H * 1, with label values 0,1,2,3...
        this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
        numpy version of tf.one_hot
        """
        one_hot = []
        for i in range(num_cls):
            _vol = np.zeros(label_vol.shape)
            _vol[label_vol == i] = 1
            one_hot.append(_vol)

        return np.stack(one_hot, axis=-1)

