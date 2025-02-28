
# generate simulated data, based on the added sigma and mu

import os
import torch
import torch.nn as nn
import gpytorch
from utils.utils import *
from utils.base_network import UNetFeatureExtractor,DKLModelInducingPts,DKLModelGrid
from utils.BernoulliLikelihood_with_noise import BernoulliLikelihood_with_Noise
from utils.data_loader import DataSet
import SimpleITK as sitk
import nibabel as nib

folder = 'NoPreLoad_TrainUnet_OptimizeHyper_Add3Sigmas3Mus_NoAddNoisePred_NoFeatBN_Sum2MuReg_random3__voted_only__dice_BCE__VariationalELBO__RBF__InducingPts_uniform500__FeatDim64__bst_8__bsv_8__fr_5'
models_folders = 'Model4DataSimulation/'
folder_path = os.path.join(os.getcwd(),models_folders,folder)
image_path = './data/images'

fd_name_save = 'data/data_simulation'
saved_path = os.getcwd()+'/'+ fd_name_save +'/' +folder
os.makedirs(saved_path,exist_ok=True)

# get args
if 'args.txt' in os.listdir(folder_path): 
    args = get_args(os.path.join(folder_path,'args.txt'),folder)
    args.batch_size_val = 8
    args.saved_name = models_folders+'/'+args.saved_name
    args.pre_labels = args.labels
    args.labels = 'inference'
else:
    raise ValueError('args.txt not found in folder')
        
    

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
dataset = DataSet(datapath=args.dataset,num_class=args.num_class,labels=args.labels,labels_inference = args.labels_infer,data_aug=args.data_aug)
train_set,val_set,test_set = dataset.load_data_split(os.path.join(args.dataset,'data_split','train_list.json'),os.path.join(args.dataset,'data_split','val_list.json'),os.path.join(args.dataset,'data_split','test_list.json'))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True, num_workers=8)

# generate labels from sampled sigma and mu
sigma_all = torch.tensor([0.3,0.25,0.1,0.15,0.4,0.45,0.2,0.25,0.5,0.55,0.6,0.35,0.5222,0.4790,0.3103,0.2295])
mu_all = torch.tensor([-3,-2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2,2.5,3,2.4487,-0.5704,1.7028,-1.1801])



for _index in range(len(sigma_all)):
    sigma = sigma_all[_index]
    mu = mu_all[_index]

    if True:

        saved_label_path = os.path.join(saved_path,"sigma_"+'%.04f'%sigma+"_mu_"+'%.04f'%mu)
        os.makedirs(saved_label_path,exist_ok=True)

        feature_extractor = UNetFeatureExtractor(n_channels = 1, n_classes=args.num_class, feat_dim = args.feat_dim, FeatBN = args.FeatBatchNorm)
        feature_extractor = feature_extractor.to(device)
        num_features = feature_extractor.outc.conv.in_channels

        # strategy for scalable GP
        if args.VariationalStrategy == 'InducingPts':
            inducing_points = generate_inducing_points(feature_extractor,num_features,train_set,args,device)
            model = DKLModelInducingPts(feature_extractor, inducing_points=inducing_points,hyperparameter_fixed=args.hyperparameter, kernel_type = args.kernel_type)

        # grid is not used in the project
        elif args.VariationalStrategy == 'Grid':
            model = DKLModelGrid(feature_extractor,input_dim = num_features,grid_size = args.grid_size)
        
        if args.addnoise == 'add3sigmas' or args.addnoise == 'add3sigmas3mus':
            likelihood = BernoulliLikelihood_with_Noise(addnoise=args.addnoise)
        else:
            raise ValueError("Invalid noise type")
        

        model.to(device)
        likelihood.to(device)

        model,likelihood,saved_model_name,saved_likelihood_name,epoch = load_best_model_and_epoch(model,likelihood,args,device,folder)
        if args.nll_type == 'VariationalELBO':
            mll_test = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))
        elif args.nll_type == 'PredictiveLogLikelihood':
            mll_test = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model.gp_layer, num_data=len(train_loader.dataset), beta=0.5)
            
        criterion = nn.BCELoss()

        model.eval()
        likelihood.eval()

        if args.addnoise == 'add3sigmas3mus':
            # add the same noise for all the test images
            likelihood.noise = torch.tensor([sigma,sigma,sigma,mu,mu,mu])
        elif args.addnoise == 'add3sigmas':
            likelihood.noise = torch.tensor([sigma,sigma,sigma])

        with torch.no_grad():
            for i_sub in sub_list_train_val:

                image_name = os.path.join(image_path,i_sub)
                itk_image = sitk.ReadImage(image_name)
                image = sitk.GetArrayFromImage(itk_image)
                image = torch.from_numpy(image).float().to(device)

                # generate predictions every two slices, and then cated them together

                if args.pre_labels == 'co':
                    raise ValueError("co is not supported for grid sample")
                
                elif args.pre_labels == 'random3':
                    if args.addnoise == 'add' or args.addnoise == 'noadd':
                        raise ValueError("add and noadd are not supported for grid sample")
                    
                    elif args.addnoise == 'add3sigmas' or args.addnoise == 'add3sigmas3mus':
                        
                        if image.shape[0]>=2:
                            for i in range(0,image.shape[0]-1,2):
                                image1 = image[i:i+2]
                                preds_sub = model(image1[:,None,...])

                                selected_observation = 0*torch.ones(image1.shape[0]).to(device).long()
                                output_addnoise = likelihood(preds_sub,args.addnoise,selected_observation)
                                binary_output = output_addnoise.mean.ge(0.5).float()
                                binary_output = binary_output.reshape(image1.shape[0],image1.shape[1],image1.shape[2])

                                if i == 0:
                                    binary_output_all = binary_output
                                else:
                                    binary_output_all = torch.cat((binary_output_all,binary_output),dim=0)

                            if image.shape[0] % 2 != 0:
                                image1 = image[i+2][None,...]
                                preds_sub = model(image1[:,None,...])

                                selected_observation = 0*torch.ones(image1.shape[0]).to(device).long()
                                output_addnoise = likelihood(preds_sub,args.addnoise,selected_observation)
                                binary_output = output_addnoise.mean.ge(0.5).float()
                                binary_output = binary_output.reshape(image1.shape[0],image1.shape[1],image1.shape[2])
                                binary_output_all = torch.cat((binary_output_all,binary_output),dim=0)

                        elif image.shape[0] == 1:
                            image1 = image
                            preds = model(image1[:,None,...])
                            selected_observation = 0*torch.ones(image1.shape[0]).to(device).long()
                            output_addnoise = likelihood(preds,args.addnoise,selected_observation)
                            binary_output = output_addnoise.mean.ge(0.5).float()
                            binary_output_all = binary_output.reshape(image1.shape[0],image1.shape[1],image1.shape[2])

                        if binary_output_all.shape[0] != image.shape[0]:
                            raise ValueError("preds and image shape mismatch")
                        
                        binary_output_all = binary_output_all.cpu().numpy().astype(np.uint8)

                
                        nifti_mask = nib.Nifti1Image(np.transpose(binary_output_all,(2,1,0)), np.eye(4))
                        nib.save(nifti_mask, os.path.join(saved_label_path,"label_"+i_sub))



       







    



    

  