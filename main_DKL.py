
import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
import gpytorch
env_path = os.path.join(os.path.dirname(__file__), '..') #This may or may not work hehe
sys.path.append(env_path)
from utils.utils import *
from utils.data_loader import DataSet
from utils.base_network import UNetFeatureExtractor,DKLModelInducingPts,DKLModelGrid
from train_DKL import train_DKL
from utils.BernoulliLikelihood_with_noise import BernoulliLikelihood_with_Noise

parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
parser.add_argument("--dataset", type=str,default="./data") 
parser.add_argument("--num_annotators", type=int,default=3) 
parser.add_argument("--num_epochs", type=int, default=500000)
parser.add_argument('--num_class', type=int, default=1, help='The size of class')
parser.add_argument("--batch_size_train", type=int, default=2)
parser.add_argument("--batch_size_val", type=int, default=2)

# 'random4': randomly select from three avaliable labels + one corrected label (which is called high quality label);
parser.add_argument("--labels", type=str, default="random4") # 
# "seg_1_2_3_HQ": use three avaliable labels and the high quality label for inference
parser.add_argument("--labels_infer", type=str, default="seg_1_2_3_HQ") 
parser.add_argument("--VariationalStrategy", type=str, default="InducingPts") # 'Grid' # 'InducingPts' 
parser.add_argument("--num_indu", type=int, default=500) #number of inducing points
parser.add_argument("--InducingPtsType", type=str, default="uniform") # 'fixed' # 'uniform'
parser.add_argument("--grid_size", type=int, default=2) 
parser.add_argument("--feat_dim", type=int, default=64)
parser.add_argument("--FeatBatchNorm", type=str, default="noadd") # 'add' # 'noadd'
parser.add_argument("--loss_type", type=str, default="dice_BCE_kl") # dice_BCE_kl; dice; BCE; dice_BCE; dice_BCE_voted;dice_voted  # likelihood
parser.add_argument("--loss_add_mu_reg", type=str, default="sum_2") # 'add' # 'noadd' # sum_2
parser.add_argument("--load_pretrained_Unet", type=str, default="noload") # 'load' # 'noload'
parser.add_argument("--trained_Unet", type=str, default="trainU") # 'trainU' # 'notrainU'
parser.add_argument("--hyperparameter", type=str, default="optimize") # 'fixed' # 'optimize'
parser.add_argument("--data_aug", type=str, default="noadd") # 'add' # 'noadd'; if add data augmentation
# 'add': add only one sigma for three labels; 
# 'add3sigmas': add three different sigmas; 
# 'add3sigmas3mus': add three different sigmas and three different mus
# 'noadd'; no noise added
parser.add_argument("--addnoise", type=str, default="add3sigmas3mus") 
parser.add_argument("--addnoise_pred", type=str, default="noadd") # 'add' # 'noadd'; obtain the noise prediction, rather than underlying ground truth
parser.add_argument("--kernel_type", type=str, default="RBF") # 'RBF' # 'Cosine' # 'Linear' # RBF_Linear
parser.add_argument("--nll_type", type=str, default="VariationalELBO")
parser.add_argument("--retrain", type=str, default="noload") # 'load' # 'noload'
parser.add_argument("--retrain_epoch", type=int, default=0) 

parser.add_argument("--seed", type=int,default=0)
parser.add_argument("--cp_path", type=str, default="checkpoints")
parser.add_argument("--plot_path", type=str, default="visulisation")
parser.add_argument("--saved_path", type=str, default="saved_path")
parser.add_argument("--FREQ_SAVE", type=int, default=5)
parser.add_argument("--FREQ_VISULISE", type=int, default=5)
parser.add_argument("-d", "--device", dest="device", help="Device to run on, the cpu or gpu.", type=str, default="cuda:0")



args = parser.parse_args()

def main(args):
    print(f"Seeding with seed: {args.seed}")
    seed_all(args.seed)
    cuda_avail, device = torch_init(args.device)
    print("pytorch using device", device)

    args = get_saved_folder_name(args)
    os.makedirs(os.path.join(args.saved_name,args.plot_path),exist_ok=True)
    os.makedirs(os.path.join(args.saved_name,args.cp_path),exist_ok=True)
    os.makedirs(os.path.join(args.saved_name,args.saved_path),exist_ok=True)
    os.makedirs(os.path.join(args.saved_name,args.saved_path,'saved_model'),exist_ok=True)
    # save the arguments into text file
    save_arguments(args)
    writer = SummaryWriter(os.path.join(args.saved_name,args.cp_path))

    # load data    
    dataset = DataSet(datapath=args.dataset,num_class=args.num_class,labels=args.labels,labels_inference = args.labels_infer,data_aug=args.data_aug)
    # dataset.split_data()
    train_set,val_set,test_set = dataset.load_data_split(os.path.join(args.dataset,'data_split','train_list.json'),os.path.join(args.dataset,'data_split','val_list.json'),os.path.join(args.dataset,'data_split','test_list.json'))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size_val, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_val, shuffle=False, num_workers=0)

    # construct model
    feature_extractor = UNetFeatureExtractor(n_channels = 1, n_classes=args.num_class, feat_dim = args.feat_dim, FeatBN = args.FeatBatchNorm)
    feature_extractor = feature_extractor.to(device)
    if args.load_pretrained_Unet == 'load' and args.FeatBatchNorm == 'noadd':
        model_name = os.listdir('PretrainedUnet_woFeatBN')[0]
        feature_extractor.load_state_dict(torch.load(os.path.join('PretrainedUnet_woFeatBN',model_name),map_location=torch.device(device)))
    num_features = feature_extractor.outc.conv.in_channels
    
    # strategy for scalable GP
    if args.VariationalStrategy == 'InducingPts':
        inducing_points = generate_inducing_points(feature_extractor,num_features,train_set,args,device)
        model = DKLModelInducingPts(feature_extractor, inducing_points=inducing_points,hyperparameter_fixed=args.hyperparameter, kernel_type = args.kernel_type)

    # grid is not used in the project
    elif args.VariationalStrategy == 'Grid':
        model = DKLModelGrid(feature_extractor,input_dim = num_features,grid_size = args.grid_size)
    
    if args.addnoise == 'add' or args.addnoise == 'add3sigmas' or args.addnoise == 'add3sigmas3mus':
        likelihood = BernoulliLikelihood_with_Noise(addnoise=args.addnoise, num_annotators = args.num_annotators)
    
    elif args.addnoise == 'noadd':
        likelihood = gpytorch.likelihoods.BernoulliLikelihood()
    else:
        raise ValueError("Invalid noise type")
    
    if args.hyperparameter == 'fixed' and args.addnoise == 'add':
        likelihood.noise = 1e-2
        likelihood.noise_covar.raw_noise.requires_grad = False

    elif args.hyperparameter == 'fixed' and args.addnoise == 'add3sigmas':
        likelihood.noise = torch.tensor([1e-2,1e-1,1e-2])
        likelihood.noise_covar[0].raw_noise0.requires_grad = False
        likelihood.noise_covar[1].raw_noise1.requires_grad = False
        likelihood.noise_covar[2].raw_noise2.requires_grad = False
        # for param_name, param in likelihood.named_parameters():
        #     print(f'Parameter name: {param_name:42} value = {param.item()}')
    elif args.hyperparameter == 'fixed' and args.addnoise == 'add3sigmas3mus':
        likelihood.noise = torch.tensor([1e-2,1e-1,1e-2,0,0,0])
        likelihood.noise_covar[0].raw_noise0.requires_grad = False
        likelihood.noise_covar[1].raw_noise1.requires_grad = False
        likelihood.noise_covar[2].raw_noise2.requires_grad = False
        likelihood.noise_covar[3].raw_mu0.requires_grad = False
        likelihood.noise_covar[4].raw_mu1.requires_grad = False
        likelihood.noise_covar[5].raw_mu2.requires_grad = False
        

    model.to(device)
    likelihood.to(device)

    if args.retrain == 'load':
        model,likelihood = load_model(model,likelihood,args,device,args.retrain_epoch)

    train_DKL(model, likelihood, device,args,train_loader,val_loader,writer)
    
    
#Run main
main(args)
