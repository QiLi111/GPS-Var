
# Deep Kernel Learning for Gaussian Process using GPytorch
# Uisng simulated labels for training; only optimise baise and variance

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
from utils.train_DKL_simulated import train_DKL_simulated
from utils.BernoulliLikelihood_with_noise import BernoulliLikelihood_with_Noise

parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
parser.add_argument("--dataset", type=str,default="./data")
parser.add_argument("--num_epochs", type=int, default=500000)
parser.add_argument('--num_class', type=int, default=1, help='The size of class')
parser.add_argument("--batch_size_train", type=int, default=8)
parser.add_argument("--batch_size_val", type=int, default=2)

# parser.add_argument("--loss_type", type=str, default="dice_loss") # 'dice_loss' # likelihood # "few_shot"
# 'random4': randomly select from three avaliable labels + one voted & corrected labels (which is called high quality label);
# 'random3': randomly select from three avaliable labels; 
# 'co: using only consistent label in GP train and GP loss calculation' # all: all three labels for GP train, the consistent label for cross entropy loss calculation
# 'simulated_ind: using simulated labels for training' 
parser.add_argument("--labels", type=str, default="simulated_ind") # 
parser.add_argument("--sigma_simulated_label_ind", type=float, default=100)
parser.add_argument("--mu_simulated_label_ind", type=float, default=-3)
# 'voted_only': using only voted label for inference; 'high_quality': using corrected voted labels for inference
parser.add_argument("--labels_infer", type=str, default="high_quality") 
parser.add_argument("--VariationalStrategy", type=str, default="InducingPts") # 'Grid' # 'InducingPts' 
parser.add_argument("--num_indu", type=int, default=500) #number of inducing points
parser.add_argument("--InducingPtsType", type=str, default="uniform") # 'fixed' # 'uniform'
parser.add_argument("--grid_size", type=int, default=2) 
parser.add_argument("--feat_dim", type=int, default=64)
parser.add_argument("--FeatBatchNorm", type=str, default="noadd") # 'add' # 'noadd'
parser.add_argument("--loss_type", type=str, default="dice_BCE") # dice; BCE; dice_BCE; dice_BCE_voted;dice_voted  # likelihood
# parser.add_argument("--load_pretrained_Unet", type=str, default="noload") # 'load' # 'noload'
# parser.add_argument("--trained_Unet", type=str, default="trainU") # 'trainU' # 'notrainU'
parser.add_argument("--hyperparameter", type=str, default="optimize") # 'fixed' # 'optimize'
parser.add_argument("--data_aug", type=str, default="noadd") # 'add' # 'noadd'; if add data augmentation
# 'add': add only one sigma for three labels; 
# 'add3sigmas': add three different sigmas; 
# 'add3sigmas3mus': add three different sigmas and three different mus
# 'noadd'; no noise added
parser.add_argument("--addnoise", type=str, default="add_1_bias_variance") 
parser.add_argument("--addnoise_pred", type=str, default="add") # 'add' # 'noadd'; obtain the noise prediction, rather than underlying ground truth
parser.add_argument("--kernel_type", type=str, default="RBF") # 'RBF' # 'Cosine' # 'Linear' # RBF_Linear
parser.add_argument("--nll_type", type=str, default="VariationalELBO")
parser.add_argument("--retrain", type=str, default="noload") # 'load' # 'noload'
parser.add_argument("--retrain_epoch", type=int, default=320) 

parser.add_argument("--seed", type=int,default=0)
parser.add_argument("--cp_path", type=str, default="checkpoints")
parser.add_argument("--plot_path", type=str, default="visulisation")
parser.add_argument("--saved_path", type=str, default="saved_path")
parser.add_argument("--FREQ_SAVE", type=int, default=5)
parser.add_argument("--FREQ_VISULISE", type=int, default=5)
parser.add_argument("-d", "--device", dest="device", help="Device to run on, the cpu or gpu.", type=str, default="cuda:0")



args = parser.parse_args()

def main(args):
    models_folders = 'final_model_GP_test_on_corrected_labels'
    folder = 'NoPreLoad_TrainUnet_OptimizeHyper_Add3Sigmas3Mus_NoAddNoisePred_NoFeatBN_Sum2MuReg_random3__voted_only__dice_BCE__VariationalELBO__RBF__InducingPts_uniform500__FeatDim64__bst_8__bsv_8__fr_5'

    print(f"Seeding with seed: {args.seed}")
    seed_all(args.seed)
    cuda_avail, device = torch_init(args.device)
    print("pytorch using device", device)

    args = get_saved_folder_name_simulated(args)
    os.makedirs(os.path.join(args.saved_name,args.plot_path),exist_ok=True)
    os.makedirs(os.path.join(args.saved_name,args.cp_path),exist_ok=True)
    os.makedirs(os.path.join(args.saved_name,args.saved_path),exist_ok=True)
    os.makedirs(os.path.join(args.saved_name,args.saved_path,'saved_model'),exist_ok=True)
    # save the arguments into text file
    save_arguments(args)
    writer = SummaryWriter(os.path.join(args.saved_name,args.cp_path))

    # load data    
    dataset = DataSet(datapath=args.dataset,num_class=args.num_class,labels=args.labels,labels_inference = args.labels_infer,data_aug=args.data_aug,
                        simulated_model=None,sigma=args.sigma_simulated_label_ind,mu=args.mu_simulated_label_ind)
    
    # dataset.split_data()
    train_set,val_set,test_set = dataset.load_data_split(os.path.join(args.dataset,'data_split','train_list.json'),os.path.join(args.dataset,'data_split','val_list.json'),os.path.join(args.dataset,'data_split','test_list.json'))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size_val, shuffle=False, num_workers=8)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_val, shuffle=False, num_workers=8)

    # construct model
    feature_extractor = UNetFeatureExtractor(n_channels = 1, n_classes=args.num_class, feat_dim = args.feat_dim, FeatBN = args.FeatBatchNorm)
    feature_extractor = feature_extractor.to(device)
    num_features = feature_extractor.outc.conv.in_channels
    
    # strategy for scalable GP
    if args.VariationalStrategy == 'InducingPts':
        inducing_points = generate_inducing_points(feature_extractor,num_features,train_set,args,device)
        model = DKLModelInducingPts(feature_extractor, inducing_points=inducing_points,hyperparameter_fixed=args.hyperparameter, kernel_type = args.kernel_type)

    
    likelihood = BernoulliLikelihood_with_Noise(addnoise=args.addnoise)
    
    model.to(device)
    likelihood.to(device)

    model = load_best_model_and_epoch_simulated(model,likelihood,device,models_folders+'/'+folder)



    train_DKL_simulated(model, likelihood, device,args,train_loader,val_loader,writer)
    
    
#Run main
main(args)
