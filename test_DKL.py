import torch
import torch.nn as nn
import gpytorch

from utils.utils import *
from utils.base_network import UNetFeatureExtractor,DKLModelInducingPts,DKLModelGrid
from utils.BernoulliLikelihood_with_noise import BernoulliLikelihood_with_Noise
from torchmetrics.functional.classification import binary_calibration_error



def test_DKL(test_set,test_loader,args,device,folder):

    feature_extractor = UNetFeatureExtractor(n_channels = 1, n_classes=args.num_class, feat_dim = args.feat_dim, FeatBN = args.FeatBatchNorm)
    feature_extractor = feature_extractor.to(device)
    num_features = feature_extractor.outc.conv.in_channels

    # strategy for scalable GP
    if args.VariationalStrategy == 'InducingPts':
        inducing_points = generate_inducing_points(feature_extractor,num_features,test_set,args,device)
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
        # likelihood.noise = 1e-2
        likelihood.noise_covar.raw_noise.requires_grad = False
        

    model.to(device)
    likelihood.to(device)

    model,likelihood,saved_model_name,saved_likelihood_name,epoch = load_best_model_and_epoch(model,likelihood,args,device,folder)
    
    if args.nll_type == 'VariationalELBO':
        mll_test = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(test_loader.dataset))
    elif args.nll_type == 'PredictiveLogLikelihood':
        mll_test = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model.gp_layer, num_data=len(test_loader.dataset), beta=0.5)
    
    criterion = nn.BCELoss()

    metrics_all,hyperpara_all,metrics_addnoise_all  = test(model,likelihood,test_loader,mll_test,device,epoch,args, criterion)

    return metrics_all,hyperpara_all,metrics_addnoise_all,saved_model_name,saved_likelihood_name


def test(model,likelihood,test_loader,mll,device,epoch,args, criterion):
    model.eval()
    likelihood.eval()

    metrics_vote,hyperpara_vote,metrics_addnoise_vote,index_non_empty_vote = initialise_metrics()
    metrics_0,hyperpara_0,metrics_addnoise_0,index_non_empty_0 = initialise_metrics()
    metrics_1,hyperpara_1,metrics_addnoise_1,index_non_empty_1 = initialise_metrics()
    metrics_2,hyperpara_2,metrics_addnoise_2,index_non_empty_2 = initialise_metrics()


    metrics_0_voted,hyperpara_0_voted,metrics_addnoise_0_voted,index_non_empty_0_voted = initialise_metrics()
    metrics_1_voted,hyperpara_1_voted,metrics_addnoise_1_voted,index_non_empty_1_voted = initialise_metrics()
    metrics_2_voted,hyperpara_2_voted,metrics_addnoise_2_voted,index_non_empty_2_voted = initialise_metrics()
       
    with torch.no_grad():
        for ii, (image_val, mask_all_val, mask_wo_noise, selected_observation) in enumerate(test_loader):
            
            images_val, masks_val, mask_wo_noise = image_val.float().to(device), mask_all_val.float().to(device), mask_wo_noise.float().to(device)
            
            mask_wo_noise = masks_val[:,-1,...]
            
            preds_val = model(images_val[:,None,...])

            if args.pre_labels == 'co':
                # use the noiseless voted labels for inference
                metrics_vote,hyperpara_vote,metrics_addnoise_vote,index_non_empty_vote = get_metrics_all(model,args,likelihood,preds_val,mask_wo_noise,criterion,mll,metrics_vote,hyperpara_vote,metrics_addnoise_vote,index_non_empty_vote)
            
            elif args.pre_labels == 'random4':
                if args.addnoise == 'add' or args.addnoise == 'noadd':
                    # use the noiseless voted labels for inference
                    metrics_vote,hyperpara_vote,metrics_addnoise_vote,index_non_empty_vote = get_metrics_all(model,args,likelihood,preds_val,mask_wo_noise,criterion,mll,metrics_vote,hyperpara_vote, metrics_addnoise_vote,index_non_empty_vote)
                    # compare with three noisy observation labels.
                    metrics_0,hyperpara_0, metrics_addnoise_0,index_non_empty_0 = get_metrics_all(model,args,likelihood,preds_val,masks_val[:,0,...],criterion,mll,metrics_0,hyperpara_0, metrics_addnoise_0,index_non_empty_0)
                    metrics_1,hyperpara_1, metrics_addnoise_1,index_non_empty_1 = get_metrics_all(model,args,likelihood,preds_val,masks_val[:,1,...],criterion,mll,metrics_1,hyperpara_1, metrics_addnoise_1,index_non_empty_1)
                    metrics_2,hyperpara_2, metrics_addnoise_2,index_non_empty_2 = get_metrics_all(model,args,likelihood,preds_val,masks_val[:,2,...],criterion,mll,metrics_2,hyperpara_2, metrics_addnoise_2,index_non_empty_2)

                elif args.addnoise == 'add3sigmas' or args.addnoise == 'add3sigmas3mus':
                
                    # obtain the performance metrics for the noisy prediction
                    selected_observation = 0*torch.ones(images_val.shape[0]).to(device).long()
                    metrics_0,hyperpara_0, metrics_addnoise_0,index_non_empty_0 = get_metrics_all(model,args,likelihood,preds_val,masks_val[:,0,...],criterion,mll,metrics_0,hyperpara_0, metrics_addnoise_0,index_non_empty_0,selected_observation)
                    metrics_0_voted,hyperpara_0_voted, metrics_addnoise_0_voted,index_non_empty_0_voted = get_metrics_all(model,args,likelihood,preds_val,mask_wo_noise,criterion,mll,metrics_0_voted,hyperpara_0_voted, metrics_addnoise_0_voted,index_non_empty_0_voted,selected_observation)

                    selected_observation = 1*torch.ones(images_val.shape[0]).to(device).long()
                    metrics_1,hyperpara_1, metrics_addnoise_1,index_non_empty_1 = get_metrics_all(model,args,likelihood,preds_val,masks_val[:,1,...],criterion,mll,metrics_1,hyperpara_1, metrics_addnoise_1,index_non_empty_1,selected_observation)
                    metrics_1_voted,hyperpara_1_voted, metrics_addnoise_1_voted,index_non_empty_1_voted = get_metrics_all(model,args,likelihood,preds_val,mask_wo_noise,criterion,mll,metrics_1_voted,hyperpara_1_voted, metrics_addnoise_1_voted,index_non_empty_1_voted,selected_observation)

                    selected_observation = 2*torch.ones(images_val.shape[0]).to(device).long()
                    metrics_2,hyperpara_2, metrics_addnoise_2,index_non_empty_2 = get_metrics_all(model,args,likelihood,preds_val,masks_val[:,2,...],criterion,mll,metrics_2,hyperpara_2, metrics_addnoise_2,index_non_empty_2,selected_observation)
                    metrics_2_voted,hyperpara_2_voted, metrics_addnoise_2_voted,index_non_empty_2_voted = get_metrics_all(model,args,likelihood,preds_val,mask_wo_noise,criterion,mll,metrics_2_voted,hyperpara_2_voted, metrics_addnoise_2_voted,index_non_empty_2_voted,selected_observation)

                    # obtain the performance metrics for the noiseless prediction
                    addnoise_copy = args.addnoise
                    args.addnoise = 'noadd'
                    metrics_vote,hyperpara_vote,metrics_addnoise_vote,index_non_empty_vote = get_metrics_all(model,args,likelihood,preds_val,mask_wo_noise,criterion,mll,metrics_vote,hyperpara_vote, metrics_addnoise_vote,index_non_empty_vote)
                    args.addnoise = addnoise_copy
            
            else:
                raise ValueError("Invalid pre_labels or noise type")
        
    if args.pre_labels == 'co':        
        metrics_vote,hyperpara_vote, metrics_addnoise_vote = list_to_numpy(metrics_vote,hyperpara_vote, metrics_addnoise_vote,index_non_empty_vote,args)
        return metrics_vote,hyperpara_vote,metrics_addnoise_vote
    
    elif args.pre_labels == 'random4':
        metrics_vote,hyperpara_vote, metrics_addnoise_vote = list_to_numpy(metrics_vote,hyperpara_vote,metrics_addnoise_vote,index_non_empty_vote,args)
        metrics_0,hyperpara_0, metrics_addnoise_0 = list_to_numpy(metrics_0,hyperpara_0, metrics_addnoise_0,index_non_empty_0,args)
        metrics_1,hyperpara_1, metrics_addnoise_1 = list_to_numpy(metrics_1,hyperpara_1, metrics_addnoise_1,index_non_empty_1,args)
        metrics_2,hyperpara_2, metrics_addnoise_2 = list_to_numpy(metrics_2,hyperpara_2, metrics_addnoise_2,index_non_empty_2,args)
        
        if len(metrics_addnoise_0_voted['dice'])>0:
            metrics_0_voted,hyperpara_0_voted, metrics_addnoise_0_voted = list_to_numpy(metrics_0_voted,hyperpara_0_voted, metrics_addnoise_0_voted,index_non_empty_0_voted,args)
            metrics_1_voted,hyperpara_1_voted, metrics_addnoise_1_voted = list_to_numpy(metrics_1_voted,hyperpara_1_voted, metrics_addnoise_1_voted,index_non_empty_1_voted,args)
            metrics_2_voted,hyperpara_2_voted, metrics_addnoise_2_voted = list_to_numpy(metrics_2_voted,hyperpara_2_voted, metrics_addnoise_2_voted,index_non_empty_2_voted,args)
        
            metrics_all = {'vote':metrics_vote,'0':metrics_0,'1':metrics_1,'2':metrics_2,'0_voted':metrics_0_voted,'1_voted':metrics_1_voted,'2_voted':metrics_2_voted}
            hyperpara_all = {'vote':hyperpara_vote,'0':hyperpara_0,'1':hyperpara_1,'2':hyperpara_2,'0_voted':hyperpara_0_voted,'1_voted':hyperpara_1_voted,'2_voted':hyperpara_2_voted}
            metrics_addnoise_all = {'vote':metrics_addnoise_vote,'0':metrics_addnoise_0,'1':metrics_addnoise_1,'2':metrics_addnoise_2,'0_voted':metrics_addnoise_0_voted,'1_voted':metrics_addnoise_1_voted,'2_voted':metrics_addnoise_2_voted}

        else:
            metrics_all = {'vote':metrics_vote,'0':metrics_0,'1':metrics_1,'2':metrics_2}
            hyperpara_all = {'vote':hyperpara_vote,'0':hyperpara_0,'1':hyperpara_1,'2':hyperpara_2}
            metrics_addnoise_all = {'vote':metrics_addnoise_vote,'0':metrics_addnoise_0,'1':metrics_addnoise_1,'2':metrics_addnoise_2}

        return  metrics_all,hyperpara_all,metrics_addnoise_all 
    