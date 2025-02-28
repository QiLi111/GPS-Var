import torch
import torch.nn as nn
import gpytorch

from utils.utils import *
from utils.base_network import UNetFeatureExtractor,DKLModelInducingPts,DKLModelGrid
from utils.BernoulliLikelihood_with_noise import BernoulliLikelihood_with_Noise
from torchmetrics.functional.classification import binary_calibration_error



def test_DKL_grid_sample(test_set,test_loader,args,device,folder,sigma,mu,saved_path):

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
    
    if args.addnoise == 'add3sigmas' or args.addnoise == 'add3sigmas3mus':
        likelihood = BernoulliLikelihood_with_Noise(addnoise=args.addnoise)
    else:
        raise ValueError("Invalid noise type")
    
    # if args.hyperparameter == 'fixed' and args.addnoise == 'add':
    #     # likelihood.noise = 1e-2
    #     likelihood.noise_covar.raw_noise.requires_grad = False
        

    model.to(device)
    likelihood.to(device)

    model,likelihood,saved_model_name,saved_likelihood_name,epoch = load_best_model_and_epoch(model,likelihood,args,device,folder)
    if args.nll_type == 'VariationalELBO':
        mll_test = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(test_loader.dataset))
    elif args.nll_type == 'PredictiveLogLikelihood':
        mll_test = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model.gp_layer, num_data=len(test_loader.dataset), beta=0.5)
        
    criterion = nn.BCELoss()

    metrics_all,hyperpara_all,metrics_addnoise_all  = test_grid_sample(model,likelihood,test_loader,mll_test,device,epoch,args, criterion,sigma,mu,saved_path,test_set)

    return metrics_all,hyperpara_all,metrics_addnoise_all,saved_model_name,saved_likelihood_name


def test_grid_sample(model,likelihood,test_loader,mll,device,epoch,args, criterion,sigma,mu,saved_path,test_set):
    model.eval()
    likelihood.eval()
    metrics_0,hyperpara_0,metrics_addnoise_0,index_non_empty_0 = initialise_metrics()

    if args.addnoise == 'add3sigmas3mus':
        # add the same noise for all the test images
        likelihood.noise = torch.tensor([sigma,sigma,sigma,mu,mu,mu])
    elif args.addnoise == 'add3sigmas':
        likelihood.noise = torch.tensor([sigma,sigma,sigma])
       
    with torch.no_grad():
        for ii, (image_val, mask_all_val, mask_wo_noise, selected_observation) in enumerate(test_loader):   
            # if ii<10:
            #     continue
            # elif ii > 10:
            #     break
            images_val, masks_val, mask_wo_noise = image_val.float().to(device), mask_all_val.float().to(device), mask_wo_noise.float().to(device)
            preds_val = model(images_val[:,None,...])

            if args.pre_labels == 'co':
                raise ValueError("co is not supported for grid sample")
            
            elif args.pre_labels == 'random3':
                if args.addnoise == 'add' or args.addnoise == 'noadd':
                    raise ValueError("add and noadd are not supported for grid sample")
                
                elif args.addnoise == 'add3sigmas' or args.addnoise == 'add3sigmas3mus':
                    # obtain the performance metrics for the noisy prediction
                    selected_observation = 0*torch.ones(images_val.shape[0]).to(device).long()
                    metrics_0,hyperpara_0, metrics_addnoise_0,index_non_empty_0 = get_metrics_all(model,args,likelihood,preds_val,mask_wo_noise,criterion,mll,metrics_0,hyperpara_0, metrics_addnoise_0,index_non_empty_0,selected_observation)                
                    
                    # # visualize the noisy prediction
                    if ii == 10:
                        output_addnoise = likelihood(preds_val,args.addnoise,selected_observation) 
                        ploting(output_addnoise,saved_path,images_val,mask_wo_noise,sigma,mu)
                
                else:
                    raise ValueError("Invalid pre_labels or noise type")
            else:
                raise ValueError("Invalid pre_labels or noise type")
        
    
    if args.pre_labels == 'random3':
        metrics_0,hyperpara_0, metrics_addnoise_0 = list_to_numpy(metrics_0,hyperpara_0, metrics_addnoise_0,index_non_empty_0,args)
                
    else:
        raise ValueError("Invalid pre_labels or noise type")
    
    return  metrics_0,hyperpara_0,metrics_addnoise_0 
    