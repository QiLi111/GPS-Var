import torch
import torch.nn as nn
import gpytorch

from utils.utils import *
from utils.base_network import UNetFeatureExtractor,DKLModelInducingPts,DKLModelGrid
from utils.BernoulliLikelihood_with_noise import BernoulliLikelihood_with_Noise
from torchmetrics.functional.classification import binary_calibration_error
import pickle



def test_DKL_grid_sample(test_set,test_loader,args,device,folder,saved_path):

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
        likelihood = BernoulliLikelihood_with_Noise(addnoise=args.addnoise, num_annotators = args.num_annotators)
    else:
        raise ValueError("Invalid noise type")
    
    
        

    model.to(device)
    likelihood.to(device)

    model,likelihood,saved_model_name,saved_likelihood_name,epoch = load_best_model_and_epoch(model,likelihood,args,device,folder)
    if args.nll_type == 'VariationalELBO':
        mll_test = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(test_loader.dataset))
    elif args.nll_type == 'PredictiveLogLikelihood':
        mll_test = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model.gp_layer, num_data=len(test_loader.dataset), beta=0.5)
        
    criterion = nn.BCELoss()

    all_metrics, sigma_all, mu_all  = test_grid_sample(model,likelihood,test_loader,mll_test,device,epoch,args, criterion,saved_path,test_set)

    return all_metrics, sigma_all, mu_all


def test_grid_sample(model,likelihood,test_loader,mll,device,epoch,args, criterion,saved_path,test_set):
    model.eval()
    likelihood.eval()

    os.makedirs(os.path.join(saved_path,'dice'),exist_ok=True)
    os.makedirs(os.path.join(saved_path,'hd95'),exist_ok=True)
    os.makedirs(os.path.join(saved_path,'ece_addnoise'),exist_ok=True)
    os.makedirs(os.path.join(saved_path,'nll'),exist_ok=True)
    os.makedirs(os.path.join(saved_path,'img'),exist_ok=True)

    sigma_all = torch.linspace(0.1, 3, 128)
    mu_all = torch.linspace(-3, 3, 128)

    # sigma_all = torch.linspace(0.1, 3, 12)
    # mu_all = torch.linspace(-3, 3, 12)
    
       
    with torch.no_grad():
        for ii, (image_val, mask_all_val, mask_wo_noise, selected_observation) in enumerate(test_loader):   
           
            all_metrics = {}
            images_val, mask_all_val, mask_wo_noise = image_val.float().to(device), mask_all_val.float().to(device), mask_wo_noise.float().to(device)
            preds_val = model(images_val[:,None,...])

            # get predictions

            # # obtain the performance metrics for the noisy prediction
            selected_observation = 0*torch.ones(images_val.shape[0]).to(device).long()
            output_addnoise_0 = likelihood(preds_val,args.addnoise,selected_observation,args.num_annotators,len(selected_observation))
            binary_output_0 = output_addnoise_0.mean.ge(0.5).float()
 
            selected_observation = 1*torch.ones(images_val.shape[0]).to(device).long()
            output_addnoise_1 = likelihood(preds_val,args.addnoise,selected_observation,args.num_annotators,len(selected_observation))
            binary_output_1 = output_addnoise_1.mean.ge(0.5).float()
 
            selected_observation = 2*torch.ones(images_val.shape[0]).to(device).long()
            output_addnoise_2 = likelihood(preds_val,args.addnoise,selected_observation,args.num_annotators,len(selected_observation))
            binary_output_2 = output_addnoise_2.mean.ge(0.5).float()
      

            plot_img_and_annotations(image_val,mask_wo_noise,mask_all_val,binary_output_0,binary_output_1,binary_output_2, os.path.join(saved_path,'img'),ii)
            
            noise_copy = likelihood.noise

            for sigma in sigma_all:
                for mu in mu_all:
                    all_metrics[(sigma.item(),mu.item())] = initialise_metrics_plot()
                    likelihood.noise = torch.tensor([sigma,sigma,sigma,mu,mu,mu])

            
                    # obtain the performance metrics for the noisy prediction
                    selected_observation = 0*torch.ones(images_val.shape[0]).to(device).long()
                    
                    all_metrics[(sigma.item(),mu.item())] = get_metrics_all_each_image(args,likelihood,preds_val,mask_wo_noise,criterion,mll,all_metrics[(sigma.item(),mu.item())],selected_observation)                
                    all_metrics[(sigma.item(),mu.item())] = list_to_numpy_plot(all_metrics[(sigma.item(),mu.item())])

            
            likelihood.noise = noise_copy    
                
            metrics_plot_each_image(sigma_all,mu_all,all_metrics,'ece_addnoise',os.path.join(saved_path,'ece_addnoise'),ii)
            metrics_plot_each_image(sigma_all,mu_all,all_metrics,'nll',os.path.join(saved_path,'nll'),ii)
            metrics_plot_each_image(sigma_all,mu_all,all_metrics,'dice',os.path.join(saved_path,'dice'),ii)

            with open(saved_path +'/' + 'all_metrics_image%d.pkl' %ii, 'wb') as f:
                pickle.dump(all_metrics, f)
            
    
    return  all_metrics, sigma_all, mu_all
    