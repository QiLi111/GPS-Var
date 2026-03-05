
import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
from torchmetrics.functional.classification import binary_calibration_error
import shutil
from monai.metrics import compute_hausdorff_distance
import gc
from datetime import datetime
import matplotlib.colors as mcolors
from skimage import measure

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def torch_init(to_device):
    cuda_avail = torch.cuda.is_available()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cpu")
    if cuda_avail and 'cuda' in to_device:
        device = torch.device(to_device)
        torch.cuda.set_device(device)

    return cuda_avail, device

def add_status(writer, epoch, loss_total_epoch_train,log_likelihood_epoch_train,dice_loss_epoch_train,dice_loss_epoch_val = None,metrics_train = None,metrics_val = None):

    writer.add_scalars('loss', {'dice_loss_train': dice_loss_epoch_train}, epoch)
    if dice_loss_epoch_val is not None:
        writer.add_scalars('loss', {'dice_loss_val': dice_loss_epoch_val}, epoch)
    
    if loss_total_epoch_train is not None and log_likelihood_epoch_train is not None:
        writer.add_scalars('loss', {'loss_total_train': loss_total_epoch_train}, epoch)
        writer.add_scalars('loss', {'log_likelihood_train_s32': log_likelihood_epoch_train['s32']}, epoch)
        writer.add_scalars('loss', {'log_likelihood_train_s16': log_likelihood_epoch_train['s16']}, epoch)

    if metrics_train is not None:
        writer.add_scalars('metrics', {'iou_train': metrics_train['iou']}, epoch)
        writer.add_scalars('metrics', {'dice_train': metrics_train['dice']}, epoch)
        writer.add_scalars('metrics', {'dice_non_empty_train': metrics_train['dice_non_empty']}, epoch)
        writer.add_scalars('metrics', {'iou_non_empty_train': metrics_train['iou_non_empty']}, epoch)

        # writer.add_scalars('metrics', {'hd_train': metrics_train['hd']}, epoch)
        # writer.add_scalars('metrics', {'hd95_train': metrics_train['hd95']}, epoch)
        writer.add_scalars('uncertainty', {'nll_train': metrics_train['nll']}, epoch)
        writer.add_scalars('uncertainty', {'ece_train': metrics_train['ece']}, epoch)

    if metrics_val is not None:
        writer.add_scalars('metrics', {'iou_val': metrics_val['iou']}, epoch)
        writer.add_scalars('metrics', {'dice_val': metrics_val['dice']}, epoch)
        writer.add_scalars('metrics', {'dice_non_empty_val': metrics_val['dice_non_empty']}, epoch)
        writer.add_scalars('metrics', {'iou_non_empty_val': metrics_val['iou_non_empty']}, epoch)
        # writer.add_scalars('metrics', {'hd_val': metrics_val['hd']}, epoch)
        # writer.add_scalars('metrics', {'hd95_val': metrics_val['hd95']}, epoch)
        writer.add_scalars('uncertainty', {'nll_val': metrics_val['nll']}, epoch)
        writer.add_scalars('uncertainty', {'ece_val': metrics_val['ece']}, epoch)
       

def add_status_GP(writer, epoch,GP_paras_epoch_train,GP_paras_epoch_val = None):
    writer.add_scalars('GP_paras', {'length_s32_train': GP_paras_epoch_train['length_s32']}, epoch)
    writer.add_scalars('GP_paras', {'scale_s32_train': GP_paras_epoch_train['scale_s32']}, epoch)
    writer.add_scalars('GP_paras', {'sigma_s32_train': GP_paras_epoch_train['sigma_s32']}, epoch)
    writer.add_scalars('GP_paras', {'length_s16_train': GP_paras_epoch_train['length_s16']}, epoch)
    writer.add_scalars('GP_paras', {'scale_s16_train': GP_paras_epoch_train['scale_s16']}, epoch)
    writer.add_scalars('GP_paras', {'sigma_s16_train': GP_paras_epoch_train['sigma_s16']}, epoch)

    if GP_paras_epoch_val is not None:
        writer.add_scalars('GP_paras', {'length_s32_val': GP_paras_epoch_val['length_s32']}, epoch)
        writer.add_scalars('GP_paras', {'scale_s32_val': GP_paras_epoch_val['scale_s32']}, epoch)
        writer.add_scalars('GP_paras', {'sigma_s32_val': GP_paras_epoch_val['sigma_s32']}, epoch)
        writer.add_scalars('GP_paras', {'length_s16_val': GP_paras_epoch_val['length_s16']}, epoch)
        writer.add_scalars('GP_paras', {'scale_s16_val': GP_paras_epoch_val['scale_s16']}, epoch)
        writer.add_scalars('GP_paras', {'sigma_s16_val': GP_paras_epoch_val['sigma_s16']}, epoch)


def add_scalars_metrics(epoch,writer,metrics_train,metrics_val):
    
    loss_epoch = metrics_train['loss']
    iou_epoch = metrics_train['iou']
    dice_epoch = metrics_train['dice']
    nll_epoch = metrics_train['nll']
    ece_epoch = metrics_train['ece']
    dice_non_empty_epoch = metrics_train['dice_non_empty']
    iou_non_empty_epoch = metrics_train['iou_non_empty']


    writer.add_scalars('loss', {'loss_train': loss_epoch}, epoch)
    writer.add_scalars('metrics', {'iou_train': iou_epoch}, epoch)
    writer.add_scalars('metrics', {'dice_train': dice_epoch}, epoch)
    writer.add_scalars('uncertainty', {'nll_train': nll_epoch}, epoch)
    writer.add_scalars('uncertainty', {'ece_train': ece_epoch}, epoch)
    writer.add_scalars('metrics', {'dice_non_empty_train': dice_non_empty_epoch}, epoch)
    writer.add_scalars('metrics', {'iou_non_empty_train': iou_non_empty_epoch}, epoch)

    if metrics_val is not None:
        loss_epoch_val = metrics_val['loss']
        iou_epoch_val = metrics_val['iou']
        dice_epoch_val = metrics_val['dice']
        nll_epoch_val = metrics_val['nll']
        ece_epoch_val = metrics_val['ece']
        dice_non_empty_epoch_val = metrics_val['dice_non_empty']
        iou_non_empty_epoch_val = metrics_val['iou_non_empty']


        writer.add_scalars('metrics', {'iou_val': iou_epoch_val}, epoch)
        writer.add_scalars('metrics', {'dice_val': dice_epoch_val}, epoch)
        writer.add_scalars('loss', {'loss_val': loss_epoch_val}, epoch)
        writer.add_scalars('uncertainty', {'nll_val': nll_epoch_val}, epoch)
        writer.add_scalars('uncertainty', {'ece_val': ece_epoch_val}, epoch)
        writer.add_scalars('metrics', {'dice_non_empty_val': dice_non_empty_epoch_val}, epoch)
        writer.add_scalars('metrics', {'iou_non_empty_val': iou_non_empty_epoch_val}, epoch)

    
def add_scalars_hyperpara_GP(epoch,writer,hyperpara_train,hyperpara_val,args):
    writer.add_scalars('GP_parameters', {'outputscale_train': hyperpara_train['outputscale']}, epoch)
    writer.add_scalars('GP_parameters', {'lengthscale_train': hyperpara_train['lengthscale']}, epoch)
    if args.addnoise == 'add':
        writer.add_scalars('GP_parameters', {'noise_train': hyperpara_train['noise']}, epoch)
    elif args.addnoise == 'add3sigmas':
        writer.add_scalars('GP_parameters', {'noise0_train': hyperpara_train['noise'][0]}, epoch)
        writer.add_scalars('GP_parameters', {'noise1_train': hyperpara_train['noise'][1]}, epoch)
        writer.add_scalars('GP_parameters', {'noise2_train': hyperpara_train['noise'][2]}, epoch)
    elif args.addnoise == 'add3sigmas3mus':
        # writer.add_scalars('GP_parameters', {'noise0_train': hyperpara_train['noise'][0]}, epoch)
        # writer.add_scalars('GP_parameters', {'noise1_train': hyperpara_train['noise'][1]}, epoch)
        # writer.add_scalars('GP_parameters', {'noise2_train': hyperpara_train['noise'][2]}, epoch)
        # writer.add_scalars('GP_parameters', {'mu0_train': hyperpara_train['noise'][3]}, epoch)
        # writer.add_scalars('GP_parameters', {'mu1_train': hyperpara_train['noise'][4]}, epoch)
        # writer.add_scalars('GP_parameters', {'mu2_train': hyperpara_train['noise'][5]}, epoch)

        num_ann = args.num_annotators
        noise_vals = hyperpara_train['noise']

        log_dict = {}

        for i in range(num_ann):
            log_dict[f'noise{i}_train'] = noise_vals[i]

        for i in range(num_ann):
            log_dict[f'mu{i}_train'] = noise_vals[num_ann + i]

        writer.add_scalars('GP_parameters', log_dict, epoch)




    elif args.addnoise == 'add_1_bias_variance':
        writer.add_scalars('GP_parameters', {'noise0_train': hyperpara_train['noise'][0]}, epoch)
        writer.add_scalars('GP_parameters', {'mu0_train': hyperpara_train['noise'][1]}, epoch)


    if hyperpara_val is not None:
        writer.add_scalars('GP_parameters', {'outputscale_val': hyperpara_val['outputscale']}, epoch)
        writer.add_scalars('GP_parameters', {'lengthscale_val': hyperpara_val['lengthscale']}, epoch)
        if args.addnoise == 'add':
            writer.add_scalars('GP_parameters', {'noise_val': hyperpara_val['noise']}, epoch)   
        elif args.addnoise == 'add3sigmas':
            writer.add_scalars('GP_parameters', {'noise0_val': hyperpara_val['noise'][0]}, epoch)
            writer.add_scalars('GP_parameters', {'noise1_val': hyperpara_val['noise'][1]}, epoch)
            writer.add_scalars('GP_parameters', {'noise2_val': hyperpara_val['noise'][2]}, epoch)
        elif args.addnoise == 'add3sigmas3mus':
            # writer.add_scalars('GP_parameters', {'noise0_val': hyperpara_val['noise'][0]}, epoch)
            # writer.add_scalars('GP_parameters', {'noise1_val': hyperpara_val['noise'][1]}, epoch)
            # writer.add_scalars('GP_parameters', {'noise2_val': hyperpara_val['noise'][2]}, epoch)
            # writer.add_scalars('GP_parameters', {'mu0_val': hyperpara_val['noise'][3]}, epoch)
            # writer.add_scalars('GP_parameters', {'mu1_val': hyperpara_val['noise'][4]}, epoch)
            # writer.add_scalars('GP_parameters', {'mu2_val': hyperpara_val['noise'][5]}, epoch)

            noise_vals = hyperpara_val['noise']

            log_dict = {}

            for i in range(num_ann):
                log_dict[f'noise{i}_val'] = noise_vals[i]

            for i in range(num_ann):
                log_dict[f'mu{i}_val'] = noise_vals[num_ann + i]

            writer.add_scalars('GP_parameters', log_dict, epoch)


        elif args.addnoise == 'add_1_bias_variance':
            writer.add_scalars('GP_parameters', {'noise0_val': hyperpara_val['noise'][0]}, epoch)
            writer.add_scalars('GP_parameters', {'mu0_val': hyperpara_val['noise'][1]}, epoch)
    
def add_scalars_hyperpara_linear_model(epoch,writer,hyperpara_train,hyperpara_val):
    
    writer.add_scalars('Linear_parameters', {'noise0_train': hyperpara_train['noise0']}, epoch)
    writer.add_scalars('Linear_parameters', {'noise1_train': hyperpara_train['noise1']}, epoch)
    writer.add_scalars('Linear_parameters', {'noise2_train': hyperpara_train['noise2']}, epoch)
    writer.add_scalars('Linear_parameters', {'mu0_train': hyperpara_train['mu0']}, epoch)
    writer.add_scalars('Linear_parameters', {'mu1_train': hyperpara_train['mu1']}, epoch)
    writer.add_scalars('Linear_parameters', {'mu2_train': hyperpara_train['mu2']}, epoch)

    if hyperpara_val is not None:
        
        writer.add_scalars('Linear_parameters', {'noise0_val': hyperpara_val['noise0']}, epoch)
        writer.add_scalars('Linear_parameters', {'noise1_val': hyperpara_val['noise1']}, epoch)
        writer.add_scalars('Linear_parameters', {'noise2_val': hyperpara_val['noise2']}, epoch)
        writer.add_scalars('Linear_parameters', {'mu0_val': hyperpara_val['mu0']}, epoch)
        writer.add_scalars('Linear_parameters', {'mu1_val': hyperpara_val['mu1']}, epoch)
        writer.add_scalars('Linear_parameters', {'mu2_val': hyperpara_val['mu2']}, epoch)
    


def save_regular_model(model,epoch, NUM_EPOCHS, FREQ_SAVE,SAVE_PATH):
    if epoch in range(0, NUM_EPOCHS, FREQ_SAVE):
        torch.save(model.state_dict(), os.path.join(SAVE_PATH,'saved_model','model_epoch%08d' % epoch))
        print('Model parameters saved.')

    list_dir = os.listdir(os.path.join(SAVE_PATH, 'saved_model'))
    saved_models = [i for i in list_dir if i.startswith('model_epoch')]
    if len(saved_models)>4:
        # print(saved_models)
        os.remove(os.path.join(SAVE_PATH,'saved_model',sorted(saved_models)[0]))

def save_regular_model_GP(model,likelihood,epoch, NUM_EPOCHS, FREQ_SAVE,SAVE_PATH):
    if epoch in range(0, NUM_EPOCHS, FREQ_SAVE):
        torch.save(model.state_dict(), os.path.join(SAVE_PATH,'saved_model','model_epoch%08d' % epoch))
        torch.save(likelihood.state_dict(), os.path.join(SAVE_PATH,'saved_model','likelihood_epoch%08d' % epoch))
        print('Model parameters saved.')

    list_dir = os.listdir(os.path.join(SAVE_PATH, 'saved_model'))
    saved_models = [i for i in list_dir if i.startswith('model_epoch')]
    if len(saved_models)>4:
        # print(saved_models)
        os.remove(os.path.join(SAVE_PATH,'saved_model',sorted(saved_models)[0]))
    
    saved_likelihoods = [i for i in list_dir if i.startswith('likelihood_epoch')]
    if len(saved_likelihoods)>4:
        os.remove(os.path.join(SAVE_PATH,'saved_model',sorted(saved_likelihoods)[0]))



def save_best_model_GP(model,likelihood,epoch, loss_epoch_val, best_dice, SAVE_PATH):
    if loss_epoch_val > best_dice:

        list_dir = os.listdir(os.path.join(SAVE_PATH, 'saved_model'))
        saved_models = [i for i in list_dir if i.startswith('best_dice_likelihood_epoch')]
        if len(saved_models)==1:
            # print(saved_models)
            os.remove(os.path.join(SAVE_PATH,'saved_model',saved_models[0]))
        elif len(saved_models)>1:
            print('More than one best model saved.')
            for i in saved_models:
                os.remove(os.path.join(SAVE_PATH,'saved_model',i))

        saved_models = [i for i in list_dir if i.startswith('best_dice_model_epoch')]
        if len(saved_models)==1:
            # print(saved_models)
            os.remove(os.path.join(SAVE_PATH,'saved_model',saved_models[0]))
        elif len(saved_models)>1:
            print('More than one best model saved.')
            for i in saved_models:
                os.remove(os.path.join(SAVE_PATH,'saved_model',i))
        
        best_dice = loss_epoch_val
        torch.save(model.state_dict(), os.path.join(SAVE_PATH,'saved_model','best_dice_model_epoch%08d' % epoch))
        torch.save(likelihood.state_dict(), os.path.join(SAVE_PATH,'saved_model','best_dice_likelihood_epoch%08d' % epoch))
        print('Best model parameters saved.')
        

    return best_dice

def save_best_model(model,epoch, loss_epoch_val, best_dice, SAVE_PATH):
    if loss_epoch_val > best_dice:

        list_dir = os.listdir(os.path.join(SAVE_PATH, 'saved_model'))
        saved_models = [i for i in list_dir if i.startswith('best_dice_model_epoch')]
        if len(saved_models)==1:
            # print(saved_models)
            os.remove(os.path.join(SAVE_PATH,'saved_model',saved_models[0]))
        elif len(saved_models)>1:
            print('More than one best model saved.')
            for i in saved_models:
                os.remove(os.path.join(SAVE_PATH,'saved_model',i))
        
        best_dice = loss_epoch_val
        torch.save(model.state_dict(), os.path.join(SAVE_PATH,'saved_model','best_dice_model_epoch%08d' % epoch))
        print('Best model parameters saved.')
        

    return best_dice

def compute_dice(pred, target, eps=1e-6):
    """
    Compute Dice Coefficient for binary masks.

    Args:
        pred (torch.Tensor): Predicted mask of shape (B, H, W).
        target (torch.Tensor): Ground truth mask of shape (B, H, W).
        eps (float): Small value to avoid division by zero.

    Returns:
        tuple: Dice coefficient (torch.Tensor) for each (B,).
            Shape of each output: (B,).
    """
    
    
    # ensure binary masks
    # if torch.unique(pred).size(0) > 2 or torch.unique(target).size(0) > 2:
    #     raise ValueError("Predicted and target masks must be binary")

    # Flatten masks across spatial dimensions (H, W)
    pred = pred.view(pred.size(0), -1)  # Shape: (B, H*W)
    target = target.view(target.size(0), -1)  # Shape: (B, H*W)

    # Compute intersection and union
    intersection = (pred * target).sum(dim=1)  # Shape: (B)
    pred_sum = pred.sum(dim=1)  # Predicted region sum: (B)
    target_sum = target.sum(dim=1)  # Ground truth region sum: (B)

    # Compute Dice coefficient
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)

    return dice

def compute_iou(pred, target, eps=1e-6):
    """
    Compute IoU for binary masks.

    Args:
        pred (torch.Tensor): Predicted mask of shape (B, H, W).
        target (torch.Tensor): Ground truth mask of shape (B, H, W).
        eps (float): Small value to avoid division by zero.

    Returns:
        tuple: Dice coefficient (torch.Tensor) and IoU (torch.Tensor) for each (B,).
            Shape of each output: (B,).
    """
    
    
    # ensure binary masks
    # if torch.unique(pred).size(0) > 2 or torch.unique(target).size(0) > 2:
    #     raise ValueError("Predicted and target masks must be binary")

    # Flatten masks across spatial dimensions (H, W)
    pred = pred.view(pred.size(0), -1)  # Shape: (B, H*W)
    target = target.view(target.size(0), -1)  # Shape: (B, H*W)

    # Compute intersection and union
    intersection = (pred * target).sum(dim=1)  # Shape: (B)
    union = (pred + target).clamp(0, 1).sum(dim=1)  # Shape: (B)

    # Compute IoU
    iou = (intersection + eps) / (union + eps)

    return iou


def compute_hausdorff_distances(pred, target):
    """
    Compute both Hausdorff Distance and 95th Percentile Hausdorff Distance
    for binary segmentation masks using PyTorch.
    
    Args:
        pred (torch.Tensor): Predicted masks of shape (B, Q, 1, H, W).
        target (torch.Tensor): Ground truth masks of shape (B, Q, 1, H, W).
        
    Returns:
        tuple: (Hausdorff distances, 95th percentile Hausdorff distances),
            each of shape (B, Q).
    """
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    # pred = pred.cpu()
    # target = target.cpu()
    # B, H, W = pred.shape

    
            
    # hd_distances = []  # Hausdorff distances
    # hd95_distances = []  # 95th percentile distances

    # for b in range(B):
    #     # Get foreground indices as tensors
    #     pred_indices = torch.nonzero(pred[b], as_tuple=False).float()
    #     target_indices = torch.nonzero(target[b], as_tuple=False).float()

    #     if pred_indices.numel() == 0 and target_indices.numel() == 0:
    #         # # Handle empty masks (either predicted or ground truth)
    #         # hd_distances[b] = float(0)
    #         # hd95_distances[b] = float(0)
    #         continue
    #     elif pred_indices.numel() == 0 and target_indices.numel() != 0:
    #         # # Handle empty predicted mask
    #         # if target_indices.shape[0] < 2:
    #         #     return float(0)
            
    #         # dists = torch.cdist(target_indices, target_indices, p=2)
    #         # d1_max = torch.max(torch.min(dists, dim=1).values)
    #         # d2_max = torch.max(torch.min(dists, dim=0).values)
    #         # hd_distances[b] = torch.max(d1_max, d2_max)

    #         # d1_95 = torch.quantile(torch.min(dists, dim=1).values, 0.95) 
    #         # d2_95 = torch.quantile(torch.min(dists, dim=0).values, 0.95)
    #         # hd95_distances[b] = torch.max(d1_95, d2_95)

    #         continue
    #     elif pred_indices.numel() != 0 and target_indices.numel() == 0:
    #         # # Handle empty ground truth mask
    #         # if pred_indices.shape[0] < 2:
    #         #     return float(0)
            
    #         # dists = torch.cdist(pred_indices, pred_indices, p=2)
    #         # d1_max = torch.max(torch.min(dists, dim=1).values)
    #         # d2_max = torch.max(torch.min(dists, dim=0).values)
    #         # hd_distances[b] = torch.max(d1_max, d2_max)

    #         # d1_95 = torch.quantile(torch.min(dists, dim=1).values, 0.95) 
    #         # d2_95 = torch.quantile(torch.min(dists, dim=0).values, 0.95)
    #         # hd95_distances[b] = torch.max(d1_95, d2_95)

    #         continue

    #     # Compute pairwise distances
    #     pred_dists = torch.cdist(pred_indices, target_indices, p=2)  # Euclidean distances
    #     target_dists = torch.cdist(target_indices, pred_indices, p=2)  # Reverse distances

    #     # Hausdorff distance (maximum distance in either direction)
    #     d1_max = torch.max(torch.min(pred_dists, dim=1).values)  # Max over pred -> target
    #     d2_max = torch.max(torch.min(target_dists, dim=1).values)  # Max over target -> pred
    #     hd_distances.append(torch.max(d1_max, d2_max).item())

    #     # 95th percentile Hausdorff distance
    #     d1_95 = torch.quantile(torch.min(pred_dists, dim=1).values, 0.95)  # 95th percentile
    #     d2_95 = torch.quantile(torch.min(target_dists, dim=1).values, 0.95)  # 95th percentile
    #     hd95_distances.append(torch.max(d1_95, d2_95).item())

    
    # hd_distances = torch.tensor(hd_distances)
    # hd95_distances = torch.tensor(hd95_distances)
    # return hd_distances, hd95_distances

    with torch.no_grad():
        hd_distances = compute_hausdorff_distance(pred.unsqueeze(1),target.unsqueeze(1), spacing=(0.18,0.16))
        hd95_distances = compute_hausdorff_distance(pred.unsqueeze(1),target.unsqueeze(1),percentile=95, spacing=(0.18,0.16))
    
    gc.collect()

    return hd_distances[hd_distances.isfinite()], hd95_distances[hd95_distances.isfinite()]

def visualize(images, masks, prediction,epoch,step,args):
    visualization_path = os.path.join(args.saved_name,args.plot_path)
        
    if epoch in range(0, args.num_epochs, args.FREQ_VISULISE) and step in range(0, args.num_epochs, 50):
        # os.makedirs(os.path.join(visualization_path,str(epoch)),exist_ok=True)
        if step ==0:
            try:
                shutil.rmtree(os.path.join(visualization_path,'best_dice'))
            except:
                pass
        os.makedirs(os.path.join(visualization_path,'best_dice'),exist_ok=True)

        images, masks = images.cpu().detach().numpy(), masks.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        
        for i in range(images.shape[0]):

            plt.imshow(images[i,...],'gray')
            plt.contour(masks[i,...],colors='red')
            plt.contour(prediction[i,...],colors='blue')
            # plt.imshow(prediction[i,...],'gray')

            # plt.savefig(os.path.join(visualization_path,str(epoch), f"epoch_{epoch}_step_{step}_no_{i}.png"))
            plt.savefig(os.path.join(visualization_path,'best_dice', f"epoch_{epoch}_step_{step}_no_{i}.png"))

            plt.close()


def load_best_model(model,likelihood,args,device):
    list_dir = os.listdir(os.path.join(args.saved_name,args.saved_path, 'saved_model'))
    saved_models = [i for i in list_dir if i.startswith('best_dice_model_epoch')]
    saved_likelihood = [i for i in list_dir if i.startswith('best_dice_likelihood_epoch')]

    if len(saved_models) != 1 or len(saved_likelihood) != 1:
        raise ValueError("More than one best model saved.")


    model.load_state_dict(torch.load(os.path.join(args.saved_name,args.saved_path,'saved_model', saved_models[0]),map_location=torch.device(device)))
    likelihood.load_state_dict(torch.load(os.path.join(args.saved_name,args.saved_path,'saved_model', saved_likelihood[0]),map_location=torch.device(device)))

    return model,likelihood

def load_model(model,likelihood,args,device,epoch):
    list_dir = os.listdir(os.path.join(args.saved_name,args.saved_path, 'saved_model'))
    saved_models = [i for i in list_dir if i.startswith('model_epoch')]
    saved_likelihood = [i for i in list_dir if i.startswith('likelihood_epoch')]

    if len(saved_models) != 1 or len(saved_likelihood) != 1:
        raise ValueError("More than one best model saved.")

    epoch1 = int(re.findall(r'\d+', saved_models[0])[0])
    if epoch1!= epoch:
        raise ValueError("The epoch of the saved model is not the same as the input epoch.")


    model.load_state_dict(torch.load(os.path.join(args.saved_name,args.saved_path,'saved_model', saved_models[0]),map_location=torch.device(device)))
    likelihood.load_state_dict(torch.load(os.path.join(args.saved_name,args.saved_path,'saved_model', saved_likelihood[0]),map_location=torch.device(device)))

    return model,likelihood

def load_last_model(model,likelihood,args,device,folder):
    list_dir = os.listdir(os.path.join(args.saved_name,args.saved_path, 'saved_model'))

    saved_models = [i for i in list_dir if i.startswith('model_epoch')]
    saved_likelihood = [i for i in list_dir if i.startswith('likelihood_epoch')]
    
    epoch = int(re.findall(r'\d+', saved_models[0])[0])
    

    model.load_state_dict(torch.load(os.path.join(args.saved_name,args.saved_path,'saved_model', saved_models[0]),map_location=torch.device(device)))
    likelihood.load_state_dict(torch.load(os.path.join(args.saved_name,args.saved_path,'saved_model', saved_likelihood[0]),map_location=torch.device(device)))

    return model,likelihood,saved_models[0],saved_likelihood[0],epoch

def load_best_model_and_epoch(model,likelihood,args,device,folder):
    try:
        list_dir = os.listdir(os.path.join(args.saved_name,args.saved_path, 'saved_model'))
    except:
        args.saved_name = folder
        args.saved_path = 'saved_path'
        list_dir = os.listdir(os.path.join(folder,'saved_path', 'saved_model'))

    saved_models = [i for i in list_dir if i.startswith('best_dice_model_epoch')]
    saved_likelihood = [i for i in list_dir if i.startswith('best_dice_likelihood_epoch')]

    if len(saved_models) != 1 or len(saved_likelihood) != 1:
        raise ValueError("More than one best model saved.")
    
    epoch = int(re.findall(r'\d+', saved_models[0])[0])
    

    model.load_state_dict(torch.load(os.path.join(args.saved_name,args.saved_path,'saved_model', saved_models[0]),map_location=torch.device(device)))
    likelihood.load_state_dict(torch.load(os.path.join(args.saved_name,args.saved_path,'saved_model', saved_likelihood[0]),map_location=torch.device(device)))

    return model,likelihood,saved_models[0],saved_likelihood[0],epoch

def load_best_model_and_epoch_simulated(model,likelihood,device,folder):
    list_dir = os.listdir(os.path.join(folder,'saved_path', 'saved_model'))
    
    saved_models = [i for i in list_dir if i.startswith('best_dice_model_epoch')]
    saved_likelihood = [i for i in list_dir if i.startswith('best_dice_likelihood_epoch')]

    if len(saved_models) != 1 or len(saved_likelihood) != 1:
        raise ValueError("More than one best model saved.")
    
    model.load_state_dict(torch.load(os.path.join(folder,'saved_path', 'saved_model', saved_models[0]),map_location=torch.device(device)))
    # likelihood.load_state_dict(torch.load(os.path.join(folder,'saved_path', 'saved_model', saved_likelihood[0]),map_location=torch.device(device)))

    return model#,likelihood


def load_best_model_and_epoch_unet(model,args,device,folder):
    try:
        list_dir = os.listdir(os.path.join(args.saved_name,args.saved_path, 'saved_model'))
    except:
        args.saved_name = folder
        args.saved_path = 'saved_path'
        list_dir = os.listdir(os.path.join(folder,'saved_path', 'saved_model'))


    saved_models = [i for i in list_dir if i.startswith('best_dice_model_epoch') or i.startswith('best_model_epoch')]

    if len(saved_models) != 1:
        raise ValueError("More than one best model saved.")
    
    epoch = int(re.findall(r'\d+', saved_models[0])[0])
    

    model.load_state_dict(torch.load(os.path.join(args.saved_name,args.saved_path,'saved_model', saved_models[0]),map_location=torch.device(device)))

    return model,saved_models[0],epoch

    
def get_saved_folder_name(args):

    
    

    

    if args.addnoise == "add":
        A = "AddNoise__"
    elif args.addnoise == "noadd":
        A = "NoAddNoise__"
    elif args.addnoise == "add3sigmas":
        A = "Add3Sigmas__"
    elif args.addnoise == "add3sigmas3mus":
        A = "Add3Sigmas3Mus__"
    
    if args.addnoise_pred == 'add':
        A = A + "AddNoisePred__"
    elif args.addnoise_pred == 'noadd':
        A = A + "NoAddNoisePred__"

    
    if args.loss_add_mu_reg == "add":
        A = A + "AddMuReg__"
    elif args.loss_add_mu_reg == "noadd":
        A = A + "NoAddMuReg__"
    elif args.loss_add_mu_reg == "sum_2":
        A = A + "Sum2MuReg__"
    

    if args.retrain == "load":
        args.saved_name = A + args.labels + '__' + args.labels_infer + '__' + args.loss_type    +'__RetrainEpoch_'+str(args.retrain_epoch) 
    else:
        args.saved_name = A + args.labels  + '__' + args.labels_infer + '__' + args.loss_type 
    
    if args.data_aug == 'add':
        args.saved_name = args.saved_name + '__DA'
    
    return args


def get_saved_folder_name_simulated(args):

    
    if args.addnoise == "add":
        A = "AddNoise_"
    elif args.addnoise == "noadd":
        A = "NoAddNoise_"
    elif args.addnoise == "add3sigmas":
        A = "Add3Sigmas_"
    elif args.addnoise == "add3sigmas3mus":
        A = "Add3Sigmas3Mus_"
    elif args.addnoise == "add_1_bias_variance":
        A = 'Add_1_bias_variance__'

    
    if args.addnoise_pred == 'add':
        A = A + "AddNoisePred_"
    elif args.addnoise_pred == 'noadd':
        A = A + "NoAddNoisePred__"

    if args.FeatBatchNorm == "add":
        A = A + "FeatBN__"
    elif args.FeatBatchNorm == "noadd":
        A = A + "NoFeatBN__"

    A = A + "Bias"+str(args.mu_simulated_label_ind) + "__Var"+str(args.sigma_simulated_label_ind)+"__"
        
    
    if args.retrain == "load":
        args.saved_name = A + args.labels + '__' + args.labels_infer + '__' + args.loss_type  + '__' + args.nll_type + '__' + args.kernel_type + '__' +args.VariationalStrategy+'_'+str(args.InducingPtsType)+str(args.num_indu) + '__FeatDim'+str(args.feat_dim) + '__bst_'+str(args.batch_size_train)+'__bsv_'+str(args.batch_size_val)+'__fr_'+str(args.FREQ_SAVE)+'__RetrainEpoch_'+str(args.retrain_epoch) 
    else:
        args.saved_name = A + args.labels  + '__' + args.labels_infer + '__' + args.loss_type + '__' + args.nll_type + '__' + args.kernel_type + '__' +args.VariationalStrategy+'_'+str(args.InducingPtsType)+str(args.num_indu)+ '__FeatDim'+str(args.feat_dim) + '__bst_'+str(args.batch_size_train)+'__bsv_'+str(args.batch_size_val)+'__fr_'+str(args.FREQ_SAVE)
    
    if args.data_aug == 'add':
        args.saved_name = args.saved_name + '__DA'
    


    return args

def get_saved_folder_name_unet_model(args):

    if args.FeatBatchNorm == "add":
        A = "__FeatBN"
    elif args.FeatBatchNorm == "noadd":
        A = "__NoFeatBN"

    if args.labels == 'seg_simulated' or args.labels == 'simulated_ind':
        args.saved_name =  'Unet_bst_'+str(args.batch_size_train) + "__labels_"+args.labels + "_sigma_"+'%.04f'%args.sigma_simulated_label+"_mu_"+'%.04f'%args.mu_simulated_label  + '__' + args.labels_infer + "__" + args.loss_type+ "__FeatDim" + str(args.feat_dim) + A +'__bsv_'+str(args.batch_size_val)+'_freq_'+str(args.FREQ_SAVE)
    else:
        args.saved_name =  'Unet_bst_'+str(args.batch_size_train) + "__labels_"+args.labels  + '__' + args.labels_infer + "__" + args.loss_type+ "__FeatDim" + str(args.feat_dim) + A +'__bsv_'+str(args.batch_size_val)+'_freq_'+str(args.FREQ_SAVE)
    
    if args.data_aug == 'add':
        args.saved_name = args.saved_name + '__DA'

    
    return args

def get_saved_folder_name_unet_linear_model(args):

    if args.FeatBatchNorm == "add":
        A = "__FeatBN"
    elif args.FeatBatchNorm == "noadd":
        A = "__NoFeatBN"

    if args.loss_add_mu_reg == "add":
        A = A + "__AddMuReg"
    elif args.loss_add_mu_reg == "noadd":
        A = A + "__NoAddMuReg"
    elif args.loss_add_mu_reg == "sum_2":
        A = A + "__Sum2MuReg"

    args.saved_name =  'Unetlinear_bst_'+str(args.batch_size_train) + "__labels_"+args.labels  + '__' + args.labels_infer + "__" + args.loss_type+ "__FeatDim" + str(args.feat_dim) + A +'__bsv_'+str(args.batch_size_val)+'_freq_'+str(args.FREQ_SAVE)
    
    if args.data_aug == 'add':
        args.saved_name = args.saved_name + '__DA'
    
    return args


def generate_inducing_points(feature_extractor,num_features,train_set,args,device):
    # generate inducing points from the feature space

    if args.InducingPtsType == 'fixed':
        # select one image containing prostate gland, and the extract features from the image 
        with torch.no_grad():
            inducing_points = torch.permute(feature_extractor(torch.from_numpy(train_set[1][0]).float().to(device)[None,None]).squeeze(),(1,2,0)).reshape(-1,num_features).to(device)[:args.num_indu,...]
        
        inducing_points = inducing_points.requires_grad_(True)
    elif args.InducingPtsType == 'uniform':
        # uniformly sample inducing points from the feature space
        inducing_points = []
        pts_per_dim = int(args.num_indu ** (1/3))+1
        
        selected_images = np.linspace(0,len(train_set)-1,pts_per_dim).astype(int)
        selected_w = np.linspace(0,train_set[0][0].shape[1]-1,pts_per_dim).astype(int)
        selected_h = np.linspace(0,train_set[0][0].shape[0]-1,pts_per_dim).astype(int)
        xx, yy = np.meshgrid(selected_h, selected_w)

        for i in range(len(selected_images)):
            with torch.no_grad():
                features = torch.permute(feature_extractor((train_set[selected_images[i]][0]).float().to(device)[None,None]).squeeze(),(1,2,0)).detach().cpu().numpy()
            features = features[xx,yy,:].reshape(-1,num_features)
            if i == 0:
                inducing_points = features
            else:
                inducing_points = np.concatenate((inducing_points,features),axis=0)
        
        inducing_points = torch.from_numpy(inducing_points).to(device).requires_grad_(True)

    return inducing_points



def BinarySegmentationNLLMetric(outputs, targets):
   
    """
    Args:
        outputs: Probabilities from the model, shape [batch_size, 1, height, width].
        targets: Ground truth binary labels, shape [batch_size, height, width].
    """
    # Flatten outputs and targets for easier computation
    outputs = outputs.view(outputs.size(0), -1)  # Shape: [batch_size, height * width]
    targets = targets.view(targets.size(0), -1)  # Shape: [batch_size, height * width]

    # Clamp outputs to prevent log(0)
    outputs = torch.clamp(outputs, min=1e-7, max=1 - 1e-7)

    # Compute log probabilities for positive and negative classes
    log_probs = torch.log(outputs)         # Log p for positive class
    log_probs_neg = torch.log(1 - outputs) # Log p for negative class

    # Compute NLL for binary targets
    nll = -torch.where(targets == 1, log_probs, log_probs_neg)

    return nll.mean(dim = 1)






class BinaryCalibrationError():
    # adapted from torchmetrics.classification.BinaryCalibrationError

    def __init__(self,n_bins):
        self.n_bins = n_bins
        self.confidences = []
        self.accuracies = []

    def update(self, preds, target):
        """Update metric states with predictions and targets."""

        preds = preds.flatten()
        target = target.flatten()

        
        confidences, accuracies = preds, target
        self.confidences = confidences
        self.accuracies = accuracies

    def compute(self):
        """Compute metric."""
        self.confidences = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in self.confidences]
        confidences = torch.cat(self.confidences, dim=0)

        self.accuracies = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in self.accuracies]
        accuracies = torch.cat(self.accuracies, dim=0)

        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, dtype=confidences.dtype, device=confidences.device)

        accuracies = accuracies.to(dtype=confidences.dtype)
        acc_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
        conf_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)
        count_bin = torch.zeros(len(bin_boundaries), device=confidences.device, dtype=confidences.dtype)

        indices = torch.bucketize(confidences, bin_boundaries, right=True) - 1

        count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))

        conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
        conf_bin = torch.nan_to_num(conf_bin / count_bin)

        acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
        acc_bin = torch.nan_to_num(acc_bin / count_bin)

        prop_bin = count_bin / count_bin.sum()

        return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)

    def reset(self):
        """
        Reset the metric for a new evaluation.
        """
        self.confidences = 0.0
        self.accuracies = 0



def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    # adapted from https://github.com/hollance/reliability-diagrams
    """Collects predictions into bins used to draw a reliability diagram.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins

    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.

    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.

    Returns 
        expected_calibration_error: a weighted average of all calibration gaps
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    # avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    # avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    # mce = np.max(gaps)

    return ece

    # return { "accuracies": bin_accuracies, 
    #          "confidences": bin_confidences, 
    #          "counts": bin_counts, 
    #          "bins": bins,
    #          "avg_accuracy": avg_acc,
    #          "avg_confidence": avg_conf,
    #          "expected_calibration_error": ece,
    #          "max_calibration_error": mce }


def save_arguments(args):
    with open(os.path.join(os.getcwd(),args.saved_name,'args.txt'), 'w') as f:
        for key in vars(args):
            f.write(f"{key}: {getattr(args, key)}\n")


def get_args(path,folder):
    args = {}
    with open(path, 'r') as f:
        for line in f:
            key, value = line.split(": ")
            args[key] = value[:-1] 


    args['batch_size_train'] = int(args['batch_size_train'])
    args['batch_size_val'] = int(args['batch_size_val'])
    args['FREQ_SAVE'] = 5
    args['FREQ_VISULISE'] = 5
    args['num_indu'] = int(args['num_indu'])
    args['feat_dim'] = int(args['feat_dim'])
    args['seed'] = int(args['seed'])
    args['num_class'] = int(args['num_class'])
    args['num_annotators'] = int(args['num_annotators'])
    
    args_convert = argparse.Namespace(**args)
 
    return args_convert

def get_args_unet(path,folder):
    args = {}
    with open(path, 'r') as f:
        for line in f:
            key, value = line.split(": ")
            args[key] = value[:-1]

    split_folder = folder.split('__')

    if 'data_aug' not in args.keys():
        if 'DA' in folder:
            args['data_aug'] = 'add'
        else:
            args['data_aug'] = 'noadd'

    
    if 'labels' not in args.keys():
        if 'random3' in folder:
            args['labels'] = 'random3'
        elif 'co' in folder and 'random3' not in folder:
            args['labels'] = 'co'
        else:
            args['labels'] = 'co'

    if 'labels_infer' not in args.keys():
        if 'high_quality' in folder:
            args['labels_infer'] = 'high_quality'
        elif 'voted_only' in folder:
            args['labels_infer'] = 'voted_only'
        else:
            args['labels_infer'] = 'high_quality'

    # if args['labels'] == 'co' and args['labels_infer'] == 'voted_only':
    #     raise ValueError('Cannot use co labels for training and voted_only for inference.')


    if 'num_class' not in args.keys():
        args['num_class'] = 1
    
    if 'loss_type' not in args.keys():
        if 'dice' in folder and 'BCE' not in folder:
            args['loss_type'] = 'dice'
        elif 'BCE' in folder and 'dice' not in folder:
            args['loss_type'] = 'BCE'
        elif 'BCE' in folder and 'dice' in folder and 'dice_BCE' in folder:
            args['loss_type'] = 'dice_BCE'
        else:
            args['loss_type'] = 'BCE'
    
    if 'feat_dim' not in args.keys():
        try:
            feat_dim = [s for s in split_folder if 'FeatDim' in s]
            num_feat_dim = re.findall(r'\d+', feat_dim[0])
            if int(num_feat_dim[0]) not in [8,16,32,64]:
                raise ValueError('Feature dimension is not in [8,16,32,64].')
            args['feat_dim'] = num_feat_dim[0]
        except:
            args['feat_dim'] = 64

    if 'FeatBatchNorm' not in args.keys():
        if 'FeatBN' in folder and 'NoFeatBN' not in folder:
            args['FeatBatchNorm'] = 'add'
        elif 'NoFeatBN' in folder:
            args['FeatBatchNorm'] = 'noadd'
        else:
            args['FeatBatchNorm'] = 'noadd'

    args['batch_size_val'] = int(args['batch_size_val'])
    args['FREQ_SAVE'] = 5
    args['FREQ_VISULISE'] = 5
    args['feat_dim'] = int(args['feat_dim'])
    args['seed'] = int(args['seed'])
    args['num_class'] = int(args['num_class'])
    
    args_convert = argparse.Namespace(**args)
 
    return args_convert

def Hyper_parameters_written(f,hyperpara,args,saved_model_name,saved_likelihood_name):
    f.write('Hyper-parameters &  ')
    f.write('$%.3f\pm%.3f$ &  ' % (hyperpara['outputscale'].mean(), np.std(hyperpara['outputscale'])))
    f.write('$%.3f\pm%.3f$ &  ' % (hyperpara['lengthscale'].mean(), np.std(hyperpara['lengthscale'])))
    f.write(' N/A &  ')
    
    f.write(str(saved_model_name))
    f.write(' &  ')
    f.write(str(saved_likelihood_name))
    f.write('\n')

def noiseless_metrics_written(f,metrics):
    f.write('Nonoise &  ')
    f.write('\n')
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['dice'].mean(), np.std(metrics['dice'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['dice_non_empty'].mean(), np.std(metrics['dice_non_empty'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['iou'].mean(), np.std(metrics['iou'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['iou_non_empty'].mean(), np.std(metrics['iou_non_empty'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['hd'].mean(), np.std(metrics['hd'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['hd95'].mean(), np.std(metrics['hd95'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['ece'].mean(), np.std(metrics['ece'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['nll'].mean(), np.std(metrics['nll'])))
    f.write('\n')

def noise_metrics_written(f,hyperpara_addnoise,metrics_addnoise):
    
    f.write('Addnoise  Sigma &  ')
    f.write('$%.3f\pm%.3f$ &  ' % (hyperpara_addnoise['noise'].mean(), np.std(hyperpara_addnoise['noise'])))
    # f.write('\n')
    if len(hyperpara_addnoise['mu'])>0:
        f.write('Mu &  ')
        f.write('$%.3f\pm%.3f$ &  ' % (hyperpara_addnoise['mu'].mean(), np.std(hyperpara_addnoise['mu'])))
    f.write('\n')

    f.write('$%.3f\pm%.3f$ &  ' % (metrics_addnoise['dice'].mean(), np.std(metrics_addnoise['dice'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics_addnoise['dice_non_empty'].mean(), np.std(metrics_addnoise['dice_non_empty'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics_addnoise['iou'].mean(), np.std(metrics_addnoise['iou'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics_addnoise['iou_non_empty'].mean(), np.std(metrics_addnoise['iou_non_empty'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics_addnoise['hd'].mean(), np.std(metrics_addnoise['hd'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics_addnoise['hd95'].mean(), np.std(metrics_addnoise['hd95'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics_addnoise['ece_addnoise'].mean(), np.std(metrics_addnoise['ece_addnoise'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics_addnoise['nll'].mean(), np.std(metrics_addnoise['nll'])))
    f.write('\n')

    
        

    
def save_metrics(csv_name,folder,metrics_all,hyperpara_all,metrics_addnoise_all,saved_model_name,saved_likelihood_name,args):

    with open(csv_name, 'a', encoding='UTF8') as f:
        f.write(folder)
        f.write('\n')

        if args.pre_labels == 'co':
            if args.labels_infer == 'voted_only':
                raise ValueError('Cannot use voted only labels for inference when training with high quality labels.')
            elif args.labels_infer == 'high_quality':
                f.write('Training only using high quality label, and during inference compare with these high quality labels \n')
            Hyper_parameters_written(f,hyperpara_all,args,saved_model_name,saved_likelihood_name)
            noiseless_metrics_written(f,metrics_all)
            if args.addnoise == 'add':
                noise_metrics_written(f,hyperpara_all,metrics_addnoise_all)
        elif args.pre_labels == 'random4':
            
            f.write('Training: Using three annotations and corrected label \n')
            
            if args.labels_infer == 'voted_only':
                f.write('Val: Comparing with voted labels \n')
            elif args.labels_infer == 'high_quality':
                f.write('Val: Comparing with high quality label \n')
            elif args.labels_infer == 'seg_1_2_3_HQ':
                f.write('Val: Comparing with seg 1, seg 2, seg 3 and high quality label \n')

            Hyper_parameters_written(f,hyperpara_all['vote'],args,saved_model_name,saved_likelihood_name)
            f.write('No noise, compare with high quality label \n')
            noiseless_metrics_written(f,metrics_all['vote'])
            if len(metrics_addnoise_all['vote']['dice'])>0:
                noise_metrics_written(f,hyperpara_all['vote'],metrics_addnoise_all['vote'])

            if args.addnoise == 'add3sigmas' or args.addnoise == 'add3sigmas3mus':
                f.write('Adding sigma0 and mu0, and compare with high quality label \n')
                noise_metrics_written(f,hyperpara_all['0_voted'],metrics_addnoise_all['0_voted'])
                f.write('Adding sigma1 and mu1, and compare with high quality label \n')
                noise_metrics_written(f,hyperpara_all['1_voted'],metrics_addnoise_all['1_voted'])
                f.write('Adding sigma2 and mu2, and compare with high quality label \n')
                noise_metrics_written(f,hyperpara_all['2_voted'],metrics_addnoise_all['2_voted'])


           

            f.write('Comparing with seg 0 \n')
            Hyper_parameters_written(f,hyperpara_all['0'],args,saved_model_name,saved_likelihood_name)
            noiseless_metrics_written(f,metrics_all['0'])
            if len(metrics_addnoise_all['0']['dice'])>0:
                noise_metrics_written(f,hyperpara_all['0'],metrics_addnoise_all['0'])

            f.write('Comparing with seg 1 \n')
            Hyper_parameters_written(f,hyperpara_all['1'],args,saved_model_name,saved_likelihood_name)
            noiseless_metrics_written(f,metrics_all['1'])
            if len(metrics_addnoise_all['1']['dice'])>0:
                noise_metrics_written(f,hyperpara_all['1'],metrics_addnoise_all['1'])

            f.write('Comparing with seg 2 \n')
            Hyper_parameters_written(f,hyperpara_all['2'],args,saved_model_name,saved_likelihood_name)
            noiseless_metrics_written(f,metrics_all['2'])
            if len(metrics_addnoise_all['2']['dice'])>0:
                noise_metrics_written(f,hyperpara_all['2'],metrics_addnoise_all['2'])

            

            

        f.write('\n') 
            


def save_metrics_grid_sample(csv_name,folder,metrics_all,hyperpara_all,metrics_addnoise_all,saved_model_name,saved_likelihood_name,args,sigma,mu):

    with open(csv_name, 'a', encoding='UTF8') as f:
        f.write(folder)
        f.write('\n')
        f.write('Sigma &  ')
        f.write('$%.3f$ &  ' % sigma)
        f.write('Mu &  ')
        f.write('$%.3f$ &  ' % mu)
        f.write('\n')

        if args.pre_labels == 'co':
            raise ValueError('Cannot use grid sample for co labels.')
        elif args.pre_labels == 'random4':
            Hyper_parameters_written(f,hyperpara_all,args,saved_model_name,saved_likelihood_name)
            # noiseless_metrics_written(f,metrics_all)
            noise_metrics_written(f,hyperpara_all,metrics_addnoise_all)


        
def noiseless_metrics_written_unet(f,metrics):

    f.write('$%.3f\pm%.3f$ &  ' % (metrics['dice'].mean(), np.std(metrics['dice'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['dice_non_empty'].mean(), np.std(metrics['dice_non_empty'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['iou'].mean(), np.std(metrics['iou'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['iou_non_empty'].mean(), np.std(metrics['iou_non_empty'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['hd'].mean(), np.std(metrics['hd'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['hd95'].mean(), np.std(metrics['hd95'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['ece'].mean(), np.std(metrics['ece'])))
    f.write('$%.3f\pm%.3f$ &  ' % (metrics['nll'].mean(), np.std(metrics['nll'])))
    f.write('\n')
    



def save_metrics_unet(csv_name,folder,metrics,saved_model_name,args):

    with open(csv_name, 'a', encoding='UTF8') as f:
        f.write(folder)
        f.write('\n')
        f.write('Noiseless &  ')
        f.write(str(saved_model_name))
        f.write('\n')
        if args.pre_labels == 'co':
            if args.labels_infer == 'voted_only':
                f.write('Training only using voted label, and during inference compare with these voted labels \n')
            elif args.labels_infer == 'high_quality':
                f.write('Training only using high quality label, and during inference compare with these high quality labels \n')
            
            noiseless_metrics_written_unet(f,metrics)

        elif args.pre_labels == 'random3':
            f.write('Training: Using three labels from three observers \n')
            if args.labels_infer == 'voted_only':
                f.write('Comparing with voted labels \n')
            elif args.labels_infer == 'high_quality':
                f.write('Comparing with high quality label \n')
            noiseless_metrics_written_unet(f,metrics['vote'])

            f.write('Comparing with seg 0 \n')
            noiseless_metrics_written_unet(f,metrics['0'])

            f.write('Comparing with seg 1 \n')
            noiseless_metrics_written_unet(f,metrics['1'])

            f.write('Comparing with seg 2 \n')
            noiseless_metrics_written_unet(f,metrics['2'])

            f.write('Average results from three comparisions \n')

            metrics_all_three = {key:np.concatenate((metrics['0'][key],metrics['1'][key],metrics['2'][key]),axis=0) for key in metrics['0'].keys()}
            noiseless_metrics_written_unet(f,metrics_all_three)

        elif args.pre_labels == 'seg1':
            f.write('Training: Using seg0 labels \n')
            # if args.labels_infer == 'voted_only':
            #     f.write('Comparing with voted labels \n')
            # elif args.labels_infer == 'high_quality':
            f.write('Comparing with high quality label \n')
            noiseless_metrics_written_unet(f,metrics['vote'])
            f.write('Comparing with seg 0 \n')
            noiseless_metrics_written_unet(f,metrics['0'])

        elif args.pre_labels == 'seg2':
            f.write('Training: Using seg1 labels \n')
            # if args.labels_infer == 'voted_only':
            #     f.write('Comparing with voted labels \n')
            # elif args.labels_infer == 'high_quality':
            f.write('Comparing with high quality label \n')
            noiseless_metrics_written_unet(f,metrics['vote'])

            f.write('Comparing with seg 1 \n')
            noiseless_metrics_written_unet(f,metrics['1'])
            # f.write('Comparing with seg 0 \n')
            # noiseless_metrics_written_unet(f,metrics['0'])

        elif args.pre_labels == 'seg3':
            f.write('Training: Using seg2 labels \n')
            # if args.labels_infer == 'voted_only':
            #     f.write('Comparing with voted labels \n')
            # elif args.labels_infer == 'high_quality':
            f.write('Comparing with high quality label \n')
            noiseless_metrics_written_unet(f,metrics['vote'])

            f.write('Comparing with seg 2 \n')
            noiseless_metrics_written_unet(f,metrics['2'])

        

        elif args.pre_labels == 'high_quality':
            f.write('Training: Using %s labels \n' % args.pre_labels)
            f.write('Comparing with high quality label \n')
            noiseless_metrics_written_unet(f,metrics['vote'])






        f.write('\n')
        

def save_metrics_linear_unet(csv_name,folder,metrics_all,hyperpara_all,saved_model_name,args):

    with open(csv_name, 'a', encoding='UTF8') as f:
        f.write(folder)
        f.write('\n')

        if args.pre_labels == 'co':
            raise ValueError('Not supported.')
            
        elif args.pre_labels == 'random3':
            f.write('Training: Using three labels from three observers \n')
            if args.labels_infer == 'voted_only':
                f.write('Comparing with voted labels \n')
            elif args.labels_infer == 'high_quality':
                f.write('Comparing with high quality label \n')
            
            f.write(str(saved_model_name))
            f.write('\n')
            
            
            noiseless_metrics_written(f,metrics_all['vote'])

            f.write('Adding sigma0 and mu0 \n')
            f.write('Addnoise  Sigma &  ')
            f.write('$%.3f$ &  ' % (hyperpara_all['noise0']))
            f.write('Mu &  ')
            f.write('$%.3f$ &  ' % (hyperpara_all['mu0']))
            f.write('\n')
            noiseless_metrics_written(f,metrics_all['0'])

            f.write('Adding sigma1 and mu1 \n')
            f.write('Addnoise  Sigma &  ')
            f.write('$%.3f$ &  ' % (hyperpara_all['noise1']))
            f.write('Mu &  ')
            f.write('$%.3f$ &  ' % (hyperpara_all['mu1']))
            f.write('\n')

            noiseless_metrics_written(f,metrics_all['1'])

            f.write('Adding sigma2 and mu2 \n')
            f.write('Addnoise  Sigma &  ')
            f.write('$%.3f$ &  ' % (hyperpara_all['noise2']))
            f.write('Mu &  ')
            f.write('$%.3f$ &  ' % (hyperpara_all['mu2']))
            f.write('\n')
            noiseless_metrics_written(f,metrics_all['2'])

            f.write('\n')

        

def get_args_from_folder(folder):

    args = {}
    args['num_class'] = 1
    args['VariationalStrategy'] = 'InducingPts'
    args['seed'] = 0
    args['dataset'] = './data'

    if 'high_quality' in folder:
        args['labels_infer'] = 'high_quality'
    elif 'voted_only' in folder:
        args['labels_infer'] = 'voted_only'
    else:
        args['labels_infer'] = 'high_quality'


    
    if 'labels' not in args.keys():
        if 'random3' in folder:
            args['labels'] = 'random3'
        elif 'co' in folder and 'random3' not in folder:
            args['labels'] = 'co'
        else:
            args['labels'] = 'co' 

    if args['labels'] == 'co' and args['labels_infer'] == 'voted_only':
        raise ValueError('Cannot use co labels for training and voted_only for inference.')

    if 'fixed' in folder:
        args['InducingPtsType'] = 'fixed'
    elif 'uniform' in folder:
        args['InducingPtsType'] = 'uniform'
    else:
        args['InducingPtsType'] = 'fixed'
    
    if 'FixedHyper' in folder:
        args['hyperparameter'] = 'fixed'
    elif 'OptimizeHyper' in folder:
        args['hyperparameter'] = 'optimize'
    else:
        args['hyperparameter'] = 'optimize'


    if 'AddNoise' in folder and 'NoAddNoise' not in folder:
        args['addnoise'] = 'add'
    elif 'NoAddNoise' in folder:
        args['addnoise'] = 'noadd'
    elif 'Add3Sigmas' in folder and 'Add3Sigmas3Mus' not in folder:
        args['addnoise'] = 'add3sigmas'
    elif 'Add3Sigmas3Mus' in folder:
        args['addnoise'] = 'add3sigmas3mus'
    else:
        args['addnoise'] = 'noadd'
    
    if args['addnoise'] == 'add':
        if 'AddNoisePred' in folder and 'NoAddNoisePred' not in folder:
            args['addnoise_pred'] = 'add'
        elif 'NoAddNoisePred' in folder:
            args['addnoise_pred'] = 'noadd'
        else:
            args['addnoise_pred'] = 'add'


    if 'PredictiveLogLikelihood' in folder:
        args['nll_type'] = 'PredictiveLogLikelihood'
    elif 'VariationalELBO' in folder:
        args['nll_type'] = 'VariationalELBO'
    else:
        args['nll_type'] = 'VariationalELBO'

    if 'FeatBN' in folder and 'NoFeatBN' not in folder:
        args['FeatBatchNorm'] = 'add'
    elif 'NoFeatBN' in folder:
        args['FeatBatchNorm'] = 'noadd'
    else:
        args['FeatBatchNorm'] = 'noadd'
    
    if 'dice' in folder and 'BCE' not in folder and 'likelihood' not in folder:
        args['loss_type'] = 'dice'
    elif 'BCE' in folder and 'dice' not in folder and 'likelihood' not in folder:
        args['loss_type'] = 'BCE'
    elif 'BCE' in folder and 'dice' in folder and 'dice_BCE' in folder and 'likelihood' not in folder:
        args['loss_type'] = 'dice_BCE'
    elif 'dice_BCE_likelihood' in folder:
        args['loss_type'] = 'dice_BCE_likelihood'
    elif 'likelihood' in folder and 'dice_BCE_likelihood' not in folder:
        args['loss_type'] = 'likelihood'
    else:
        args['loss_type'] = 'likelihood'

    if 'RBF' in folder and 'Linear' not in folder:
        args['kernel_type'] = 'RBF'
    elif 'Linear' in folder and 'RBF' in folder:
        args['kernel_type'] = 'RBF_Linear'
    elif 'RBF' not in folder and 'Linear' not in folder:
        args['kernel_type'] = 'RBF'

    split_folder = folder.split('_')
    split_folder1 = folder.split('__')
    # find split_folder which include 'InducingPts', and make the code in one line
    try:
        inducing_pts = [s for s in split_folder1 if 'InducingPts' in s or 'uniform' in s or 'fixed' in s]
        num_inducing_pts = re.findall(r'\d+', inducing_pts[0])
        if int(num_inducing_pts[0]) not in [500,1000,1500,2000]:
            raise ValueError('Number of inducing points is not in [500,1000,2000].')

        args['num_indu'] = num_inducing_pts[0]
    except:
        args['num_indu'] = 500
    
    try:
        feat_dim = [s for s in split_folder if 'FeatDim' in s]
        num_feat_dim = re.findall(r'\d+', feat_dim[0])
        if int(num_feat_dim[0]) not in [8,16,32,64]:
            raise ValueError('Feature dimension is not in [8,16,32,64].')
        args['feat_dim'] = num_feat_dim[0]
    except:
        args['feat_dim'] = 64
    
    try:
        batch_size = [s for s in split_folder1 if 'bst' in s]
        num_batch_size = re.findall(r'\d+', batch_size[0])
        if int(num_batch_size[0]) not in [2,4,8,16,32,64]:
            raise ValueError('Batch size is not in [2,4,8,16,32,64].')
        args['batch_size_train'] = num_batch_size[0]
        args['batch_size_val'] = num_batch_size[0]
    except:
        args['batch_size_train'] = 4
        args['batch_size_val'] = 4


    args['batch_size_train'] = int(args['batch_size_train'])
    args['batch_size_val'] = int(args['batch_size_val'])
    args['FREQ_SAVE'] = 5
    args['FREQ_VISULISE'] = 5
    args['num_indu'] = int(args['num_indu'])
    args['feat_dim'] = int(args['feat_dim'])
    args['seed'] = int(args['seed'])
    args['num_class'] = int(args['num_class'])
    
    args_convert = argparse.Namespace(**args)

    return args_convert


def get_args_from_folder_unet(folder):
    split_folder = folder.split('__')
    args = {}
    args['num_class'] = 1
    args['dataset'] = './data'

    if 'high_quality' in folder:
        args['labels_infer'] = 'high_quality'
    elif 'voted_only' in folder:
        args['labels_infer'] = 'voted_only'
    else:
        args['labels_infer'] = 'high_quality'

    if 'random3' in folder:
        args['labels'] = 'random3'
    elif 'co' in folder and 'random3' not in folder:
        args['labels'] = 'co'
    else:
        args['labels'] = 'co'


    if args['labels'] == 'co' and args['labels_infer'] == 'voted_only':
        raise ValueError('Cannot use co labels for training and voted_only for inference.')

    if 'dice' in folder and 'BCE' not in folder:
        args['loss_type'] = 'dice'
    elif 'BCE' in folder and 'dice' not in folder:
        args['loss_type'] = 'BCE'
    elif 'BCE' in folder and 'dice' in folder and 'dice_BCE' in folder:
        args['loss_type'] = 'dice_BCE'
    else:
        args['loss_type'] = 'BCE'

    try:
        feat_dim = [s for s in split_folder if 'FeatDim' in s]
        num_feat_dim = re.findall(r'\d+', feat_dim[0])
        if int(num_feat_dim[0]) not in [8,16,32,64]:
            raise ValueError('Feature dimension is not in [8,16,32,64].')
        args['feat_dim'] = num_feat_dim[0]
    except:
        args['feat_dim'] = 64


    if 'FeatBN' in folder and 'NoFeatBN' not in folder:
        args['FeatBatchNorm'] = 'add'
    elif 'NoFeatBN' in folder:
        args['FeatBatchNorm'] = 'noadd'
    else:
        args['FeatBatchNorm'] = 'noadd'

    args['FREQ_SAVE'] = 5
    args['FREQ_VISULISE'] = 5
    args['feat_dim'] = int(args['feat_dim'])
    args['seed'] = 0
    args['num_class'] = int(args['num_class'])
    
    
    args_convert = argparse.Namespace(**args)
 
    return args_convert


def label_check(labels,mask_all_val,mask_wo_noise):
    if labels == 'co':
        if not torch.all(mask_all_val == mask_wo_noise):
            raise ValueError('Labels are not consistent.')




def get_metrics(output,masks_val,args,preds_val,criterion,mll,ece):
# get metrics for each iteration

    binary_output = output.mean.ge(0.5).float() # Transform these probabilities to be 0/1 labels
            
    cls_probs = output.mean
    cls_probs_reshape = cls_probs.reshape(masks_val.shape[0],masks_val.shape[1],masks_val.shape[2])

    if args.loss_type == 'likelihood':
        loss =  -mll(preds_val, masks_val.view(-1))
    elif args.loss_type == 'dice' or args.loss_type == 'dice_voted':
        soft_dice = compute_dice(cls_probs_reshape, masks_val.long())
        loss = (1-soft_dice.mean())
    elif args.loss_type == 'BCE':
        loss = criterion(cls_probs_reshape, masks_val)
    elif args.loss_type == 'dice_BCE' or args.loss_type == 'dice_BCE_kl' or args.loss_type == 'dice_BCE_voted':
        soft_dice = compute_dice(cls_probs_reshape, masks_val.long())
        BCE = criterion(cls_probs_reshape, masks_val)
        loss = BCE + (1-soft_dice.mean())
    elif args.loss_type == 'dice_BCE_likelihood':
        soft_dice = compute_dice(cls_probs_reshape, masks_val.long())
        BCE = criterion(cls_probs_reshape, masks_val)
        loss = BCE + (1-soft_dice.mean())-mll(preds_val, masks_val.view(-1))
    else:
        raise ValueError('Loss type not recognized')


    nll = BinarySegmentationNLLMetric(cls_probs_reshape, masks_val)

    for _i in range(masks_val.shape[0]):
        ece.append(binary_calibration_error(cls_probs_reshape[_i,...], masks_val[_i,...].int(), n_bins=15, norm='l1').item())


    hard_dice = compute_dice(binary_output.reshape(masks_val.shape[0],masks_val.shape[1],masks_val.shape[2]), masks_val.long())
    # calculate IoU
    hard_iou = compute_iou(binary_output.reshape(masks_val.shape[0],masks_val.shape[1],masks_val.shape[2]), masks_val.long())
    
    hd_distances_val, hd95_distances_val = compute_hausdorff_distances(binary_output.reshape(masks_val.shape[0],masks_val.shape[1],masks_val.shape[2]), masks_val.long())
                
    return loss,nll,ece,hard_dice,hard_iou,hd_distances_val, hd95_distances_val

def list_to_numpy(metrics,hyperpara, metrics_addnoise,index_non_empty,args):

    dice_epoch_non_empty = [metrics['dice'][i] for i in range(len(metrics['dice'])) if index_non_empty[i] == 1]
    iou_epoch_non_empty = [metrics['iou'][i] for i in range(len(metrics['iou'])) if index_non_empty[i] == 1]
        
    dice_epoch_non_empty = np.array(dice_epoch_non_empty, dtype='float32')
    iou_epoch_non_empty = np.array(iou_epoch_non_empty, dtype='float32')

    

    metrics['dice'] = np.array(metrics['dice'], dtype='float32')
    metrics['iou'] = np.array(metrics['iou'], dtype='float32')
    metrics['nll'] = np.array(metrics['nll'], dtype='float32')
    metrics['hd'] = np.array(metrics['hd'], dtype='float32')
    metrics['hd95'] = np.array(metrics['hd95'], dtype='float32')
    metrics['ece'] = np.array(metrics['ece'], dtype='float32')
    metrics['loss'] = np.array(metrics['loss'], dtype='float32')
    metrics['dice_non_empty'] = dice_epoch_non_empty
    metrics['iou_non_empty'] = iou_epoch_non_empty

    hyperpara['outputscale'] = np.array(hyperpara['outputscale'], dtype='float32')
    hyperpara['lengthscale'] = np.array(hyperpara['lengthscale'], dtype='float32')
    hyperpara['variance'] = np.array(hyperpara['variance'], dtype='float32')

    if len(metrics_addnoise['dice'])>0:

        dice_epoch_non_empty_addnoise = [metrics_addnoise['dice'][i] for i in range(len(metrics_addnoise['dice'])) if index_non_empty[i] == 1]
        iou_epoch_non_empty_addnoise = [metrics_addnoise['iou'][i] for i in range(len(metrics_addnoise['iou'])) if index_non_empty[i] == 1]

        dice_epoch_non_empty_addnoise = np.array(dice_epoch_non_empty_addnoise, dtype='float32')
        iou_epoch_non_empty_addnoise = np.array(iou_epoch_non_empty_addnoise, dtype='float32')

        metrics_addnoise['dice'] = np.array(metrics_addnoise['dice'], dtype='float32')
        metrics_addnoise['iou'] = np.array(metrics_addnoise['iou'], dtype='float32')
        metrics_addnoise['nll'] = np.array(metrics_addnoise['nll'], dtype='float32')
        metrics_addnoise['hd'] = np.array(metrics_addnoise['hd'], dtype='float32')
        metrics_addnoise['hd95'] = np.array(metrics_addnoise['hd95'], dtype='float32')
        metrics_addnoise['ece_addnoise'] = np.array(metrics_addnoise['ece_addnoise'], dtype='float32')
        metrics_addnoise['loss'] = np.array(metrics_addnoise['loss'], dtype='float32')
        metrics_addnoise['dice_non_empty'] = dice_epoch_non_empty_addnoise
        metrics_addnoise['iou_non_empty'] = iou_epoch_non_empty_addnoise

        hyperpara['noise'] = np.array(hyperpara['noise'], dtype='float32')
        hyperpara['mu'] = np.array(hyperpara['mu'], dtype='float32')
    # else:
    #     metrics_addnoise = None
    #     hyperpara['noise'] = None

    
    return metrics,hyperpara, metrics_addnoise

def list_to_numpy_plot(metrics_addnoise):


    metrics_addnoise['dice'] = np.array(metrics_addnoise['dice'], dtype='float32')
    metrics_addnoise['nll'] = np.array(metrics_addnoise['nll'], dtype='float32')
    metrics_addnoise['hd95'] = np.array(metrics_addnoise['hd95'], dtype='float32')
    metrics_addnoise['ece_addnoise'] = np.array(metrics_addnoise['ece_addnoise'], dtype='float32')

    
    return metrics_addnoise

def list_to_numpy_unet(metrics,index_non_empty):

    dice_epoch_non_empty = [metrics['dice'][i] for i in range(len(metrics['dice'])) if index_non_empty[i] == 1]
    iou_epoch_non_empty = [metrics['iou'][i] for i in range(len(metrics['iou'])) if index_non_empty[i] == 1]
        
    dice_epoch_non_empty = np.array(dice_epoch_non_empty, dtype='float32')
    iou_epoch_non_empty = np.array(iou_epoch_non_empty, dtype='float32')

    metrics['dice'] = np.array(metrics['dice'], dtype='float32')
    metrics['iou'] = np.array(metrics['iou'], dtype='float32')
    metrics['nll'] = np.array(metrics['nll'], dtype='float32')
    metrics['hd'] = np.array(metrics['hd'], dtype='float32')
    metrics['hd95'] = np.array(metrics['hd95'], dtype='float32')
    metrics['ece'] = np.array(metrics['ece'], dtype='float32')
    metrics['loss'] = np.array(metrics['loss'], dtype='float32')
    metrics['dice_non_empty'] = dice_epoch_non_empty
    metrics['iou_non_empty'] = iou_epoch_non_empty

    return metrics

         


def get_metrics_all(model,args,likelihood,preds_val,masks_val,criterion,mll,metrics,hyperpara, metrics_addnoise,index_non_empty,selected_observation = None): 
    
    if args.addnoise == 'add':
        output_addnoise = likelihood(preds_val,'add')   # Get classification predictions
        output = likelihood(preds_val,'noadd')
        loss,nll,metrics['ece'],hard_dice,hard_iou,hd_distances_val, hd95_distances_val = get_metrics(output,masks_val,args,preds_val,criterion,mll,metrics['ece'])
        loss_addnoise,nll_addnoise,metrics_addnoise['ece_addnoise'],hard_dice_addnoise,hard_iou_addnoise,hd_distances_val_addnoise, hd95_distances_val_addnoise = get_metrics(output_addnoise,masks_val,args,preds_val,criterion,mll,metrics_addnoise['ece_addnoise'])
        
        hyperpara['noise'].append(likelihood.noise_covar.noise.item())

        metrics['loss'].append(loss.item())
        metrics['dice'] += hard_dice.tolist()
        metrics['iou'] += hard_iou.tolist()
        metrics['nll'] += nll.tolist()
        metrics['hd95'] += hd95_distances_val.tolist()
        metrics['hd'] += hd_distances_val.tolist()
        

        metrics_addnoise['loss'].append(loss_addnoise.item())
        metrics_addnoise['dice'] += hard_dice_addnoise.tolist()
        metrics_addnoise['iou'] += hard_iou_addnoise.tolist()
        metrics_addnoise['nll'] += nll_addnoise.tolist()
        metrics_addnoise['hd95'] += hd95_distances_val_addnoise.tolist()
        metrics_addnoise['hd'] += hd_distances_val_addnoise.tolist()
    
    elif args.addnoise == 'add3sigmas' or args.addnoise == 'add3sigmas3mus':
        output_addnoise = likelihood(preds_val,args.addnoise,selected_observation,args.num_annotators,len(selected_observation))   # Get classification predictions        
        output = likelihood(preds_val,'noadd')
        loss_addnoise,nll_addnoise,metrics_addnoise['ece_addnoise'],hard_dice_addnoise,hard_iou_addnoise,hd_distances_val_addnoise, hd95_distances_val_addnoise = get_metrics(output_addnoise,masks_val,args,preds_val,criterion,mll,metrics_addnoise['ece_addnoise'])
        loss,nll,metrics['ece'],hard_dice,hard_iou,hd_distances_val, hd95_distances_val = get_metrics(output,masks_val,args,preds_val,criterion,mll,metrics['ece'])

        hyperpara['noise'].append(likelihood.noise[selected_observation[0]].item())
        if args.addnoise == 'add3sigmas3mus':
            hyperpara['mu'].append(likelihood.noise[args.num_annotators+selected_observation[0]].item())
        if args.addnoise == 'add_1_bias_variance':
            hyperpara['mu'].append(likelihood.noise[1+selected_observation[0]].item())


        metrics_addnoise['loss'].append(loss_addnoise.item())
        metrics_addnoise['dice'] += hard_dice_addnoise.tolist()
        metrics_addnoise['iou'] += hard_iou_addnoise.tolist()
        metrics_addnoise['nll'] += nll_addnoise.tolist()
        metrics_addnoise['hd95'] += hd95_distances_val_addnoise.tolist()
        metrics_addnoise['hd'] += hd_distances_val_addnoise.tolist()

        metrics['loss'].append(loss.item())
        metrics['dice'] += hard_dice.tolist()
        metrics['iou'] += hard_iou.tolist()
        metrics['nll'] += nll.tolist()
        metrics['hd95'] += hd95_distances_val.tolist()
        metrics['hd'] += hd_distances_val.tolist()
    elif args.addnoise == 'noadd':
        # noiseless
        output = likelihood(preds_val,'noadd')
        loss,nll,metrics['ece'],hard_dice,hard_iou,hd_distances_val, hd95_distances_val = get_metrics(output,masks_val,args,preds_val,criterion,mll,metrics['ece'])
        metrics['loss'].append(loss.item())
        metrics['dice'] += hard_dice.tolist()
        metrics['iou'] += hard_iou.tolist()
        metrics['nll'] += nll.tolist()
        metrics['hd95'] += hd95_distances_val.tolist()
        metrics['hd'] += hd_distances_val.tolist()


    # calculate dice on only non-empty masks
    flattened_masks = masks_val.view(masks_val.size(0), -1) 
    index_non_empty += torch.tensor([torch.unique(flattened_masks[i]).numel() > 1 for i in range(masks_val.size(0))], dtype=torch.float32, device=masks_val.device).tolist()

    hyperpara['outputscale'].append(model.gp_layer.covar_module.outputscale.item())
    hyperpara['lengthscale'].append(model.gp_layer.covar_module.base_kernel.lengthscale.item())
     
    return metrics,hyperpara, metrics_addnoise,index_non_empty

def initialise_metrics():
    loss_epoch,dice_epoch,iou_epoch,hd_epoch,hd95_epoch,ece,nll_epoch,index_non_empty = [],[],[],[],[],[],[],[]
    loss_epoch_addnoise,dice_epoch_addnoise,iou_epoch_addnoise,hd_epoch_addnoise,hd95_epoch_addnoise,ece_addnoise,nll_epoch_addnoise = [],[],[],[],[],[],[]
    noise_epoch,mu_epoch, lengthscale_epoch, outputscale_epoch,variance_epoch = [],[],[],[],[]

    hyperpara = {'outputscale': outputscale_epoch, 'noise': noise_epoch,'mu':mu_epoch, 'lengthscale': lengthscale_epoch, 'variance': variance_epoch}
    metrics = {'loss': loss_epoch, 'dice': dice_epoch, 'iou': iou_epoch, \
               'nll': nll_epoch, 'ece': ece, 'hd': hd_epoch, 'hd95': hd95_epoch}
    metrics_addnoise = {'ece_addnoise':ece_addnoise,'loss': loss_epoch_addnoise, \
                        'dice': dice_epoch_addnoise, 'iou': iou_epoch_addnoise, 'nll': nll_epoch_addnoise, \
                         'hd': hd_epoch_addnoise, 'hd95': hd95_epoch_addnoise}

    return metrics,hyperpara,metrics_addnoise,index_non_empty
    
def initialise_metrics_plot():
    loss_epoch,dice_epoch,iou_epoch,hd_epoch,hd95_epoch,ece,nll_epoch,index_non_empty = [],[],[],[],[],[],[],[]
    loss_epoch_addnoise,dice_epoch_addnoise,iou_epoch_addnoise,hd_epoch_addnoise,hd95_epoch_addnoise,ece_addnoise,nll_epoch_addnoise = [],[],[],[],[],[],[]
    noise_epoch,mu_epoch, lengthscale_epoch, outputscale_epoch,variance_epoch = [],[],[],[],[]

    # hyperpara = {'outputscale': outputscale_epoch, 'noise': noise_epoch,'mu':mu_epoch, 'lengthscale': lengthscale_epoch, 'variance': variance_epoch}
    # metrics = {'loss': loss_epoch, 'dice': dice_epoch, 'iou': iou_epoch, \
    #            'nll': nll_epoch, 'ece': ece, 'hd': hd_epoch, 'hd95': hd95_epoch}
    metrics_addnoise = {'ece_addnoise':ece_addnoise,'loss': loss_epoch_addnoise, \
                        'dice': dice_epoch_addnoise, 'iou': iou_epoch_addnoise, 'nll': nll_epoch_addnoise, \
                         'hd': hd_epoch_addnoise, 'hd95': hd95_epoch_addnoise}

    return metrics_addnoise
    

def get_metrics_all_unet(preds_val,masks_val,criterion,args,metrics_val,index_non_empty_val):

    BCE_val = criterion(preds_val.squeeze(1), masks_val.float())

    # compute metrics
    hard_dice_val = compute_dice(torch.sigmoid(preds_val.squeeze(1))>0.5, masks_val.long())
    soft_dice_val = compute_dice(torch.sigmoid(preds_val.squeeze(1)), masks_val.long())
    iou_val = compute_iou(torch.sigmoid(preds_val.squeeze(1))>0.5, masks_val.long())
    hd_distances_val, hd95_distances_val = compute_hausdorff_distances(torch.sigmoid(preds_val.squeeze(1))>0.5, masks_val.long())

    nll_val = BinarySegmentationNLLMetric(torch.sigmoid(preds_val.squeeze(1)), masks_val)
    for _i in range(masks_val.shape[0]):
        metrics_val['ece'].append(binary_calibration_error(torch.sigmoid(preds_val.squeeze(1))[_i,...], masks_val[_i,...].int(), n_bins=15, norm='l1').item())

    if args.loss_type == 'dice':
        loss_val = (1-soft_dice_val.mean())
    elif args.loss_type == 'BCE':
        loss_val = BCE_val
    elif args.loss_type == 'dice_BCE':
        loss_val = BCE_val + (1-soft_dice_val.mean())

    # calculate dice on only non-empty masks
    flattened_masks_val = masks_val.view(masks_val.size(0), -1) 
    index_non_empty_val += torch.tensor([torch.unique(flattened_masks_val[i]).numel() > 1 for i in range(masks_val.size(0))], dtype=torch.float32, device=masks_val.device).tolist()

    metrics_val['loss'].append(loss_val.item())
    metrics_val['dice'] += hard_dice_val.tolist()
    metrics_val['iou'] += iou_val.tolist()
    metrics_val['nll'] += nll_val.tolist()
    metrics_val['hd'] += hd_distances_val.tolist()
    metrics_val['hd95'] += hd95_distances_val.tolist()

    return metrics_val,index_non_empty_val


def metrics_plot(sigma_all,mu_all,all_metrics,metrics_name,save_path):

    X = sigma_all
    Y = mu_all
    X, Y = np.meshgrid(X, Y,indexing='ij')
    Z = np.zeros((X.shape[0],X.shape[1]))
    for i,sigma in enumerate(sigma_all):
        for j,mu in enumerate(mu_all):
            Z[i,j] = all_metrics[(sigma.item(),mu.item())][metrics_name].mean()
    
    plt.figure(figsize=(8, 6))
    # plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')  # Use z as color
    plt.imshow(Z.T)
    # print(Z.T.shape)
    fontsize=20
    plt.xticks(range(0,len(sigma_all),2),[f"{i:.3f}" for i in sigma_all.numpy()[::2]],fontsize=fontsize)
    plt.yticks(range(0,len(mu_all),2),[f"{i:.3f}" for i in mu_all.numpy()[::2]],fontsize=fontsize)
    plt.colorbar()
    plt.xlabel('Sigma',fontsize=fontsize)
    plt.ylabel('Mu',fontsize=fontsize)
    plt.title(metrics_name,fontsize=fontsize)
    plt.savefig(save_path + '/'+metrics_name+'.png')
    plt.savefig(save_path + '/'+metrics_name+'.pdf')
    plt.savefig(save_path + '/'+metrics_name+'.eps')
    plt.close()

def metrics_plot_each_image(sigma_all,mu_all,all_metrics,metrics_name,save_path,img_no):
    
    # matplotlib.rcParams['mathtext.fontset'] = 'cm'
    # matplotlib.rc('font', family='serif', serif='CMU Serif')
 
    X = sigma_all
    Y = mu_all
    X, Y = np.meshgrid(X, Y,indexing='ij')
    Z = np.zeros((X.shape[0],X.shape[1]))
    for i,sigma in enumerate(sigma_all):
        for j,mu in enumerate(mu_all):
            Z[i,j] = all_metrics[(sigma.item(),mu.item())][metrics_name].item()
    
    # plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')  # Use z as color
    im=plt.imshow(Z.T,origin='lower',norm=mcolors.LogNorm())
    # im=plt.imshow(np.log(Z.T),origin='lower')
    # plt.xlim(0,len(sigma_all)-1)
    # plt.ylim(0,len(mu_all)-1)
    # print(Z.T.shape)
    fontsize=40
    # plt.xticks(range(0,len(sigma_all),2),[f"{i:.2f}" for i in sigma_all.numpy()[::2]],fontsize=fontsize)
    # plt.yticks(range(0,len(mu_all),2),[f"{i:.2f}" for i in mu_all.numpy()[::2]],fontsize=fontsize)
    # plt.xticks([0,int((len(sigma_all)-1)/2),len(sigma_all)-1],[f"{i:.2f}" for i in sigma_all.numpy()[[0,int(len(sigma_all)/2),len(sigma_all)-1]]],fontsize=fontsize)
    # plt.yticks([0,int((len(mu_all)-1)/2),len(mu_all)-1],[f"{i:.2f}" for i in mu_all.numpy()[[0,int(len(mu_all)/2),len(mu_all)-1]]],fontsize=fontsize)
 
    # plt.xticks([0,len(sigma_all)-1],[f"{i:.1f}" for i in sigma_all.numpy()[[0,len(sigma_all)-1]]],fontsize=fontsize)
    # plt.yticks([0,len(mu_all)-1],[f"{i:.1f}" for i in mu_all.numpy()[[0,len(mu_all)-1]]],fontsize=fontsize)
    # plt.xticks([])
    # plt.yticks([])

    # plt.colorbar()
    # cbar = plt.colorbar()
    
    cbar = plt.colorbar(im)

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)

    # for t in cbar.ax.get_yticklabels():
    #     t.set_fontsize(19)
    plt.xlabel('$\sigma$',fontsize=fontsize)
    plt.ylabel('$\mu$',fontsize=fontsize)
    # plt.title(metrics_name,fontsize=fontsize)
    plt.savefig(save_path + '/'+metrics_name+'_image'+str(img_no)+'.png')
    plt.savefig(save_path + '/'+metrics_name+'_image'+str(img_no)+'.pdf')
    plt.savefig(save_path + '/'+metrics_name+'_image'+str(img_no)+'.eps')
    plt.close()
 

def metrics_plot_2(sigma_all,mu_all,all_metrics,metrics_name,save_path):

    X = sigma_all
    Y = mu_all
    X, Y = np.meshgrid(X, Y,indexing='ij')
    Z = np.zeros(X.shape[0]*X.shape[1])
    for i,(sigma,mu) in enumerate(zip(sigma_all,mu_all)):
            Z[i] = all_metrics[(sigma.item(),mu.item())][metrics_name].mean()
    Z = Z.reshape(X.shape[0],X.shape[1])
    
    plt.figure(figsize=(8, 6))
    # plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')  # Use z as color
    plt.imshow(Z.T)
    # print(Z.T.shape)
    fontsize=20
    plt.xticks(range(0,len(sigma_all),2),[f"{i:.3f}" for i in sigma_all.numpy()[::2]],fontsize=fontsize)
    plt.yticks(range(0,len(mu_all),2),[f"{i:.3f}" for i in mu_all.numpy()[::2]],fontsize=fontsize)
    plt.colorbar()
    plt.xlabel('Sigma',fontsize=fontsize)
    plt.ylabel('Mu',fontsize=fontsize)
    plt.title(metrics_name,fontsize=fontsize)
    plt.savefig(save_path + '/'+metrics_name+'.png')
    plt.savefig(save_path + '/'+metrics_name+'.pdf')
    plt.savefig(save_path + '/'+metrics_name+'.eps')
    plt.close()

def get_metrics_all_each_image(args,likelihood,preds_val,masks_val,criterion,mll, metrics_addnoise,selected_observation = None):
 
    output_addnoise = likelihood(preds_val,args.addnoise,selected_observation,args.num_annotators,len(selected_observation))   # Get classification predictions
    loss_addnoise,nll_addnoise,metrics_addnoise['ece_addnoise'],hard_dice_addnoise,hard_iou_addnoise,hd_distances_val_addnoise, hd95_distances_val_addnoise = get_metrics(output_addnoise,masks_val,args,preds_val,criterion,mll,metrics_addnoise['ece_addnoise'])
 
    metrics_addnoise['dice'] += hard_dice_addnoise.tolist()
    metrics_addnoise['nll'] += nll_addnoise.tolist()
    metrics_addnoise['hd95'] += hd95_distances_val_addnoise.tolist()
 
             
    return metrics_addnoise
 


def plot_img_and_annotations(image_val,mask_wo_noise,mask_all,binary_output_0,binary_output_1,binary_output_2,saved_path,ii):
    # colors = sns.color_palette("colorblind")
    # colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"]
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:olive','tab:cyan','tab:gray']
    
    fig, ax = plt.subplots(1, 1, figsize=(40, 10))

    Linewidth = 1.5

    ax.imshow(image_val[0],'gray')
    plt.axis('off')


    if len(torch.unique(mask_all[0,0]))==2:
        contours = measure.find_contours(mask_all[0,0].cpu().detach().numpy(), level=0.5)[0]
        ax.plot(contours[:, 1], contours[:, 0], linewidth=Linewidth, label='Annotation 1',color = colors[0])

    if len(torch.unique(mask_all[0,1]))==2:
        contours = measure.find_contours(mask_all[0,1].cpu().detach().numpy(), level=0.5)[0]
        ax.plot(contours[:, 1], contours[:, 0], linewidth=Linewidth, label='Annotation 2',color = colors[2])

    if len(torch.unique(mask_all[0,2]))==2:
        contours = measure.find_contours(mask_all[0,2].cpu().detach().numpy(), level=0.5)[0]
        ax.plot(contours[:, 1], contours[:, 0], linewidth=Linewidth, label='Annotation 3',color = colors[3])

    
    if len(torch.unique(binary_output_0))==2:
        contours = measure.find_contours(binary_output_0.reshape(image_val.shape[1],image_val.shape[2]).cpu().detach().numpy(), level=0.5)[0]
        ax.plot(contours[:, 1], contours[:, 0], linewidth=2,linestyle = '--', label='SVGP prediction for Annotation 1',color = colors[0])

    if len(torch.unique(binary_output_1))==2:
        contours = measure.find_contours(binary_output_1.reshape(image_val.shape[1],image_val.shape[2]).cpu().detach().numpy(), level=0.5)[0]
        ax.plot(contours[:, 1], contours[:, 0], linewidth=2,linestyle = '--', label='SVGP prediction for Annotation 2',color = colors[2])

    if len(torch.unique(binary_output_2))==2:
        contours = measure.find_contours(binary_output_2.reshape(image_val.shape[1],image_val.shape[2]).cpu().detach().numpy(), level=0.5)[0]
        ax.plot(contours[:, 1], contours[:, 0], linewidth=2,linestyle = '--', label='SVGP prediction for Annotation 3',color = colors[3])

    ax.legend(ncol=6, loc='center', bbox_to_anchor=(0.5, 1.065),fontsize=33)
    
    plt.savefig(saved_path+'/'+'image_'+str(ii)+".png")
    plt.savefig(saved_path+'/'+'image_'+str(ii)+".pdf")
    plt.savefig(saved_path+'/'+'image_'+str(ii)+".eps")
    plt.close()
 
 

def ploting(output_addnoise,saved_path,images,masks,sigma,mu):
    prediction = output_addnoise.mean.ge(0.5).float()
    images, masks = images.cpu().detach().numpy(), masks.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    plt.imshow(images[0],'gray')
    plt.contour(masks[0],colors='red')
    plt.contour(prediction.reshape(images.shape[0],images.shape[1],images.shape[2])[0],colors='blue')

    visualization_path = os.path.join(saved_path,'plots')
    plt.savefig(visualization_path+'/'+ "sigma_"+'%.3f'%sigma+"_mu_"+'%.3f'%mu+".png")
    plt.close()


def unet_linear_add_noise(logits,model,selected_observation,var,Softplus):

    full_mean = logits + model.transform_mu[selected_observation][:,None,None,None]
    full_var = Softplus(var) + model.transform_sigma[selected_observation][:,None,None,None]
    link = full_mean.div(torch.sqrt(1+full_var))
    output_probs = torch.distributions.Normal(0, 1).cdf(link)
    preds_val = torch.distributions.Bernoulli(probs=output_probs).mean

    return preds_val

def unet_linear_without_noise(logits,var,Softplus):

    full_mean = logits
    full_var = Softplus(var)
    link = full_mean.div(torch.sqrt(1+full_var))
    output_probs = torch.distributions.Normal(0, 1).cdf(link)
    preds_val = torch.distributions.Bernoulli(probs=output_probs).mean

    return preds_val

def plot_simulated_labels_and_GT(img,simulated,GT,sub_name,saved_label_path):
    saved_path_fig = os.path.join(saved_label_path,'plots',sub_name)
    os.makedirs(saved_path_fig,exist_ok=True)

    for i in range(img.shape[0]):
        plt.figure(figsize=(8, 6))
        plt.imshow(img[i],'gray')
        plt.contour(GT[i],colors='red')
        plt.contour(simulated[i],colors='blue')
        plt.savefig(saved_path_fig + '/img_'+str(i)+'.png')
        plt.close()

