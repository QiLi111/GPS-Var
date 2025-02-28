
# Unet
import os
import argparse
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics.functional.classification import binary_calibration_error
system_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(system_path)
from utils.utils import *
from torch.utils.tensorboard import SummaryWriter
from utils.data_loader import DataSet
from utils.base_network import UNet_linear

parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
parser.add_argument("--dataset", type=str,default="./data")

parser.add_argument("--num_epochs", type=int, default=500000)
parser.add_argument('--num_class', type=int, default=1, help='The size of class')
parser.add_argument("--batch_size_train", type=int, default=8)
parser.add_argument("--batch_size_val", type=int, default=8)
# 'co: using only consistent label in GP train and GP loss calculation' 
# all: all three labels for GP train, the consistent label for cross entropy loss calculation
# 'seg1': using only labels form the first annotator
# 'seg2': using only labels form the second annotator
# 'seg3': using only labels form the third annotator
# 'random3': randomly select from three avaliable labels;
parser.add_argument("--labels", type=str, default="random3") 
# 'voted_only': using only voted label for inference; 'high_quality': using corrected voted labels for inference
parser.add_argument("--labels_infer", type=str, default="voted_only") 
parser.add_argument("--loss_type", type=str, default="dice_BCE") # dice; BCE; dice_BCE
parser.add_argument("--loss_add_mu_reg", type=str, default="sum_2") # 'add' # 'noadd' # sum_2
parser.add_argument("--feat_dim", type=int, default=64)
parser.add_argument("--FeatBatchNorm", type=str, default="noadd") # 'add' # 'noadd'
parser.add_argument("--data_aug", type=str, default="noadd") # 'add' # 'noadd'; if add data augmentation

parser.add_argument("--seed", type=int,default=0)
parser.add_argument("--cp_path", type=str, default="checkpoints")
parser.add_argument("--plot_path", type=str, default="visulisation")
parser.add_argument("--saved_path", type=str, default="saved_path")
parser.add_argument("--FREQ_SAVE", type=int, default=5)
parser.add_argument("--FREQ_VISULISE", type=int, default=5)

parser.add_argument("-d", "--device", dest="device", help="Device to run on, the cpu or gpu.", type=str, default="cuda:0")

args = parser.parse_args()


def visualize(args, images, masks, prediction,epoch,step,visualization_path):
        
    if epoch in range(0, args.num_epochs, args.FREQ_VISULISE) and step in range(0, args.num_epochs, 50):
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

            plt.savefig(os.path.join(visualization_path,'best_dice', f"epoch_{epoch}_step_{step}_no_{i}.png"))
            plt.close()


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
    pred = pred.cpu()
    target = target.cpu()
    B, H, W = pred.shape
            
    hd_distances = []  # Hausdorff distances
    hd95_distances = []  # 95th percentile distances

    for b in range(B):
        # Get foreground indices as tensors
        pred_indices = torch.nonzero(pred[b], as_tuple=False).float()
        target_indices = torch.nonzero(target[b], as_tuple=False).float()

        if pred_indices.numel() == 0 and target_indices.numel() == 0:
            # # Handle empty masks (either predicted or ground truth)
            # hd_distances[b] = float(0)
            # hd95_distances[b] = float(0)
            continue
        elif pred_indices.numel() == 0 and target_indices.numel() != 0:
            # # Handle empty predicted mask
            # if target_indices.shape[0] < 2:
            #     return float(0)
            
            # dists = torch.cdist(target_indices, target_indices, p=2)
            # d1_max = torch.max(torch.min(dists, dim=1).values)
            # d2_max = torch.max(torch.min(dists, dim=0).values)
            # hd_distances[b] = torch.max(d1_max, d2_max)

            # d1_95 = torch.quantile(torch.min(dists, dim=1).values, 0.95) 
            # d2_95 = torch.quantile(torch.min(dists, dim=0).values, 0.95)
            # hd95_distances[b] = torch.max(d1_95, d2_95)

            continue
        elif pred_indices.numel() != 0 and target_indices.numel() == 0:
            # # Handle empty ground truth mask
            # if pred_indices.shape[0] < 2:
            #     return float(0)
            
            # dists = torch.cdist(pred_indices, pred_indices, p=2)
            # d1_max = torch.max(torch.min(dists, dim=1).values)
            # d2_max = torch.max(torch.min(dists, dim=0).values)
            # hd_distances[b] = torch.max(d1_max, d2_max)

            # d1_95 = torch.quantile(torch.min(dists, dim=1).values, 0.95) 
            # d2_95 = torch.quantile(torch.min(dists, dim=0).values, 0.95)
            # hd95_distances[b] = torch.max(d1_95, d2_95)

            continue

        # Compute pairwise distances
        pred_dists = torch.cdist(pred_indices, target_indices, p=2)  # Euclidean distances
        target_dists = torch.cdist(target_indices, pred_indices, p=2)  # Reverse distances

        # Hausdorff distance (maximum distance in either direction)
        d1_max = torch.max(torch.min(pred_dists, dim=1).values)  # Max over pred -> target
        d2_max = torch.max(torch.min(target_dists, dim=1).values)  # Max over target -> pred
        hd_distances.append(torch.max(d1_max, d2_max).item())

        # 95th percentile Hausdorff distance
        d1_95 = torch.quantile(torch.min(pred_dists, dim=1).values, 0.95)  # 95th percentile
        d2_95 = torch.quantile(torch.min(target_dists, dim=1).values, 0.95)  # 95th percentile
        hd95_distances.append(torch.max(d1_95, d2_95).item())

    hd_distances = torch.tensor(hd_distances)
    hd95_distances = torch.tensor(hd95_distances)
    
    return hd_distances, hd95_distances


def main(args):
    print(f"Seeding with seed: {args.seed}")
    seed_all(args.seed)
    cuda_avail, device = torch_init(args.device)
    print("pytorch using device", device)

    args = get_saved_folder_name_unet_linear_model(args)
    os.makedirs(os.path.join(args.saved_name,args.plot_path),exist_ok=True)
    os.makedirs(os.path.join(args.saved_name,args.cp_path),exist_ok=True)
    os.makedirs(os.path.join(args.saved_name,args.saved_path),exist_ok=True)
    os.makedirs(os.path.join(args.saved_name,args.saved_path,'saved_model'),exist_ok=True)
    writer = SummaryWriter(os.path.join(args.saved_name,args.cp_path))
    save_arguments(args)

    ## training
    model = UNet_linear(n_channels = 1, n_classes=args.num_class, feat_dim = args.feat_dim, FeatBN = args.FeatBatchNorm)  
    model.to(device)
    criterion = nn.BCELoss()

    dataset = DataSet(datapath=args.dataset,num_class=args.num_class,labels=args.labels,labels_inference = args.labels_infer,data_aug=args.data_aug)
    train_set,val_set,test_set = dataset.load_data_split(os.path.join(args.dataset,'data_split','train_list.json'),os.path.join(args.dataset,'data_split','val_list.json'),os.path.join(args.dataset,'data_split','test_list.json'))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_train, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size_val, shuffle=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_val, shuffle=False, num_workers=8)

    train(model, device, args, train_loader,val_loader, writer,criterion)


def train(model, device, args, train_loader,val_loader, writer,criterion):
    epoch = -1
    best_dice = -10e6
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    Softplus = nn.Softplus()
    # lower_bound_sigma = 1e-4
    # lower_bound_mu = -100.0
    # upper_bound_mu = 100.0
    # mu_activation = lambda x,lower_bound, upper_bound: torch.sigmoid(x)*(upper_bound-lower_bound) + lower_bound

    

    while epoch < args.num_epochs:
        epoch += 1

        # training
        model.train(True)
        loss_epoch,dice_epoch,iou_epoch,hd_epoch,hd95_epoch,ece,nll_epoch,index_non_empty = [],[],[],[],[],[],[],[]
        noise_epoch0, noise_epoch1, noise_epoch2,mu_epoch0, mu_epoch1, mu_epoch2 = [],[],[],[],[],[]
        
        for ii, (image, mask_all, mask_wo_noise, selected_observation) in enumerate(train_loader):
            
            images, masks, mask_wo_noise = image.float().to(device), mask_all.float().to(device), mask_wo_noise.float().to(device)

            optimizer.zero_grad(set_to_none=True)
            logits,var = torch.chunk(model(images[:,None,...]),2,dim=1)

            # full_mean = logits + mu_activation(model.mu_values[selected_observation][:,None,None,None],lower_bound_mu,upper_bound_mu)
            # full_var = Softplus(var) + Softplus(model.noise_values[selected_observation][:,None,None,None]) + lower_bound_sigma
            
            full_mean = logits + model.transform_mu[selected_observation][:,None,None,None]
            full_var = Softplus(var) + model.transform_sigma[selected_observation][:,None,None,None]
            link = full_mean.div(torch.sqrt(1+full_var))
            output_probs = torch.distributions.Normal(0, 1).cdf(link)
            preds = torch.distributions.Bernoulli(probs=output_probs).mean

            BCE = criterion(preds.squeeze(1), masks.float())
            soft_dice = compute_dice(preds.squeeze(1), masks.long())
            iou = compute_iou(preds.squeeze(1)>0.5, masks.long())
            hard_dice = compute_dice(preds.squeeze(1)>0.5, masks.long())
            
            if args.loss_type == 'dice':
                loss = (1-soft_dice.mean())
            elif args.loss_type == 'BCE':
                loss = BCE
            elif args.loss_type == 'dice_BCE':
                loss = BCE + (1-soft_dice.mean())

            if args.loss_add_mu_reg == 'add':
                loss = loss + torch.sum(model.transform_mu**2) 
            elif args.loss_add_mu_reg == 'sum_2':
                loss = loss + torch.sum(model.transform_mu)**2


            loss.backward()
            optimizer.step()

            with torch.no_grad():
                nll = BinarySegmentationNLLMetric(torch.sigmoid(preds.squeeze(1)), masks)
                for _i in range(masks.shape[0]):
                    ece.append(binary_calibration_error(torch.sigmoid(preds.squeeze(1))[_i,...], masks[_i,...].int(), n_bins=15, norm='l1').item())

                # calculate dice on only non-empty masks
                flattened_masks = masks.view(masks.size(0), -1) 
                index_non_empty += torch.tensor([torch.unique(flattened_masks[i]).numel() > 1 for i in range(masks.size(0))], dtype=torch.float32, device=masks.device).tolist()

                loss_epoch.append(loss.item())
                dice_epoch += hard_dice.tolist()
                iou_epoch += iou.tolist()
                nll_epoch += nll.tolist()

                noise_epoch0.append(model.transform_sigma[0].item())
                noise_epoch1.append(model.transform_sigma[1].item())
                noise_epoch2.append(model.transform_sigma[2].item())
                mu_epoch0.append(model.transform_mu[0].item())
                mu_epoch1.append(model.transform_mu[1].item())
                mu_epoch2.append(model.transform_mu[2].item())


        dice_epoch_non_empty = sum([dice_epoch[i] for i in range(len(dice_epoch)) if index_non_empty[i] == 1]) / sum(index_non_empty)
        iou_epoch_non_empty = sum([iou_epoch[i] for i in range(len(iou_epoch)) if index_non_empty[i] == 1]) / sum(index_non_empty)

        ece_epoch = sum(ece)/len(ece)
        loss_epoch = sum(loss_epoch)/len(loss_epoch)
        dice_epoch = sum(dice_epoch)/len(dice_epoch)
        iou_epoch = sum(iou_epoch)/len(iou_epoch)
        nll_epoch = sum(nll_epoch)/len(nll_epoch)

        noise_epoch0 = sum(noise_epoch0)/len(noise_epoch0)
        noise_epoch1 = sum(noise_epoch1)/len(noise_epoch1)
        noise_epoch2 = sum(noise_epoch2)/len(noise_epoch2)
        mu_epoch0 = sum(mu_epoch0)/len(mu_epoch0)
        mu_epoch1 = sum(mu_epoch1)/len(mu_epoch1)
        mu_epoch2 = sum(mu_epoch2)/len(mu_epoch2)

        metrics_epoch = {'dice': dice_epoch, 'iou': iou_epoch, 'nll': nll_epoch, 'ece': ece_epoch, 'dice_non_empty': dice_epoch_non_empty, 'iou_non_empty': iou_epoch_non_empty}
        hyperpara_train = {'noise0': noise_epoch0, 'noise1': noise_epoch1, 'noise2': noise_epoch2, 'mu0': mu_epoch0, 'mu1': mu_epoch1, 'mu2': mu_epoch2}
        
        # validation
        if epoch in range(0, args.num_epochs, args.FREQ_VISULISE):
            model.train(False)
            with torch.no_grad():

                loss_epoch_val,dice_epoch_val,iou_epoch_val,hd_distances_epoch_val, hd95_distances_epoch_val,ece_val,nll_epoch_val,index_non_empty_val = [],[],[],[],[],[],[],[]
                noise_epoch0_val, noise_epoch1_val, noise_epoch2_val,mu_epoch0_val, mu_epoch1_val, mu_epoch2_val = [],[],[],[],[],[]

                for ii, (image_val, mask_all_val, mask_wo_noise_val, selected_observation_val) in enumerate(val_loader):
                    images_val, masks_val, mask_wo_noise_val = image_val.float().to(device), mask_all_val.float().to(device), mask_wo_noise_val.float().to(device)
                    if args.labels  == 'random3':
                        # use noiseless mask for validation
                        masks_val = mask_wo_noise_val
            
                    logits_val,var_val = torch.chunk(model(images_val[:,None,...]),2,dim=1)

                    # full_mean_val = logits_val + mu_activation(model.mu_values[selected_observation_val][:,None,None,None],lower_bound_mu,upper_bound_mu)
                    # full_var_val = Softplus(var_val) + Softplus(model.noise_values[selected_observation_val][:,None,None,None]) + lower_bound_sigma
                    full_mean_val = logits_val + model.transform_mu[selected_observation_val][:,None,None,None]
                    full_var_val = Softplus(var_val) + model.transform_sigma[selected_observation_val][:,None,None,None]
                    link_val = full_mean_val.div(torch.sqrt(1+full_var_val))
                    output_probs_val = torch.distributions.Normal(0, 1).cdf(link_val)
                    preds_val = torch.distributions.Bernoulli(probs=output_probs_val).mean


                    BCE_val = criterion(preds_val.squeeze(1), masks_val.float())

                    # compute metrics
                    hard_dice_val = compute_dice(torch.sigmoid(preds_val.squeeze(1))>0.5, masks_val.long())
                    soft_dice_val = compute_dice(torch.sigmoid(preds_val.squeeze(1)), masks_val.long())
                    iou_val = compute_iou(torch.sigmoid(preds_val.squeeze(1))>0.5, masks_val.long())
                    # hd_distances_val, hd95_distances_val = compute_hausdorff_distances(F.sigmoid(preds_val.squeeze(1))>0.5, masks_val.long())

                    nll_val = BinarySegmentationNLLMetric(torch.sigmoid(preds_val.squeeze(1)), masks_val)
                    for _i in range(masks_val.shape[0]):
                        ece_val.append(binary_calibration_error(torch.sigmoid(preds_val.squeeze(1))[_i,...], masks_val[_i,...].int(), n_bins=15, norm='l1').item())

                    if args.loss_type == 'dice':
                        loss_val = (1-soft_dice_val.mean())
                    elif args.loss_type == 'BCE':
                        loss_val = BCE_val
                    elif args.loss_type == 'dice_BCE':
                        loss_val = BCE_val + (1-soft_dice_val.mean())

                    if args.loss_add_mu_reg == 'add':
                        loss_val = loss_val + torch.sum(model.transform_mu**2) 
                    elif args.loss_add_mu_reg == 'sum_2':
                        loss_val = loss_val + torch.sum(model.transform_mu)**2

                    # calculate dice on only non-empty masks
                    flattened_masks_val = masks_val.view(masks_val.size(0), -1) 
                    index_non_empty_val += torch.tensor([torch.unique(flattened_masks_val[i]).numel() > 1 for i in range(masks_val.size(0))], dtype=torch.float32, device=masks_val.device).tolist()

                    loss_epoch_val.append(loss_val.item())
                    dice_epoch_val += hard_dice_val.tolist()
                    iou_epoch_val += iou_val.tolist()
                    nll_epoch_val += nll_val.tolist()
                    # hd_distances_epoch_val += hd_distances_val.tolist()
                    # hd95_distances_epoch_val += hd95_distances_val.tolist()

                    noise_epoch0_val.append(model.transform_sigma[0].item())
                    noise_epoch1_val.append(model.transform_sigma[1].item())
                    noise_epoch2_val.append(model.transform_sigma[2].item())
                    mu_epoch0_val.append(model.transform_mu[0].item())
                    mu_epoch1_val.append(model.transform_mu[1].item())
                    mu_epoch2_val.append(model.transform_mu[2].item())

                    visualize(args, images_val, masks_val, torch.sigmoid(preds_val.squeeze(1))>0.5,epoch,ii,os.path.join(args.saved_name,args.plot_path))
 


                dice_epoch_non_empty_val = sum([dice_epoch_val[i] for i in range(len(dice_epoch_val)) if index_non_empty_val[i] == 1]) / sum(index_non_empty_val)
                iou_epoch_non_empty_val = sum([iou_epoch_val[i] for i in range(len(iou_epoch_val)) if index_non_empty_val[i] == 1]) / sum(index_non_empty_val)

                ece_epoch_val = sum(ece_val)/len(ece_val)
                loss_epoch_val = sum(loss_epoch_val)/len(loss_epoch_val)
                dice_epoch_val = sum(dice_epoch_val)/len(dice_epoch_val)
                iou_epoch_val = sum(iou_epoch_val)/len(iou_epoch_val)
                nll_epoch_val = sum(nll_epoch_val)/len(nll_epoch_val)
                # hd_distances_epoch_val = sum(hd_distances_epoch_val)/len(hd_distances_epoch_val)
                # hd95_distances_epoch_val = sum(hd95_distances_epoch_val)/len(hd95_distances_epoch_val)

                noise_epoch0_val = sum(noise_epoch0_val)/len(noise_epoch0_val)
                noise_epoch1_val = sum(noise_epoch1_val)/len(noise_epoch1_val)
                noise_epoch2_val = sum(noise_epoch2_val)/len(noise_epoch2_val)
                mu_epoch0_val = sum(mu_epoch0_val)/len(mu_epoch0_val)
                mu_epoch1_val = sum(mu_epoch1_val)/len(mu_epoch1_val)
                mu_epoch2_val = sum(mu_epoch2_val)/len(mu_epoch2_val)


                # metrics_epoch_val = {'dice': dice_epoch_val, 'iou': iou_epoch_val, 'hd': hd_distances_epoch_val, 'hd95': hd95_distances_epoch_val, 'nll': nll_epoch_val, 'ece': ece_epoch_val, 'dice_non_empty': dice_epoch_non_empty_val, 'iou_non_empty': iou_epoch_non_empty_val}
                metrics_epoch_val = {'dice': dice_epoch_val, 'iou': iou_epoch_val, 'nll': nll_epoch_val, 'ece': ece_epoch_val, 'dice_non_empty': dice_epoch_non_empty_val, 'iou_non_empty': iou_epoch_non_empty_val}
                hyperpara_val = {'noise0': noise_epoch0_val, 'noise1': noise_epoch1_val, 'noise2': noise_epoch2_val, 'mu0': mu_epoch0_val, 'mu1': mu_epoch1_val, 'mu2': mu_epoch2_val}

            best_dice = save_best_model(model,epoch, dice_epoch_val, best_dice, os.path.join(args.saved_name,args.saved_path))
            model.train(True)

        else:
            loss_epoch_val,metrics_epoch_val,hyperpara_val = None,None,None

        add_status(writer, epoch, None,None,loss_epoch,loss_epoch_val,metrics_epoch,metrics_epoch_val )
        save_regular_model(model,epoch, args.num_epochs, args.FREQ_SAVE,os.path.join(args.saved_name,args.saved_path))
        add_scalars_hyperpara_linear_model(epoch,writer,hyperpara_train,hyperpara_val)

#Run main
main(args)