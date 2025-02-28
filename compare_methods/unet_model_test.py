

import os
import torch
import csv
import sys
system_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(system_path)
from utils.utils import *
from utils.data_loader import DataSet
from utils.base_network import UNet
import torch.nn as nn
from torchmetrics.functional.classification import binary_calibration_error
import pickle 
from scipy.spatial.distance import directed_hausdorff

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


def test(test_loader,args,device,folder):

    model = UNet(n_channels = 1, n_classes=args.num_class, feat_dim = args.feat_dim, FeatBN = args.FeatBatchNorm)  
    model.to(device)
    model,saved_model_name,epoch = load_best_model_and_epoch_unet(model,args,device,folder)
    criterion = nn.BCEWithLogitsLoss()
    model.train(False)

    with torch.no_grad():

        metrics_vote,_,_,index_non_empty_vote = initialise_metrics()
        metrics_0,_,_,index_non_empty_0 = initialise_metrics()
        metrics_1,_,_,index_non_empty_1 = initialise_metrics()
        metrics_2,_,_,index_non_empty_2 = initialise_metrics()


        for ii, (image_val, mask_all_val, mask_wo_noise, selected_observation) in enumerate(test_loader):
            images_val, masks_val, mask_wo_noise = image_val.float().to(device), mask_all_val.float().to(device), mask_wo_noise.float().to(device)
            preds_val = model(images_val[:,None,...])
            if args.pre_labels == 'co' or args.pre_labels == 'seg_simulated':
                metrics_vote,index_non_empty_vote = get_metrics_all_unet(preds_val,mask_wo_noise,criterion,args,metrics_vote,index_non_empty_vote)
            
            elif args.pre_labels == 'random3':
                metrics_vote,index_non_empty_vote = get_metrics_all_unet(preds_val,mask_wo_noise,criterion,args,metrics_vote,index_non_empty_vote)
                metrics_0,index_non_empty_0 = get_metrics_all_unet(preds_val,masks_val[:,0,...],criterion,args,metrics_0,index_non_empty_0)
                metrics_1,index_non_empty_1 = get_metrics_all_unet(preds_val,masks_val[:,1,...],criterion,args,metrics_1,index_non_empty_1)
                metrics_2,index_non_empty_2 = get_metrics_all_unet(preds_val,masks_val[:,2,...],criterion,args,metrics_2,index_non_empty_2)

            elif args.pre_labels == 'seg1':
                metrics_vote,index_non_empty_vote = get_metrics_all_unet(preds_val,mask_wo_noise,criterion,args,metrics_vote,index_non_empty_vote)
                metrics_0,index_non_empty_0 = get_metrics_all_unet(preds_val,masks_val[:,0,...],criterion,args,metrics_0,index_non_empty_0)
            
            elif args.pre_labels == 'seg2':
                metrics_vote,index_non_empty_vote = get_metrics_all_unet(preds_val,mask_wo_noise,criterion,args,metrics_vote,index_non_empty_vote)
                metrics_1,index_non_empty_1 = get_metrics_all_unet(preds_val,masks_val[:,1,...],criterion,args,metrics_1,index_non_empty_1)
                metrics_0,index_non_empty_0 = get_metrics_all_unet(preds_val,masks_val[:,0,...],criterion,args,metrics_0,index_non_empty_0)

            elif args.pre_labels == 'seg3':
                metrics_vote,index_non_empty_vote = get_metrics_all_unet(preds_val,mask_wo_noise,criterion,args,metrics_vote,index_non_empty_vote)
                metrics_2,index_non_empty_2 = get_metrics_all_unet(preds_val,masks_val[:,2,...],criterion,args,metrics_2,index_non_empty_2)



        if args.pre_labels == 'co' or args.pre_labels == 'seg_simulated':     
            metrics_vote = list_to_numpy_unet(metrics_vote,index_non_empty_vote)
            
            return metrics_vote,saved_model_name

        elif args.pre_labels == 'random3' or args.pre_labels == 'random4':
            metrics_vote = list_to_numpy_unet(metrics_vote,index_non_empty_vote)
            metrics_0 = list_to_numpy_unet(metrics_0,index_non_empty_0)
            metrics_1 = list_to_numpy_unet(metrics_1,index_non_empty_1)
            metrics_2 = list_to_numpy_unet(metrics_2,index_non_empty_2)
            metrics_all = {'vote':metrics_vote,'0':metrics_0,'1':metrics_1,'2':metrics_2}
            
            return metrics_all,saved_model_name
        
        elif args.pre_labels == 'seg1':
            metrics_vote = list_to_numpy_unet(metrics_vote,index_non_empty_vote)
            metrics_0 = list_to_numpy_unet(metrics_0,index_non_empty_0)
            metrics_all = {'vote':metrics_vote,'0':metrics_0}
            
            return metrics_all,saved_model_name
        elif args.pre_labels == 'seg2':
            metrics_vote = list_to_numpy_unet(metrics_vote,index_non_empty_vote)
            metrics_1 = list_to_numpy_unet(metrics_1,index_non_empty_1)
            metrics_0 = list_to_numpy_unet(metrics_0,index_non_empty_0)
            metrics_all = {'vote':metrics_vote,'1':metrics_1,'0':metrics_0}
            
            return metrics_all,saved_model_name
        elif args.pre_labels == 'seg3':
            metrics_vote = list_to_numpy_unet(metrics_vote,index_non_empty_vote)
            metrics_2 = list_to_numpy_unet(metrics_2,index_non_empty_2)
            metrics_all = {'vote':metrics_vote,'2':metrics_2}
            
            return metrics_all,saved_model_name



        


models_folders = 'previous_trained_models_UnetSimulated20250221_test_on_corrected_labels'
high_quality_labels = True

all_test_folders_path = os.path.join(os.getcwd(),models_folders)

all_test_folders = [f for f in os.listdir(all_test_folders_path) if os.path.isdir(os.path.join(all_test_folders_path, f)) and f.startswith('Unet')]
all_test_folders = sorted(all_test_folders)
csv_name = 'metrics_unet.csv'
csv_name_error = 'error_unet.csv'
fd_name_save = 'results'+'/'+models_folders

# csv file
if not os.path.exists(os.getcwd()+'/'+fd_name_save):
    os.makedirs(os.getcwd()+'/'+fd_name_save)

with open(os.getcwd()+'/'+ fd_name_save +'/'+csv_name_error, 'a', encoding='UTF8') as f_error:
    writer_error = csv.writer(f_error)
    writer_error.writerow(['error_folder:\n'])

with open(os.getcwd()+'/'+ fd_name_save +'/'+csv_name, 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name','model_name',\
                    'Dice & ','Non-empty-dice & '\
                    'IoU & ','Non-empty-IoU & '\
                    'hd & ',\
                    'hd95 & '
                    'ECE & ',\
                    'NLL & ',\
                    
                    ])


for folder in all_test_folders:
    # get args
    try:
        if 'args.txt' in os.listdir(os.path.join(all_test_folders_path,folder)): 
            args = get_args_unet(os.path.join(all_test_folders_path,folder,'args.txt'),folder)
            args.batch_size_val = 2
            args.saved_name = models_folders+'/'+args.saved_name
            args.pre_labels = args.labels
            args.labels = 'inference'
            if high_quality_labels:
                args.labels_infer = 'high_quality'

        else:
            # get args from folder name
            args = get_args_from_folder_unet(folder)
            args.batch_size_val = 2
            args.pre_labels = args.labels
            args.labels = 'inference'

    except:
        with open(os.getcwd()+'/'+ fd_name_save +'/'+csv_name_error, 'a', encoding='UTF8') as f_error:
            f_error.write(f"Error in {folder}")
            f_error.write('\n')
        continue


    print(f"Seeding with seed: {args.seed}")
    seed_all(int(args.seed))
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cuda_avail, device = torch_init(args.device)
    print("pytorch using device", device)

    # load data    
    try:
        dataset = DataSet(datapath=args.dataset,num_class=args.num_class,labels=args.labels,labels_inference = args.labels_infer,data_aug=args.data_aug,
                    simulated_model=args.simulated_model,sigma=float(args.sigma_simulated_label),mu=float(args.mu_simulated_label))
    except:
        dataset = DataSet(datapath=args.dataset,num_class=args.num_class,labels=args.labels,labels_inference = args.labels_infer,data_aug=args.data_aug)

    # dataset.split_data()
    _,_,test_set = dataset.load_data_split(os.path.join(args.dataset,'data_split','train_list.json'),os.path.join(args.dataset,'data_split','val_list.json'),os.path.join(args.dataset,'data_split','test_list.json'))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(args.batch_size_val), shuffle=False, num_workers=8)


        
    # try:
    if True:
        metrics,saved_model_name = test(test_loader,args,device,models_folders+'/'+folder)
        # save metrics and hyperparameters
        save_metrics_unet(os.getcwd()+'/'+ fd_name_save +'/'+csv_name,folder,metrics,saved_model_name,args)
        with open(os.getcwd()+'/'+ fd_name_save +'/'+ folder +'_metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f) 
    # except:
    #     with open(os.getcwd()+'/'+ fd_name_save +'/'+csv_name_error, 'a', encoding='UTF8') as f_error:
    #         f_error.write(f"Error in {folder}")
    #         f_error.write('\n')
    #     continue
        
        






    



    

  