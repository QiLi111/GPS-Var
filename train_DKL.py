
# train script

import torch.optim as optim
import gpytorch
import torch,os
import torch.nn as nn
from utils.utils import compute_dice, compute_iou,compute_hausdorff_distances,BinarySegmentationNLLMetric,BinaryCalibrationError,compute_calibration
from utils.utils import add_scalars_metrics,save_best_model_GP,save_regular_model_GP,add_scalars_hyperpara_GP
from utils.utils import visualize
from torchmetrics.functional.classification import binary_calibration_error


def train_DKL(model, likelihood, device,args,train_loader,val_loader,writer):
    
    lr = 1e-4
    if args.trained_Unet == 'trainU':
        optimizer = optim.AdamW([
                {'params': filter(lambda p: p.requires_grad, model.feature_extractor.parameters()), 'weight_decay': 1e-4},
                {'params': filter(lambda p: p.requires_grad, model.gp_layer.hyperparameters())},
                {'params': filter(lambda p: p.requires_grad, model.gp_layer.variational_parameters())},
                {'params': filter(lambda p: p.requires_grad, likelihood.parameters())},
            ], lr=lr, weight_decay=1e-3)
        
    elif args.trained_Unet == 'notrainU':
    
        optimizer = optim.AdamW([
                {'params': filter(lambda p: p.requires_grad, model.gp_layer.hyperparameters())},
                {'params': filter(lambda p: p.requires_grad, model.gp_layer.variational_parameters())},
                {'params': filter(lambda p: p.requires_grad, likelihood.parameters())},
            ], lr=lr, weight_decay=1e-4)

    lr_sched = optim.lr_scheduler.LambdaLR(optimizer, (lambda n:
                                                       1.0 if n <= 30 else
                                                       0.1))
    if args.nll_type == 'VariationalELBO':
        mll_train = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))
        mll_val = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(val_loader.dataset))
    elif args.nll_type == 'PredictiveLogLikelihood':
        mll_train = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model.gp_layer, num_data=len(train_loader.dataset), beta=0.5)
        mll_val = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model.gp_layer, num_data=len(val_loader.dataset), beta=0.5)

    
    best_dice = -10e6
    criterion = nn.BCELoss()

    if args.retrain == 'load':
        epoch_min = args.retrain_epoch
        epoch_max = args.num_epochs+args.retrain_epoch
    else:
        epoch_min = 0
        epoch_max = args.num_epochs

    for epoch in range(epoch_min, epoch_max):
        with gpytorch.settings.use_toeplitz(False):
            metrics_train, hyperpara_train = train(model,likelihood,train_loader,optimizer,mll_train,device,args.trained_Unet, args.loss_type, criterion,args.addnoise,args.loss_add_mu_reg,args.num_annotators)
            
            if epoch in range(epoch_min, epoch_max, args.FREQ_VISULISE):
                with torch.no_grad():
                    metrics_val, hyperpara_val = val(model,likelihood,val_loader,mll_val,device,epoch,args,criterion)
                add_scalars_metrics(epoch,writer,metrics_train,metrics_val)
                add_scalars_hyperpara_GP(epoch,writer,hyperpara_train,hyperpara_val,args)
                best_dice = save_best_model_GP(model,likelihood,epoch, metrics_val['dice'], best_dice, os.path.join(args.saved_name,args.saved_path))


            else:
                add_scalars_metrics(epoch,writer,metrics_train,None)
                add_scalars_hyperpara_GP(epoch,writer,hyperpara_train,None,args)

        save_regular_model_GP(model,likelihood,epoch, args.num_epochs, args.FREQ_SAVE,os.path.join(args.saved_name,args.saved_path))
        lr_sched.step()
    
    


def train(model,likelihood,train_loader,optimizer,mll,device,trained_Unet, loss_type, criterion,add_noise,loss_add_mu_reg,num_annotators):
    model.train()
    if trained_Unet == 'notrainU':
        model.feature_extractor.eval()
    likelihood.train()
    loss_epoch,dice_epoch,iou_epoch,hd_epoch,hd95_epoch,ece,nll_epoch,index_non_empty = [],[],[],[],[],[],[],[]
    noise_epoch, lengthscale_epoch, outputscale_epoch = [],[],[]
    noise_epoch0, noise_epoch1, noise_epoch2 = [],[],[]
    mu_epoch0, mu_epoch1, mu_epoch2 = [],[],[]
    noise_epochs = [[] for _ in range(num_annotators)]
    mu_epochs    = [[] for _ in range(num_annotators)]

    for ii, (image, mask_all, mask_wo_noise, selected_observation) in enumerate(train_loader):
        images, masks, mask_wo_noise = image.float().to(device), mask_all.float().to(device), mask_wo_noise.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        # get GP’s latent posterior predictions. These will be MultivariateNormal distributions
        output = model(images[:,None,...])
        # transform these outputs to classification probabilities using likelihood.
        # add noise to account for the noise in observations
        cls_probs = likelihood(output,add_noise,selected_observation,num_annotators,images.shape[0]).mean
        # print('train',likelihood.noise)

        if loss_type == 'likelihood':
            loss = -mll(output, masks.view(-1))
        elif loss_type == 'dice':
            soft_dice = compute_dice(cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
            loss = (1-soft_dice.mean())
        elif loss_type == 'BCE':
            loss = criterion(cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
        elif loss_type == 'dice_BCE':
            soft_dice = compute_dice(cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
            BCE = criterion(cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
            loss = BCE + (1-soft_dice.mean())
            
        elif loss_type == 'dice_BCE_kl':
            soft_dice = compute_dice(cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
            BCE = criterion(cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
            kl = model.gp_layer.variational_strategy.kl_divergence()
            loss = BCE + (1-soft_dice.mean())+kl/len(train_loader.dataset)
        elif loss_type == 'dice_BCE_likelihood':
            soft_dice = compute_dice(cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
            BCE = criterion(cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
            loss = BCE + (1-soft_dice.mean())-mll(output, masks.view(-1))
        
        elif loss_type == 'dice_BCE_voted':
            cls_probs_voetd = likelihood(output,'noadd',None).mean
            soft_dice_voetd = compute_dice(cls_probs_voetd.reshape(mask_wo_noise.shape[0],mask_wo_noise.shape[1],mask_wo_noise.shape[2]), mask_wo_noise)
            BCE_voetd = criterion(cls_probs_voetd.reshape(mask_wo_noise.shape[0],mask_wo_noise.shape[1],mask_wo_noise.shape[2]), mask_wo_noise)
            soft_dice = compute_dice(cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
            BCE = criterion(cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
            loss = BCE + (1-soft_dice.mean())+BCE_voetd + (1-soft_dice_voetd.mean())
            
        elif loss_type == 'dice_voted':
            cls_probs_voetd = likelihood(output,'noadd',None).mean
            soft_dice_voetd = compute_dice(cls_probs_voetd.reshape(mask_wo_noise.shape[0],mask_wo_noise.shape[1],mask_wo_noise.shape[2]), mask_wo_noise)
            soft_dice = compute_dice(cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
            loss = (1-soft_dice.mean())+(1-soft_dice_voetd.mean())

        else:
            raise ValueError('Loss type not recognized')
        
        if loss_add_mu_reg == 'add' and add_noise == 'add3sigmas3mus':
            loss = loss + torch.sum(likelihood.noise[3:]**2)
        elif loss_add_mu_reg == 'sum_2' and add_noise == 'add3sigmas3mus':
            loss = loss + torch.sum(likelihood.noise[num_annotators:])**2


        loss.backward()
        optimizer.step()


        with torch.no_grad():
            
            # calculate NLL
            nll = BinarySegmentationNLLMetric(cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
            # calculate ECE
            # for each image, calculate the ECE, and then obtain the mean ECE across all images
            cls_probs_reshape = cls_probs.reshape(masks.shape[0],masks.shape[1],masks.shape[2])
            for _i in range(masks.shape[0]):
                ece.append(binary_calibration_error(cls_probs_reshape[_i,...], masks[_i,...].int(), n_bins=15, norm='l1').item())

            
            # calculate Dice score
            hard_dice = compute_dice(cls_probs.ge(0.5).float().reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)
            # calculate IoU
            hard_iou = compute_iou(cls_probs.ge(0.5).float().reshape(masks.shape[0],masks.shape[1],masks.shape[2]), masks)

            # calculate dice on only non-empty masks
            flattened_masks = masks.view(masks.size(0), -1) 
            index_non_empty += torch.tensor([torch.unique(flattened_masks[i]).numel() > 1 for i in range(masks.size(0))], dtype=torch.float32, device=masks.device).tolist()

            loss_epoch.append(loss.item())
            dice_epoch += hard_dice.tolist()
            iou_epoch += hard_iou.tolist()
            nll_epoch += nll.tolist()

            outputscale_epoch.append(model.gp_layer.covar_module.outputscale.item())
            lengthscale_epoch.append(model.gp_layer.covar_module.base_kernel.lengthscale.item())
            if add_noise == 'add':
                noise_epoch.append(likelihood.noise_covar.noise.item())
            elif add_noise == 'add3sigmas':
                noise_epoch0.append(likelihood.noise[0].item())
                noise_epoch1.append(likelihood.noise[1].item())
                noise_epoch2.append(likelihood.noise[2].item())
            elif add_noise == 'add3sigmas3mus':          

                for i_anno in range(num_annotators):
                    noise_epochs[i_anno].append(likelihood.noise[i_anno].item())
                    mu_epochs[i_anno].append(likelihood.noise[num_annotators+i_anno].item())

                





    dice_epoch_non_empty = sum([dice_epoch[i] for i in range(len(dice_epoch)) if index_non_empty[i] == 1]) / sum(index_non_empty)
    iou_epoch_non_empty = sum([iou_epoch[i] for i in range(len(iou_epoch)) if index_non_empty[i] == 1]) / sum(index_non_empty)

    ece_epoch = sum(ece)/len(ece)
    loss_epoch = sum(loss_epoch)/len(loss_epoch)
    dice_epoch = sum(dice_epoch)/len(dice_epoch)
    iou_epoch = sum(iou_epoch)/len(iou_epoch)
    nll_epoch = sum(nll_epoch)/len(nll_epoch)

    outputscale_epoch = sum(outputscale_epoch)/len(outputscale_epoch)
    lengthscale_epoch = sum(lengthscale_epoch)/len(lengthscale_epoch)
    if add_noise == 'add':
        noise_epoch = sum(noise_epoch)/len(noise_epoch)
    elif add_noise == 'add3sigmas':
        noise_epoch0 = sum(noise_epoch0)/len(noise_epoch0)
        noise_epoch1 = sum(noise_epoch1)/len(noise_epoch1)
        noise_epoch2 = sum(noise_epoch2)/len(noise_epoch2)
        noise_epoch = [noise_epoch0, noise_epoch1, noise_epoch2]
    elif add_noise == 'add3sigmas3mus':

        noise_mean = [
            sum(epoch_vals) / len(epoch_vals)
            for epoch_vals in noise_epochs
        ]

        mu_mean = [
            sum(epoch_vals) / len(epoch_vals)
            for epoch_vals in mu_epochs
        ]

        noise_epoch = noise_mean + mu_mean



    elif add_noise == 'noadd':
        noise_epoch = None
    else:
        raise ValueError('Invalid noise type')

        
    metrics = {'loss': loss_epoch, 'dice': dice_epoch, 'iou': iou_epoch, 'nll': nll_epoch, 'ece': ece_epoch, 'dice_non_empty': dice_epoch_non_empty, 'iou_non_empty': iou_epoch_non_empty}
    hyperpara = {'outputscale': outputscale_epoch, 'lengthscale': lengthscale_epoch, 'noise': noise_epoch}
    
    return metrics, hyperpara

def val(model,likelihood,val_loader,mll,device,epoch,args, criterion):
    model.eval()
    likelihood.eval()
    loss_epoch,dice_epoch,iou_epoch,hd_epoch,hd95_epoch,ece,nll_epoch,index_non_empty = [],[],[],[],[],[],[],[]
    noise_epoch, lengthscale_epoch, outputscale_epoch = [],[],[]
    noise_epoch0, noise_epoch1, noise_epoch2 = [],[],[]
    mu_epoch0, mu_epoch1, mu_epoch2 = [],[],[]
    noise_epochs = [[] for _ in range(args.num_annotators)]
    mu_epochs    = [[] for _ in range(args.num_annotators)]

    with torch.no_grad():

        for ii, (image_val, mask_all_val, mask_wo_noise_val,selected_observation) in enumerate(val_loader):
            images_val, masks_val, mask_wo_noise_val = image_val.float().to(device), mask_all_val.float().to(device), mask_wo_noise_val.float().to(device)
            masks_val = mask_wo_noise_val[:,-1,:,:]

            preds_val = model(images_val[:,None,...])


            # get prediction with/without noise
            output = likelihood(preds_val,args.addnoise_pred)   # Get classification predictions
            binary_output = output.mean.ge(0.5).float() # Transform these probabilities to be 0/1 labels
            # print('val',likelihood.noise_covar.noise.item())
            cls_probs = output.mean
            cls_probs_reshape = cls_probs.reshape(masks_val.shape[0],masks_val.shape[1],masks_val.shape[2])


            if args.loss_type == 'likelihood':
                loss =  -mll(preds_val, masks_val.view(-1))
            elif args.loss_type == 'dice' or args.loss_type == 'dice_voted':
                soft_dice = compute_dice(cls_probs_reshape, masks_val.long())
                loss = (1-soft_dice.mean())
            elif args.loss_type == 'BCE':
                loss = criterion(cls_probs_reshape, masks_val)
            elif args.loss_type == 'dice_BCE' or args.loss_type == 'dice_BCE_voted':
                soft_dice = compute_dice(cls_probs_reshape, masks_val.long())
                BCE = criterion(cls_probs_reshape, masks_val)
                loss = BCE + (1-soft_dice.mean())

            elif args.loss_type == 'dice_BCE_kl':
                soft_dice = compute_dice(cls_probs_reshape, masks_val.long())
                BCE = criterion(cls_probs_reshape, masks_val)
                kl = model.gp_layer.variational_strategy.kl_divergence()
                loss = BCE + (1-soft_dice.mean())+kl/len(val_loader.dataset)
                
            elif args.loss_type == 'dice_BCE_likelihood':
                soft_dice = compute_dice(cls_probs_reshape, masks_val.long())
                BCE = criterion(cls_probs_reshape, masks_val)
                loss = BCE + (1-soft_dice.mean())-mll(preds_val, masks_val.view(-1))
            else:
                raise ValueError('Loss type not recognized')
        
            if args.loss_add_mu_reg == 'add' and args.addnoise == 'add3sigmas3mus':
                loss = loss + torch.sum(likelihood.noise[3:]**2)
            
            elif args.loss_add_mu_reg == 'sum_2' and args.addnoise == 'add3sigmas3mus':
                loss = loss + torch.sum(likelihood.noise[args.num_annotators:])**2


            nll = BinarySegmentationNLLMetric(cls_probs_reshape, masks_val)

            for _i in range(masks_val.shape[0]):
                ece.append(binary_calibration_error(cls_probs_reshape[_i,...], masks_val[_i,...].int(), n_bins=15, norm='l1').item())


            hard_dice = compute_dice(binary_output.reshape(masks_val.shape[0],masks_val.shape[1],masks_val.shape[2]), masks_val.long())
            
            if args.labels == 'random4' and args.labels_infer == 'seg_1_2_3_HQ':
                output_0 = likelihood(preds_val,args.addnoise,[0]*image_val.shape[0],args.num_annotators,images_val.shape[0])
                binary_output_0 = output_0.mean.ge(0.5).float() # Transform these probabilities to be 0/1 labels
                hard_dice_0 = compute_dice(binary_output_0.reshape(masks_val.shape[0],masks_val.shape[1],masks_val.shape[2]), mask_wo_noise_val[:,0,...].long())

                output_1 = likelihood(preds_val,args.addnoise,[1]*image_val.shape[0],args.num_annotators,images_val.shape[0])
                binary_output_1 = output_1.mean.ge(0.5).float() # Transform these probabilities to be 0/1 labels
                hard_dice_1 = compute_dice(binary_output_1.reshape(masks_val.shape[0],masks_val.shape[1],masks_val.shape[2]), mask_wo_noise_val[:,1,...].long())

                output_2 = likelihood(preds_val,args.addnoise,[2]*image_val.shape[0],args.num_annotators,images_val.shape[0])
                binary_output_2 = output_2.mean.ge(0.5).float() # Transform these probabilities to be 0/1 labels
                hard_dice_2 = compute_dice(binary_output_2.reshape(masks_val.shape[0],masks_val.shape[1],masks_val.shape[2]), mask_wo_noise_val[:,2,...].long())

                hard_dice = (hard_dice + hard_dice_0 + hard_dice_1 + hard_dice_2)/4


            # calculate IoU
            hard_iou = compute_iou(binary_output.reshape(masks_val.shape[0],masks_val.shape[1],masks_val.shape[2]), masks_val.long())
            
            # hd_distances_val, hd95_distances_val = compute_hausdorff_distances(binary_output.reshape(masks_val.shape[0],masks_val.shape[1],masks_val.shape[2]), masks_val.long())
                        
            # calculate dice on only non-empty masks
            flattened_masks = masks_val.view(masks_val.size(0), -1) 
            index_non_empty += torch.tensor([torch.unique(flattened_masks[i]).numel() > 1 for i in range(masks_val.size(0))], dtype=torch.float32, device=masks_val.device).tolist()


            loss_epoch.append(loss.item())
            dice_epoch += hard_dice.tolist()
            iou_epoch += hard_iou.tolist()
            nll_epoch += nll.tolist()

            outputscale_epoch.append(model.gp_layer.covar_module.outputscale.item())
            lengthscale_epoch.append(model.gp_layer.covar_module.base_kernel.lengthscale.item())
            if args.addnoise == 'add':
                noise_epoch.append(likelihood.noise_covar.noise.item())
            elif args.addnoise == 'add3sigmas':
                noise_epoch0.append(likelihood.noise[0].item())
                noise_epoch1.append(likelihood.noise[1].item())
                noise_epoch2.append(likelihood.noise[2].item())
            elif args.addnoise == 'add3sigmas3mus':
                       
                for i_anno in range(args.num_annotators):
                    noise_epochs[i_anno].append(likelihood.noise[i_anno].item())
                    mu_epochs[i_anno].append(likelihood.noise[args.num_annotators+i_anno].item())

            visualize(images_val, masks_val, binary_output.reshape(masks_val.shape[0],masks_val.shape[1],masks_val.shape[2]),epoch,ii,args)


    dice_epoch_non_empty = sum([dice_epoch[i] for i in range(len(dice_epoch)) if index_non_empty[i] == 1]) / sum(index_non_empty)
    iou_epoch_non_empty = sum([iou_epoch[i] for i in range(len(iou_epoch)) if index_non_empty[i] == 1]) / sum(index_non_empty)

    ece_epoch = sum(ece)/len(ece)
    loss_epoch = sum(loss_epoch)/len(loss_epoch)
    dice_epoch = sum(dice_epoch)/len(dice_epoch)
    iou_epoch = sum(iou_epoch)/len(iou_epoch)
    nll_epoch = sum(nll_epoch)/len(nll_epoch)

    outputscale_epoch = sum(outputscale_epoch)/len(outputscale_epoch)
    lengthscale_epoch = sum(lengthscale_epoch)/len(lengthscale_epoch)
    if args.addnoise == 'add':
        noise_epoch = sum(noise_epoch)/len(noise_epoch)
    elif args.addnoise == 'add3sigmas':
        noise_epoch0 = sum(noise_epoch0)/len(noise_epoch0)
        noise_epoch1 = sum(noise_epoch1)/len(noise_epoch1)
        noise_epoch2 = sum(noise_epoch2)/len(noise_epoch2)
        noise_epoch = [noise_epoch0, noise_epoch1, noise_epoch2]
    elif args.addnoise == 'add3sigmas3mus':
        
        noise_mean = [
            sum(epoch_vals) / len(epoch_vals)
            for epoch_vals in noise_epochs
        ]

        mu_mean = [
            sum(epoch_vals) / len(epoch_vals)
            for epoch_vals in mu_epochs
        ]

        noise_epoch = noise_mean + mu_mean

    elif args.addnoise == 'noadd':
        noise_epoch = None
    else:
        raise ValueError('Invalid noise type')


    metrics = {'loss': loss_epoch, 'dice': dice_epoch, 'iou': iou_epoch, 'nll': nll_epoch, 'ece': ece_epoch, 'dice_non_empty': dice_epoch_non_empty, 'iou_non_empty': iou_epoch_non_empty}
    hyperpara = {'outputscale': outputscale_epoch, 'lengthscale': lengthscale_epoch, 'noise': noise_epoch}

    return metrics,hyperpara

            
