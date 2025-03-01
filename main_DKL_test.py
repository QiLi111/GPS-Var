
import os
import torch
import csv
from utils.utils import *
from utils.data_loader import DataSet
from test_DKL import test_DKL
import pickle

models_folders = 'path/to/saved/models'
high_quality_labels = True
all_test_folders_path = os.path.join(os.getcwd(),models_folders)

all_test_folders = [f for f in os.listdir(all_test_folders_path) if os.path.isdir(os.path.join(all_test_folders_path, f)) and not f.startswith('Unet')]
all_test_folders = sorted(all_test_folders)
csv_name = 'metrics.csv'
csv_name_error = 'error.csv'
fd_name_save = 'results'+'/'+models_folders

# csv file
if not os.path.exists(os.getcwd()+'/'+fd_name_save):
    os.makedirs(os.getcwd()+'/'+fd_name_save)

with open(os.getcwd()+'/'+ fd_name_save +'/'+csv_name_error, 'a', encoding='UTF8') as f_error:
    writer_error = csv.writer(f_error)
    writer_error.writerow(['error_folder:\n'])

with open(os.getcwd()+'/'+ fd_name_save +'/'+csv_name, 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name','outputscale & '\
                     'lengthscale & '\
                     'variance & '\
                     'model_name &'\
                    'likelihood_name &'\
                    'nonoise metrics: '\
                    'Dice & ''Non-empty-dice & '\
                    'IoU & ''Non-empty-IoU & '\
                    'hd & '\
                    'hd95 & '
                    'ECE & '\
                    'NLL & '\
                    'noise metrics: '\
                    'noise & '\
                    '...'
                    ])


for folder in all_test_folders:
    # get args
    try:
        if 'args.txt' in os.listdir(os.path.join(all_test_folders_path,folder)): 
            args = get_args(os.path.join(all_test_folders_path,folder,'args.txt'),folder)
            args.batch_size_val = 2
            args.saved_name = models_folders+'/'+args.saved_name
            args.pre_labels = args.labels
            args.labels = 'inference'
            if high_quality_labels:
                args.labels_infer = 'high_quality'
            
        else:
            # get args from folder name
            args = get_args_from_folder(folder)
            args.batch_size_val = 2
            args.pre_labels = args.labels
            args.labels = 'inference'
            # args.saved_name = models_folders+'/'+args.saved_name
    except:
        with open(os.getcwd()+'/'+ fd_name_save +'/'+csv_name_error, 'a', encoding='UTF8') as f_error:
            f_error.write(f"Error in {folder}")
            f_error.write('\n')
        continue

    # try:
    if True:

        print(f"Seeding with seed: {args.seed}")
        seed_all(int(args.seed))
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cuda_avail, device = torch_init(args.device)
        print("pytorch using device", device)

        # load data    
        dataset = DataSet(datapath=args.dataset,num_class=args.num_class,labels=args.labels,labels_inference = args.labels_infer,data_aug=args.data_aug)
        # dataset.split_data()
        _,_,test_set = dataset.load_data_split(os.path.join(args.dataset,'data_split','train_list.json'),os.path.join(args.dataset,'data_split','val_list.json'),os.path.join(args.dataset,'data_split','test_list.json'))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(args.batch_size_val), shuffle=False, num_workers=8)
        
        metrics_all,hyperpara_all,metrics_addnoise_all,saved_model_name,saved_likelihood_name = test_DKL(test_set,test_loader,args,device,models_folders+'/'+folder)
        # save metrics and hyperparameters
        save_metrics(os.getcwd()+'/'+ fd_name_save +'/'+csv_name,folder,metrics_all,hyperpara_all,metrics_addnoise_all,saved_model_name,saved_likelihood_name,args)        
        
        with open(os.getcwd()+'/'+ fd_name_save +'/'+ folder +'_metrics.pkl', 'wb') as f:
            pickle.dump(metrics_all, f) 
        with open(os.getcwd()+'/'+ fd_name_save +'/'+ folder +'_hyperparameters.pkl', 'wb') as f:
            pickle.dump(hyperpara_all, f)
        with open(os.getcwd()+'/'+ fd_name_save +'/'+ folder +'_metrics_addnoise.pkl', 'wb') as f:
            pickle.dump(metrics_addnoise_all, f)
    # except:
    #     with open(os.getcwd()+'/'+ fd_name_save +'/'+csv_name_error, 'a', encoding='UTF8') as f_error:
    #         f_error.write(f"Error in {folder}")
    #         f_error.write('\n')
    #     continue
    
    # load saved metrics
    # with open('saved_dictionary.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)
        

    



    

  