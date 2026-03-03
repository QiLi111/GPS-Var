
import os
import torch
import csv
from utils.utils import *
from utils.data_loader import DataSet
from utils.test_DKL_grid_sample import test_DKL_grid_sample
import pickle

high_quality_labels = True
folder = 'model/name/for/performance/query'
models_folders = 'grid_sample_performance/'
folder_path = os.path.join(os.getcwd(),models_folders,folder)

csv_name = 'metrics.csv'
fd_name_save = 'grid_sample_performance/results'
saved_path = os.getcwd()+'/'+ fd_name_save +'/' +folder

os.makedirs(os.path.join(saved_path,'plots'),exist_ok=True)
os.makedirs(os.getcwd()+'/'+fd_name_save,exist_ok=True)
os.makedirs(os.getcwd()+'/'+fd_name_save+'/'+folder,exist_ok=True)


# csv file
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


# get args
if 'args.txt' in os.listdir(folder_path): 
    args = get_args(os.path.join(folder_path,'args.txt'),folder)
    args.batch_size_val = 1
    args.saved_name = models_folders+'/'+args.saved_name
    args.pre_labels = args.labels
    args.labels = 'inference'
    if high_quality_labels:
        args.labels_infer = 'high_quality'

else:
    raise ValueError('args.txt not found in folder')
        
    

print(f"Seeding with seed: {args.seed}")
seed_all(int(args.seed))
args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cuda_avail, device = torch_init(args.device)
print("pytorch using device", device)

# load data    
dataset = DataSet(datapath=args.dataset,num_class=args.num_class,labels=args.labels,labels_inference = args.labels_infer,data_aug=args.data_aug)
# dataset.split_data()
_,_,test_set = dataset.load_data_split(os.path.join(args.dataset,'data_split','train_list.json'),os.path.join(args.dataset,'data_split','val_list.json'),os.path.join(args.dataset,'data_split','test_list.json'))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(args.batch_size_val), shuffle=False, num_workers=0)


all_metrics, sigma_all, mu_all = test_DKL_grid_sample(test_set,test_loader,args,device,models_folders+'/'+folder,saved_path)     



    

    



    

  