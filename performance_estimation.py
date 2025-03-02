
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
    args.batch_size_val = 4
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
test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(args.batch_size_val), shuffle=False, num_workers=8)

# grid sample sigma and mu
# sigma_all = torch.linspace(0.1, 5, 99) #19
# mu_all = torch.linspace(-5, 5, 41) #21
# sigma_all = torch.linspace(0.1, 1.2, 11) #19
# mu_all = torch.linspace(-1, 1, 11)

sigma_all = torch.linspace(0.1, 3, 256)
mu_all = torch.linspace(-3, 3, 256)

# sigma_all = torch.tensor([0.3,0.25,0.1,0.15,0.4,0.45,0.2,0.25,0.5,0.55,0.6,0.35,0.5222,0.4790,0.3103,0.2295])
# mu_all = torch.tensor([-3,-2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2,2.5,3,2.4487,-0.5704,1.7028,-1.1801])


all_metrics = {}
# for sigma in sigma_all:
#     for mu in mu_all:
for _index in range(len(sigma_all)):
    sigma = sigma_all[_index]
    mu = mu_all[_index]

    if True:
        metrics,hyperpara,metrics_addnoise,saved_model_name,saved_likelihood_name = test_DKL_grid_sample(test_set,test_loader,args,device,models_folders+'/'+folder,sigma,mu,saved_path)
        all_metrics[(sigma.item(),mu.item())] = metrics_addnoise
        # save metrics and hyperparameters
        save_metrics_grid_sample(os.getcwd()+'/'+ fd_name_save +'/'+csv_name,folder,metrics,hyperpara,metrics_addnoise,saved_model_name,saved_likelihood_name,args,sigma,mu)        


        with open(saved_path +'/'+ 'sigma' + str('%.4f'% sigma.item()) + '_mu' + str('%.4f'%mu.item()) + '_' + folder +'_hyperparameters.pkl', 'wb') as f:
            pickle.dump(hyperpara, f)
        with open(saved_path +'/' + 'sigma' + str('%.4f'%sigma.item()) + '_mu' + str('%.4f'%mu.item()) + '_' + folder +'_metrics_addnoise.pkl', 'wb') as f:
            pickle.dump(metrics_addnoise, f)
        with open(saved_path +'/' + 'sigma' + str('%.4f'%sigma.item()) + '_mu' + str('%.4f'%mu.item()) + '_' + folder +'_metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)


# # plot the performance of sigma and mu

with open(saved_path +'/' + 'all_metrics.pkl', 'wb') as f:
    pickle.dump(all_metrics, f)

metrics_plot_2(sigma_all,mu_all,all_metrics,'dice',saved_path)
metrics_plot_2(sigma_all,mu_all,all_metrics,'dice_non_empty',saved_path)
metrics_plot_2(sigma_all,mu_all,all_metrics,'iou',saved_path)
metrics_plot_2(sigma_all,mu_all,all_metrics,'iou_non_empty',saved_path)
metrics_plot_2(sigma_all,mu_all,all_metrics,'hd',saved_path)
metrics_plot_2(sigma_all,mu_all,all_metrics,'hd95',saved_path)
metrics_plot_2(sigma_all,mu_all,all_metrics,'ece_addnoise',saved_path)
metrics_plot_2(sigma_all,mu_all,all_metrics,'nll',saved_path)









# load saved metrics
# with open('saved_dictionary.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)
    

    



    

  