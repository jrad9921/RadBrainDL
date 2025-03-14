#%%
# Imports
import sys
sys.path.append('../dataloaders')
sys.path.append('../architectures')
import dataloader_table, dataloader_new
import dataloader
import sfcn_new, linear, sfcn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm 
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
import monai_swin
import os
import monai
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import torch.nn as nn
import torch.nn.functional as F

#%%
#Parameters
cohort = 'ukb'
img_size = 180
column_name = 'age'
task = 'regression'
subg = 'male'
task_name = f'{column_name}'
method_name = 'supervised'
n_classes = 1
model_name = 'dense'
n_folds = 1
num_epochs = 1000
nrows = None
nrows_test = None
batch_size = 4
b_size = 4
lr = 1e-05
n_splits = 3

# Set Paths
tensor_dir = f'../../images/{cohort}/npy_{cohort}{img_size}'
unique_name = f'{task_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_s{n_splits}_im{img_size}'
csv_test = f'../data/{cohort}/test/demographics.csv'
output_path = f'../scores/{cohort}/{model_name}/test/{unique_name}'
model_path = f'../models/{model_name}/{column_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_s{n_splits}_im{img_size}'
log_path = f'../logs/aurocs/{cohort}/{model_name}/test'

#Swin Parameters
patch_size = [8, 8, 8]
window_size = [16, 16, 16]
num_heads = [3,6,12,24]
depths = [2,2,2,2]
feature_size = 96

#Set Device
dev = "cuda:1"
torch.cuda.set_device(dev)
seed = 42

# %%  
auroc_list = [] 
for fold in range(1, n_folds + 1):
    print(f'External validation for fold {fold}')
    #model = SFCN(output_dim=n_classes).to(dev)        
    #model = sfcn_reg.SFCN().to(dev)
    #model = monai_swin.SwinTransformer(in_chans = 1, embed_dim = feature_size, window_size = window_size, patch_size = patch_size, depths = depths, num_heads = num_heads, n_classes = n_classes).to(dev)
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=n_classes).to(dev)

    pretrained_model = f'{model_path}_k{fold}_best.pth'
    checkpoint = torch.load(pretrained_model, map_location=dev)
    model.load_state_dict(checkpoint['state_dict'])  
    model.to(dev)
    
    # Define dataset and dataloader
    test_dataset = dataloader_new.BrainDataset(csv_file=csv_test, root_dir=tensor_dir, column_name=column_name, num_rows=nrows_test, num_classes=n_classes, task = task)  
    test_loader = DataLoader(test_dataset, batch_size=b_size, num_workers=8, drop_last=True)
    
    test_outputs = []
    test_labels = []
    test_eids = []  
    
    model.eval()
    with torch.no_grad():
        for j, (eid, images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            test_eids.extend(eid)
            images = images.to(dev)
            labels = labels.float().to(dev)
            test_labels.extend(labels.tolist())            
            outputs = model(images).to(dev)
            test_outputs.extend(outputs.tolist())
    
    # Calculate MAE
    mae = mean_absolute_error(test_labels, test_outputs)
    print(f'MAE for fold {fold}: {mae:.4f}')
    
    # Calculate Pearson's r
    #pearson_corr, _ = pearsonr(test_labels, test_outputs)
    #print(f"Pearson's r for fold {fold}: {pearson_corr:.4f}")
    
    # Calculate AUROC if desired
    # auroc = roc_auc_score(test_labels, test_outputs)
    # print(f'AUROC for fold {fold}: {auroc:.4f}')
    
    # Save predictions to CSV
    output_csv = f'{output_path}_k{fold}.csv'
    df = pd.DataFrame(data={"eid": test_eids, "label": test_labels, "prediction": test_outputs}) 
    df.to_csv(output_csv, index=False)
    print(f'Predictions saved to {output_csv}')

