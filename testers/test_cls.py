#%%
# Imports
import sys
sys.path.append('../dataloaders')
sys.path.append('../architectures')
#import dataloader_table
import dataloader
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm 
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
import monai_swin, sfcn_mod
import os
import monai
import torch.nn as nn
import torch.nn.functional as F

#%%
#Parameters
cohort = 'ppmi'
img_size = 180
column_name = 'sex'
ttype = 'test'
subg = '60'
task_name = f'{column_name}'
task = 'classification'
method_name = 'supervised'
model_name = 'sfcn'
n_folds = 1
num_epochs = 1000
nrows = None
nrows_test = None
batch_size = 4
n_classes = 2
batch_size_test = 4
lr = 1e-05
n_splits = 3

# Set Paths
tensor_dir = f'../../images/{cohort}/npy_{cohort}{img_size}'
csv_test = f'../data/{cohort}/{ttype}/demographics.csv'
unique_name = f'{task_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_s{n_splits}_im{img_size}'
output_path = f'../scores/{cohort}/{model_name}/{ttype}/{unique_name}'
model_path = f'../models/{model_name}/{column_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_s{n_splits}_im{img_size}'
log_path = f'../logs/aurocs/{cohort}/{model_name}/{ttype}'

#Check table
data = pd.read_csv(csv_test, dtype={'eid': str}, delimiter = ',')
print(data)

#Swin Parameters
patch_size = [8, 8, 8]
window_size = [16, 16, 16]
num_heads = [3,6,12,24]
depths = [2,2,2,2]
feature_size = 96
dev = "cuda:0"
torch.cuda.set_device(dev)
seed = 42
 
#%%
auroc_list = [] 
for fold in range(1, n_folds + 1):
    print(f'External validation for fold {fold}')        
    
    model = sfcn_mod.SFCN(output_dim=n_classes, task=task).to(dev)
    #model = monai_swin.SwinTransformer(in_chans = 1, embed_dim = feature_size, window_size = window_size, patch_size = patch_size, depths = depths, num_heads = num_heads).to(dev)
    #model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=n_classes).to(dev)
    print(model)
    
    pretrained_model = f'{model_path}_k{fold}_best.pth'
    checkpoint = torch.load(pretrained_model, map_location=dev)
    
    model.load_state_dict(checkpoint['state_dict'])  
    model.to(dev)
    
    # Define dataset and dataloader
    test_dataset = dataloader.BrainDataset(csv_file=csv_test, root_dir=tensor_dir, column_name=column_name, num_rows=nrows_test, num_classes=n_classes)  
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, num_workers=8, drop_last=True)
    
    test_outputs_binary = []
    test_labels = []
    test_eids = []  
    test_genders = [] 
    
    model.eval()
    with torch.no_grad():
        for j, (eid, images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):  # Assuming gender is part of the returned data
            test_eids.extend(eid)
            images = images.to(dev)
            labels = labels.float().to(dev)
            binary_labels = labels[:, 1]
            test_labels.extend(binary_labels.tolist())
            
            outputs = model(images).to(dev)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            binary_outputs = probs[:, 1]
            
            test_outputs_binary.extend(binary_outputs.tolist())
    
    # Calculate AUROC
    auroc = roc_auc_score(test_labels, test_outputs_binary)
    print(f'AUROC for fold {fold}: {auroc:.5f}')
    
    # Append AUROC to the list
    auroc_list.append({'fold': fold, 'AUROC': auroc})
    
    # Save predictions to CSV
    output_csv = f'{output_path}_k{fold}.csv'
    df = pd.DataFrame(data={"eid": test_eids, "label": test_labels, "prediction": test_outputs_binary}) 
    df.to_csv(output_csv, index=False)
    print(f'Predictions saved to {output_csv}')
# %%
