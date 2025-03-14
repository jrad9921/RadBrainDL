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
import monai_swin
import os
import monai
import torch.nn as nn
import torch.nn.functional as F
#%%
#Parameters
cohort = 'ppmi'
img_size = 180
tensor_dir = f'../../images/{cohort}/npy_{cohort}{img_size}'
#tensor_dir = f'../images/{cohort}/npy{img_size}'
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
batch_size_test = 4
lr = 1e-05
n_splits = 3
csv_test = f'../data/{cohort}/{ttype}/demographics.csv'
unique_name = f'{task_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_s{n_splits}_im{img_size}'
output_path = f'../scores/{cohort}/{model_name}/{ttype}/{unique_name}'
model_path = f'../models/{model_name}/{column_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_s{n_splits}_im{img_size}'
log_path = f'../logs/aurocs/{cohort}/{model_name}/{ttype}'
data = pd.read_csv(csv_test, dtype={'eid': str}, delimiter = ',')
print(data)
# Check the 'labels' column for positive and negative values
#Assuming 1.0 is positive and 0.0 is negative
positive_count = (data[column_name] == 1.0).sum()
negative_count = (data[column_name] == 0.0).sum()

# Calculate the positive-to-negative ratio
if negative_count > 0:
    pos_neg_ratio = positive_count / negative_count
else:
    pos_neg_ratio = float('inf')  # Avoid division by zero

# Print the results
print(f"Positive count: {positive_count}")
print(f"Negative count: {negative_count}")
print(f"Positive-to-negative ratio: {pos_neg_ratio:.2f}")

n_classes = 2
patch_size = [8, 8, 8]
window_size = [16, 16, 16]
num_heads = [3,6,12,24]
depths = [2,2,2,2]
feature_size = 96
dev = "cuda:0"
torch.cuda.set_device(dev)
seed = 42
# %%  
class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=2, dropout=True):
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        avg_shape = [5, 5, 5]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        #out = list()
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = F.log_softmax(x, dim=1)
        #print(x.shape)
        x = x.mean(dim = [2,3,4])
        #x = x.squeeze()
        return x
#%%
auroc_list = [] 
for fold in range(1, n_folds + 1):
    print(f'External validation for fold {fold}')        
    model = SFCN().to(dev)
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
