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
tensor_dir = f'../../images/{cohort}/npy_{cohort}{img_size}'
column_name = 'age'
task = 'regression'
subg = 'male'
task_name = f'{column_name}'
method_name = 'supervised'
model_name = 'dense'
n_folds = 1
num_epochs = 1000
nrows = None
nrows_test = None
batch_size = 4
b_size = 4
lr = 1e-05
n_splits = 3
unique_name = f'{task_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_s{n_splits}_im{img_size}'
csv_test = f'../data/{cohort}/test/demographics.csv'
output_path = f'../scores/{cohort}/{model_name}/test/{unique_name}'
model_path = f'../models/{model_name}/{column_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_s{n_splits}_im{img_size}'
log_path = f'../logs/aurocs/{cohort}/{model_name}/test'
n_classes = 1
patch_size = [8, 8, 8]
window_size = [16, 16, 16]
num_heads = [3,6,12,24]
depths = [2,2,2,2]
feature_size = 96
dev = "cuda:1"
torch.cuda.set_device(dev)
seed = 42
#%%
class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=1, dropout=True):
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
        #x = F.log_softmax(x, dim=1)
        #print(x.shape)
        x = x.mean(dim = [2,3,4])
        x = x.squeeze()
        return x
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

