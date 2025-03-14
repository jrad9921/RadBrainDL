
#%%
# 0.Imports 
import pandas as pd
import matplotlib.pyplot as plt
import os
import monai
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import dateutil
dateutil.__version__
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import interp
from collections import Counter
import datetime
import time 
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, f1_score, brier_score_loss
from sklearn.model_selection import KFold, StratifiedKFold
import random
import sys
sys.path.append('../dataloaders')
sys.path.append('../architectures')
import dataloader, dataloader_new
import sfcn, linear, monai_swin, monai_vit

#%%
# 5.Parameters
# Basic parameters
cohort = 'ukb'
model_name = 'dense'
method_name = 'supervised'
column_name = 'age'
task = 'regression'
csv_train = f'../data/ukb/train/demographics.csv'
img_size = 180
tensor_dir = f'../../images/{cohort}/npy_{cohort}{img_size}'

#Training parameters
batch_size = 4
num_epochs = 1000
n_splits = 3
nrows = None
dev = "cuda:1"
n_classes = 1
n_channels = 1
lr = 1e-03
seed = 42 
best_val_loss = 10000

#transforms = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# Calculate the ratio
df = pd.read_csv(csv_train)
print(df)
ratio = (df[column_name] == 1).sum() / (df[column_name] == 0).sum()
print("Ratio of positive to negative cases:", ratio)
# logging parameters
unique_name = f"{column_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_s{n_splits}_im{img_size}"
scores_train = f'../scores/{cohort}/{model_name}/train/{unique_name}'
scores_val = f'../scores/{cohort}/{model_name}/val/{unique_name}'
timelog_dir = f'../logs/timelog/{model_name}/'
trainlog_dir = f'../logs/trainlog/{model_name}/'
vallog_dir = f'../logs/vallog/{model_name}/'
log_dir = f'../logs/aurocs/{model_name}/'
save_model = f'../models/{model_name}/'
fig_name = f"{unique_name}.png"

# ssl parameters 
ssl_batch_size = 8
ssl_n_epochs = 20
pretrained_model = f'../logs/ssl/{model_name}/best_model_b{ssl_batch_size}_e{ssl_n_epochs}.pt'
vitae_pretrained_model = f'../logs/best_model.pt'

# swin parameters
patch_size = [8, 8, 8]
window_size = [16, 16, 16]
num_heads = [3,6,12,24]
depths = [2,2,2,2]
feature_size = 96

# early stopping parameters
patience = 10

#Set Device
if torch.cuda.is_available():
    torch.cuda.set_device(dev)

#Set a random seed for PyTorch (for GPU and CPU operations)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

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
        avg_shape = [3, 3, 3]
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
        x = x.mean(dim = [2,3,4])
        x = x.squeeze()
        return x
#%%
#Training dataset
train_dataset = dataloader_new.BrainDataset(csv_train, tensor_dir, column_name, task='regression', num_rows = nrows)

#%%
# Training loop
trainlog_file = os.path.join(trainlog_dir, f"{unique_name}.txt")
total_time = 0
with open(trainlog_file, "a") as log:
    log.write(f'Fold, Epoch, Training Loss, Validation Loss\n')

# Initialize KFold
skf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)

for fold, (train_ids, val_ids) in enumerate(skf.split(np.arange(len(train_dataset)))):
    
    print(f"  Training samples: {train_ids}")
    print(f"  Validation samples: {val_ids}")
    train_losses = []
    val_losses = []
    
    early_stop_counter = 0
    start_time = time.time()
    print(f"Starting Fold {fold + 1}")
    
    # Retrieve patient id lists for the fold
    train_eids = [train_dataset.annotations.eid[i] for i in train_ids]
    val_eids = [train_dataset.annotations.eid[i] for i in val_ids]

    # Retrieve labels lists for the fold 
    train_labels = [train_dataset.annotations[column_name][i] for i in train_ids]
    val_labels = [train_dataset.annotations[column_name][i] for i in val_ids]

    # Check fold distribution
    train_label_distribution = Counter(train_labels)
    val_label_distribution = Counter(val_labels)

    print(f"Training set label distribution for Fold {fold + 1}: {train_label_distribution}")
    print(f"Validation set label distribution for Fold {fold + 1}: {val_label_distribution}")

    train_subset = torch.utils.data.Subset(train_dataset, train_ids)
    val_subset = torch.utils.data.Subset(train_dataset, val_ids)
    
    # Set dataloaders
    train_loader = DataLoader(train_subset, batch_size = batch_size, num_workers=8, drop_last = True)
    val_loader = DataLoader(val_subset, batch_size = batch_size, num_workers=8, drop_last = True)
    
    # Set Model
    #model = Classifier(base, feature_size = feature_size, last_layer = last_layer, num_classes = n_classes).to(dev)
    #model = SFCN(output_dim=n_classes).to(dev)#.load_state_dict(checkpoint['state_dict']).to(dev)
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels= n_channels, out_channels = n_classes).to(dev)
    #model = monai_vit.ViT(spatial_dims=3, in_channels = 1, img_size=img_size, proj_type = 'conv', patch_size = patch_size, hidden_size = feature_size, num_heads = 4, classification = True, num_classes = 2).to(dev)
    #model = monai_swin.SwinTransformer(in_chans = 1, embed_dim = feature_size, window_size = window_size, patch_size = patch_size, depths = depths, num_heads = num_heads, n_classes = n_classes).to(dev)#
    #model = linear.LinearNN(input_size = 64, output_size = 2).to(dev)
    
    # Set Optimizer and Loss
     # Adjusted initialization without class weights
    criterion = torch.nn.MSELoss().to(dev)  # Example: Using Mean Squared Error for regression
    #Define the optimizer with initial learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(model)
    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    for epoch in range(num_epochs):

        train_outputs = []
        train_outputs_binary = []
        train_labels = []
        val_outputs = []
        val_outputs_binary = []
        val_labels = []
        train_table = []
        val_table = []
        train_eids = []
        val_eids = []
        # Training loop
        model.train()
        running_train_loss = 0.0
        for i, (eid, images, labels) in tqdm(enumerate(train_loader), total = len(train_loader)):
            images = images.to(dev)
            eid = eid
            train_eids.extend(eid)
            labels = labels.float().to(dev)        
            optimizer.zero_grad()
            outputs = model(images).to(dev)
            train_outputs.extend(outputs.tolist())
            train_labels.extend(labels.tolist())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() 
        train_loss = running_train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation loop
        model.eval() 
        running_val_loss = 0.0
        with torch.no_grad():
            for j, (eid, images, labels) in tqdm(enumerate(val_loader), total = len(val_loader)):
                images = images.to(dev)
                eid = eid
                val_eids.extend(eid)
                labels = labels.float().to(dev)
                outputs = model(images).to(dev)
                #print(outputs.shape)
                #print(labels.shape)
                val_outputs.extend(outputs.tolist())
                val_labels.extend(labels.tolist())
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() 
        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')  
        
        if val_loss < best_val_loss:
            print(f"Saving new model based on validation loss {val_loss:.4f}")
            best_val_loss = val_loss
            checkpoint = {"epoch": num_epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            # Save Model
            torch.save(checkpoint, os.path.join(save_model, f"{unique_name}_k{fold+1}_best.pth")) 
            print(f'Model saved at {save_model}')

            best_val_labels = val_labels
            best_val_outputs = val_outputs
            best_val_outputs_binary = val_outputs_binary
            early_stop_counter = 0
        else:
            early_stop_counter += 1 

        if early_stop_counter >= patience:
            print(f'Early stopping after {epoch + 1} epochs without improvement in validation loss for {patience} epochs')
            break
        
        # Update the learning rate
        scheduler.step()
        # Optionally, print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr}')
        
        trainlog_file = os.path.join(trainlog_dir, f"{unique_name}.txt")
        with open(trainlog_file, "a") as log:
            log.write(f'{fold + 1}, {epoch + 1}, {train_loss:.4f}, {val_loss:.4f} \n') 
           
    
    # Save prediction scores into dictionaries
    train_data = {
        #'fold': fold + 1,
        'eid': train_eids,
        #'sex': train_gender,
        #'ageclass': train_ageclass,
        'label': train_labels,
        'logits': train_outputs, 
        #'prediction': train_outputs_binary,
        }
        
    val_data = {
        #'fold': [fold + 1],
        'eid': val_eids,
        #'sex': val_gender,
        #'ageclass': val_ageclass,
        'label': best_val_labels,
        'logits': best_val_outputs, 
        #'prediction': best_val_outputs_binary,
        }
    
    # Log Predictions into csvs
    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)
    df_train.to_csv(f'{scores_train}_k{fold+1}.csv', index=False)
    df_val.to_csv(f'{scores_val}_k{fold+1}.csv', index=False)
    print(f'Predictions saved!')

    #Log Loss
    vallog_file = os.path.join(vallog_dir, f"{unique_name}.txt")
    with open(vallog_file, "a") as log:
        log.write(f'Fold {fold + 1} completed. Best Validation Loss: {best_val_loss:.4f} \n')
        log.write(f'Early stopping after {epoch + 1} epochs without improvement in validation loss for {patience} epochs \n')

    # Log Time
    end_time = time.time()
    duration = end_time - start_time
    n_samples = len(train_dataset)
    total_time += duration
    norm_time = duration/n_samples

    timelog_file = os.path.join(timelog_dir, f"{unique_name}.txt")
    with open(timelog_file, "a") as log:
        log.write(f"Fold {fold + 1} - Duration: {duration} seconds -  Start Time: {datetime.datetime.fromtimestamp(start_time)} - End Time: {datetime.datetime.fromtimestamp(end_time)} - model params: {sum(p.numel() for p in model.parameters())} \n")    


    print(f"-------------------------------------Fold {fold +1} Saved------------------------------------------")

    break

fold_time = total_time / n_splits
print(f"Fold training time: {fold_time} seconds")
# %%
