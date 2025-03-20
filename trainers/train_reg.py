
#%%
#Imports 
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
import sfcn_mod, monai_swin

#%%
# Parameters

# Basic parameters
cohort = 'ukb'
model_name = 'dense'
method_name = 'supervised'
column_name = 'age'
task = 'regression'
img_size = 180

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

# Set Paths
tensor_dir = f'../../images/{cohort}/npy_{cohort}{img_size}'
csv_train = f'../data/ukb/train/demographics.csv'
unique_name = f"{column_name}_e{num_epochs}_n{nrows}_b{batch_size}_lr{lr}_s{n_splits}_im{img_size}"
scores_train = f'../scores/{cohort}/{model_name}/train/{unique_name}'
scores_val = f'../scores/{cohort}/{model_name}/val/{unique_name}'
timelog_dir = f'../logs/timelog/{model_name}/'
trainlog_dir = f'../logs/trainlog/{model_name}/'
vallog_dir = f'../logs/vallog/{model_name}/'
log_dir = f'../logs/aurocs/{model_name}/'
save_model = f'../models/{model_name}/'
fig_name = f"{unique_name}.png"

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
#Training dataset
train_dataset = dataloader.BrainDataset(csv_train, tensor_dir, column_name, task='regression', num_rows = nrows)

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
    #model = sfcn_mod.SFCN(input_size=img_size, output_dim=n_classes, task=task).to(dev)
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels= n_channels, out_channels = n_classes).to(dev)
    #model = monai_swin.SwinTransformer(in_chans = 1, embed_dim = feature_size, window_size = window_size, patch_size = patch_size, depths = depths, num_heads = num_heads, n_classes = n_classes).to(dev)#
    
    # Set Optimizer and Loss
    criterion = torch.nn.MSELoss().to(dev) 
    #Define the optimizer with initial learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(model)
    
    # Define the learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

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
        #scheduler.step()
        # Optionally, print the current learning rate
        #current_lr = optimizer.param_groups[0]['lr']
        #print(f'Current Learning Rate: {current_lr}')
        
        trainlog_file = os.path.join(trainlog_dir, f"{unique_name}.txt")
        with open(trainlog_file, "a") as log:
            log.write(f'{fold + 1}, {epoch + 1}, {train_loss:.4f}, {val_loss:.4f} \n') 
           
    
    # Save prediction scores into dictionaries
    train_data = {
        #'fold': fold + 1,
        'eid': train_eids,
        'label': train_labels,
        'logits': train_outputs, 
        }
        
    val_data = {
        #'fold': [fold + 1],
        'eid': val_eids,
        'label': best_val_labels,
        'logits': best_val_outputs, 
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
