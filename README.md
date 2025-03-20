# 3D Deep Learning for Neuroimaging

This repository contains code for training and testing 3D deep learning models for extracting neuroimaging representations, 
and can be used for several classification and regression tasks, for example age prediction and sex classification.
The images for each cohort should be in standard dimensions of (180, 180, 180) voxels after registration to MNI space, and brain extraction. 
The data for each cohort should be in standard format with eid and column_name tables. 

## Project Structure

```bash
├── main/                     # This is where the training and testing happens using images and labels
│   ├── trainers/
│   │   ├── train_cls.py      # Training script for sex classification
│   │   └── train_reg.py      # Training script for age prediction
│   ├── testers/
│   │   ├── test_cls.py       # Testing script for sex classification
│   │   └── test_reg.py       # Testing script for age prediction
│   ├── dataloaders/
│   │   └── dataloader.py     # Dataloader for both tasks
│   ├── architectures/
│   │   └── sfcn_mod.py       # Add architectures to train
│   └── README.md  
├── images/                  # This is where the 3D images (as npy files) for each cohort are stored 
└── data/                    # This is where the data tables (as csv files) for each cohort are stored 
```


## Architectures
3D architectures used:
1. SFCN from Peng et al, modified to 
2. Monai's 3D implementation of Densenet121
3. Monai's 3D implementation of SwinTransformer


## Directories
Create directories of models, logs and scores inside main. 
Create directories of images and data outside the main.
