import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class BrainDataset(Dataset):
    def __init__(self, csv_file, root_dir, column_name, task, num_classes=None, deterministic=True, num_rows=None, transform=None):
        # Load the CSV file
        full_annotations = pd.read_csv(csv_file, dtype={'eid': str})

        # If num_rows is specified and valid, take only that many rows
        if num_rows is not None and 0 < num_rows <= len(full_annotations):
            self.annotations = full_annotations.head(num_rows)
        else:
            # Otherwise, take all rows
            self.annotations = full_annotations
        
        print(f"Loaded {len(self.annotations)} rows.")
        
        self.root_dir = root_dir
        self.deterministic = deterministic
        self.transform = transform
        self.column_name = column_name
        
        # Validate the task parameter
        if task not in ['regression', 'classification']:
            raise ValueError("Invalid task, must be 'classification' or 'regression'.")
        self.task = task
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        eid = self.annotations.iloc[index, 0]
        npfilepath = os.path.join(self.root_dir, f"{eid}.npy")
        img_data = np.load(npfilepath)
        img_tensor = torch.tensor(img_data.astype(np.float32))
        images = img_tensor.unsqueeze(0)
        
        if self.transform:
            images = self.transform(images)

        # Retrieve the label based on the task
        if self.task == 'regression':
            label = torch.tensor(float(self.annotations[self.column_name].iloc[index]))
        elif self.task == 'classification':
            label = torch.tensor(int(self.annotations[self.column_name].iloc[index]))
            if self.num_classes is not None:
                label = F.one_hot(label, num_classes=self.num_classes).float()
        
        return eid, images, label
