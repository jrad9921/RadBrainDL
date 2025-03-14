import torch
import torch.nn as nn
import torch.nn.functional as F

class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=1, dropout=True, task_type="regression"):
        """
        task_type: "regression" for continuous output, "classification" for categorical output
        """
        super(SFCN, self).__init__()
        self.task_type = task_type  # Store the task type
        n_layer = len(channel_number)

        # Feature extractor
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            in_channel = 1 if i == 0 else channel_number[i - 1]
            out_channel = channel_number[i]
            maxpool = i < n_layer - 1
            kernel_size = 3 if maxpool else 1
            padding = 1 if maxpool else 0
            self.feature_extractor.add_module(
                f'conv_{i}',
                self.conv_layer(in_channel, out_channel, maxpool=maxpool, kernel_size=kernel_size, padding=padding)
            )

        # Classifier
        self.classifier = nn.Sequential()
        avg_shape = [3, 3, 3]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        
        in_channel = channel_number[-1]
        self.classifier.add_module(f'conv_{n_layer}', nn.Conv3d(in_channel, output_dim, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        layers = [
            nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
            nn.BatchNorm3d(out_channel),
            nn.ReLU()
        ]
        if maxpool:
            layers.append(nn.MaxPool3d(2, stride=maxpool_stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_f = self.feature_extractor(x)
        x = self.classifier(x)
        
        if self.task_type == "classification":
            x = x.mean(dim=[2, 3, 4])  # Global average pooling
            x = F.log_softmax(x, dim=1)  # Softmax activation for classification
        else:  # Regression
            x = x.mean(dim=[2, 3, 4])  # Global average pooling
            x = x.squeeze()  # Remove unnecessary dimensions

        return x
