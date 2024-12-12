import torch
import torch.nn as nn

class CompactFeatureLoss(nn.Module):
    def __init__(self):
        super(CompactFeatureLoss, self).__init__()

    def forward(self, features):
        mean_features = torch.mean(features, dim=0)
        loss = torch.mean((features - mean_features) ** 2)
        return loss