import torch
import torch.nn as nn
import librosa

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, original_imu, reconstructed_imu, original_audio, reconstructed_audio):
        imu_loss = self.mse_loss(reconstructed_imu,original_imu)
        audio_loss = self.mse_loss(reconstructed_audio,original_audio)
        total_loss = 0*imu_loss + audio_loss
        return total_loss
    

class CompactFeatureLoss(nn.Module):
    def __init__(self):
        super(CompactFeatureLoss, self).__init__()

    def forward(self, features):
        mean_features = torch.mean(features, dim=0)
        loss = torch.mean((features - mean_features) ** 2)
        return loss

    


