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
        total_loss =  audio_loss+imu_loss
        return total_loss
    

class CompactFeatureLoss(nn.Module):
    def __init__(self):
        super(CompactFeatureLoss, self).__init__()

    def forward(self, features):
        mean_features = torch.mean(features, dim=0)
        loss = torch.mean((features - mean_features) ** 2)
        return loss

def huber_loss(input, target, delta=1.0, reduction='sum'):
    abs_diff = torch.abs(input - target)
    quadratic = torch.min(abs_diff, torch.tensor(delta))
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    if reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'mean':
        return torch.mean(loss)
    else:
        return loss

def VAE_loss(recon_x1, x1, recon_x2, x2, mu1, logvar1, mu2, logvar2, z, c, lambda_c=1.0):
    delta = 1.0  # Huber loss delta parameter
    Huber1 = huber_loss(recon_x1, x1, delta, reduction='sum')
    Huber2 = huber_loss(recon_x2, x2, delta, reduction='sum')
    reconstruct_loss = Huber1+Huber2

    KLD1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    KLD2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    KLD_loss = KLD1+KLD2

    center_loss = torch.sum((z - z) ** 2)

    total_loss = reconstruct_loss +  0.5*KLD_loss+center_loss
    return total_loss, reconstruct_loss, KLD_loss, center_loss
    # return MSE1 + MSE2 + KLD1 + KLD2 + lambda_c * center_loss
    

class ReconstructionLoss_audio(nn.Module):
    def __init__(self):
        super(ReconstructionLoss_audio, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self,  original_audio, reconstructed_audio):

        audio_loss = self.mse_loss(reconstructed_audio,original_audio)
        total_loss =  audio_loss
        return total_loss
    

