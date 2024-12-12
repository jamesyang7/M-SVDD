import torch

def initialize_center_c(dataloader, model, latent_dim,device):
    c = torch.zeros(int(latent_dim)).to(device)
    model.eval()
    with torch.no_grad():
        n_samples = 0
        for i, data in enumerate(dataloader, 0):
            spec, imu, audio,imu_recons,audio_recons = data
            spec, imu, audio,imu_recons,audio_recons = spec.to(device), imu.to(device), audio.to(device),imu_recons.to(device), audio_recons.to(device)
            recon_x1, recon_x2, mu1, logvar1, mu2, logvar2, z = model(audio, imu)
            n_samples += z.size(0)
            c += torch.sum(z, dim=0)
    c /= n_samples
    return c
