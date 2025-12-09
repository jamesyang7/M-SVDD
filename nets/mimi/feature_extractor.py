import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvAutoencoder(nn.Module):
    def __init__(self):
        super(DeepConvAutoencoder, self).__init__()

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),    # -> (16, 64, 313)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   # -> (32, 32, 157)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # -> (64, 16, 79)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (128, 8, 40)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1), # -> (256, 4, 20)
            nn.ReLU(),
        )

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 8, 40)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 80)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 32, 160)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # -> (16, 64, 320)
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),             # -> (1, 64, 320)
        )

        # Final crop to match input shape (1, 64, 313)
        self.final_crop = lambda x: x[..., :64, :313]

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        x_recon = self.final_crop(x_recon)
        return x_recon, z

class DeepConvAutoencoder_svdd(nn.Module):
    def __init__(self):
        super(DeepConvAutoencoder_svdd, self).__init__()

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),    # -> (16, 64, 313)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   # -> (32, 32, 157)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # -> (64, 16, 79)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (128, 8, 40)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1), # -> (256, 4, 20)
            nn.ReLU(),
        )

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 8, 40)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 80)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 32, 160)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # -> (16, 64, 320)
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),             # -> (1, 64, 320)
        )

        # Final crop to match input shape (1, 64, 313)
        self.final_crop = lambda x: x[..., :64, :313]
        self.fc = nn.Linear(5120,16)
    def forward(self, x):
        z = self.encoder(x)
        z_svdd = z.view(z.size(0),-1)
        z_svdd = self.fc(z_svdd)
        x_recon = self.decoder(z)
        x_recon = self.final_crop(x_recon)
        return x_recon, z_svdd


import torch
import torch.nn as nn

class DeepConvVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(DeepConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),    # (16, 64, 313)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   # (32, 32, 157)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # (64, 16, 79)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 8, 40)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),  # (64, 4, 20)
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.encoder_output_dim = 64 * 4 * 20  # = 5120

        # Latent space
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_latent_to_feature = nn.Linear(latent_dim, self.encoder_output_dim)

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            # nn.Unflatten(1, (64, 4, 20)),                              # (64, 4, 20)
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 8, 40)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 16, 80)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # (32, 32, 160)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # (16, 64, 320)
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),             # (1, 64, 320)
        )

        self.final_crop = lambda x: x[..., :64, :313]  # Crop time axis to 313
        self.fc = nn.Linear(latent_dim,latent_dim)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z = self.encoder(x)
        z_flat = self.flatten(z)

        mu = self.fc_mu(z_flat)
        logvar = self.fc_logvar(z_flat)
        z = self.fc(mu)
        z_sample = self.reparameterize(mu, logvar)
        z_feature = self.fc_latent_to_feature(z_sample)

        z_feature = z_feature.view(-1, 64, 4, 20)
        x_recon = self.decoder(z_feature)
        x_recon = self.final_crop(x_recon)

        return x_recon, mu, logvar,z

# model = DeepConvAutoencoder()
# x = torch.randn(1, 1, 64, 313)  # (batch, channels, height, width)
# x_recon, latent = model(x)
# print(latent.shape)  # torch.Size([1, 1, 64, 313])
