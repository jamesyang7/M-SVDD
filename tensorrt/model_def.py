import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


class Conv1DFeatureExtractor(nn.Module):
    def __init__(self, input_channels: int, fc_output_dim: int = 512, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256, fc_output_dim)
        self.fc_bn = nn.BatchNorm1d(fc_output_dim)
        self.lstm = nn.LSTM(68, 256, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        audio_feature = x
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x, audio_feature


class DeconvModule(nn.Module):
    def __init__(self, input_channels: int = 64, output_channels: int = 2, kernel_size: int = 3):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(input_channels, 128, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.deconv3 = nn.ConvTranspose1d(64, 64, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.deconv4 = nn.ConvTranspose1d(64, 32, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.deconv5 = nn.ConvTranspose1d(32, 32, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm1d(32)
        self.deconv6 = nn.ConvTranspose1d(32, output_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.relu(self.bn5(self.deconv5(x)))
        x = self.deconv6(x)
        return x


class IMUEncoder(nn.Module):
    def __init__(self, fc_output_dim: int = 512):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 256, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(256, fc_output_dim, batch_first=True, bidirectional=False)
        self.fc_mu = nn.Linear(fc_output_dim, fc_output_dim)
        self.fc_var = nn.Linear(fc_output_dim, fc_output_dim)
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.fc_mu(x)
        return x, x


class IMUDecoder(nn.Module):
    def __init__(self, fc_output_dim: int, input_dim: int = 400, latent_dim: int = 256):
        super().__init__()
        self.lstm1 = nn.LSTM(1, latent_dim, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(latent_dim, 1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(fc_output_dim, input_dim)
        self.fc_final = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.unsqueeze(2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.squeeze(2)
        x = self.relu(self.fc_final(x))
        return x


class AttentionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        src = src.transpose(0, 1)
        tar = tar.transpose(0, 1)
        src2 = self.self_attn(tar, src, src, attn_mask=None, key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1)
        return src


class GaussianSVDDModel(nn.Module):
    """
    Inference-safe variant of the repo model.
    Unlike the original training code, mu/sigma_inv/radius are registered as buffers
    so they travel with the model during ONNX export.
    """
    def __init__(self, output_dim: int = 32, feature_dim: int = 32, reg_const: float = 1e-4):
        super().__init__()
        self.audio_encoder = Conv1DFeatureExtractor(2, feature_dim)
        self.audio_decoder = DeconvModule()
        self.imu_encoder = IMUEncoder(fc_output_dim=feature_dim)
        self.imu_decoder = IMUDecoder(fc_output_dim=feature_dim)
        self.cross_atten1 = AttentionLayer(feature_dim, 8, 0.3)
        self.cross_atten2 = AttentionLayer(feature_dim, 8, 0.3)
        self.fc_audio = nn.Linear(4352, 4410)
        self.fc_imu = nn.Linear(400, 400)
        self.fc1 = nn.Linear(feature_dim, output_dim)
        self.reg_const = reg_const

        self.register_buffer("mu", torch.zeros(output_dim, dtype=torch.float32))
        self.register_buffer("sigma_inv", torch.eye(output_dim, dtype=torch.float32))
        self.register_buffer("radius", torch.ones(1, dtype=torch.float32))

    def load_checkpoint(self, checkpoint_path: str, map_location: str = "cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if "mu" in checkpoint:
                self.mu.copy_(checkpoint["mu"].detach().float().view(-1))
            if "sigma_inv" in checkpoint:
                self.sigma_inv.copy_(checkpoint["sigma_inv"].detach().float())
            if "radius" in checkpoint:
                radius = checkpoint["radius"]
                if isinstance(radius, nn.Parameter):
                    radius = radius.data
                self.radius.copy_(radius.detach().float().view(-1)[:1])
        else:
            self.load_state_dict(checkpoint, strict=False)
        return self

    def mahalanobis_distance(self, z: torch.Tensor) -> torch.Tensor:
        diff = z - self.mu.unsqueeze(0)
        return torch.sqrt(torch.sum(diff * (diff @ self.sigma_inv), dim=1) + 1e-12)

    def forward(self, x_audio: torch.Tensor, x_imu: torch.Tensor):
        batch = x_audio.size(0)
        audio_feature, recons_feature = self.audio_encoder(x_audio)
        audio_feature_flat = audio_feature.view(batch, -1).float()

        imu_feature, imu_recons = self.imu_encoder(x_imu)
        imu_feature_flat = imu_feature.view(batch, -1).float()

        fav = self.cross_atten1(imu_feature_flat.unsqueeze(1), audio_feature_flat.unsqueeze(1)).squeeze(1)
        fva = self.cross_atten2(audio_feature_flat.unsqueeze(1), imu_feature_flat.unsqueeze(1)).squeeze(1)
        f_all = fav + fva
        z_combined = self.fc1(f_all)

        distances = self.mahalanobis_distance(z_combined)
        anomaly_score = distances / torch.clamp(self.radius, min=1e-12)

        x_audio_recon = self.audio_decoder(recons_feature)
        x_audio_recon = self.fc_audio(x_audio_recon)
        x_imu_recon = self.imu_decoder(imu_recons)
        x_imu_recon = self.fc_imu(x_imu_recon)

        return {
            "distance": distances,
            "radius": self.radius.expand_as(distances),
            "anomaly_score": anomaly_score,
            "audio_recon": x_audio_recon,
            "imu_recon": x_imu_recon,
            "embedding": z_combined,
        }


class ONNXExportWrapper(nn.Module):
    def __init__(self, core: GaussianSVDDModel):
        super().__init__()
        self.core = core.eval()

    def forward(self, x_audio: torch.Tensor, x_imu: torch.Tensor):
        out = self.core(x_audio, x_imu)
        return (
            out["distance"],
            out["radius"],
            out["anomaly_score"],
            out["audio_recon"],
            out["imu_recon"],
            out["embedding"],
        )
