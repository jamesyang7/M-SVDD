import torch.nn as nn
import torch.nn.functional as F
import torch
# from svdd_detectnet import DetectNet
from nets.eca_attention import eca_layer
from nets.attentionLayer import attentionLayer

import torch.nn.functional as F

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

# class IMU_encoder(nn.Module):
#     def __init__(self, fc_output_dim=512):
#         super(IMU_encoder, self).__init__()
#         self.lstm1 = nn.LSTM(1, 256, batch_first=True, bidirectional=False)
#         self.bn1 = nn.BatchNorm1d(256)  # Batch Normalization for stability
#         self.lstm2 = nn.LSTM(256, fc_output_dim, batch_first=True, bidirectional=False)
#         self.bn2 = nn.BatchNorm1d(fc_output_dim)
#
#         self.fc_mu = nn.Linear(fc_output_dim, fc_output_dim)
#         self.fc_var = nn.Linear(fc_output_dim, fc_output_dim)
#         self.dropout = nn.Dropout(0.3)  # Dropout for regularization
#
#         self.apply(kaiming_init)
#
#     def forward(self, x):
#         x = x.unsqueeze(-1)  # (bs, seq_len, 1)
#         x, _ = self.lstm1(x)
#         x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # Apply BatchNorm
#         x = F.relu(x)  # Activation function
#
#         x, _ = self.lstm2(x)
#         x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
#         x = F.relu(x)
#
#         x = x[:, -1, :]  # Take the output from the last time step
#
#         x = self.dropout(x)  # Dropout layer
#         mu = self.fc_mu(x)
#         log_var = self.fc_var(x)
#         return mu, log_var
#
# class IMU_decoder(nn.Module):
#     def __init__(self, fc_output_dim, input_dim=400, latent_dim=256):
#         super(IMU_decoder, self).__init__()
#         self.fc = nn.Linear(fc_output_dim, input_dim)
#         self.bn_fc = nn.BatchNorm1d(input_dim)
#
#         self.lstm1 = nn.LSTM(1, latent_dim, batch_first=True, bidirectional=False)
#         self.bn1 = nn.BatchNorm1d(latent_dim)
#         self.lstm2 = nn.LSTM(latent_dim, 1, batch_first=True, bidirectional=False)
#
#         self.fc_final = nn.Linear(input_dim, input_dim)
#         self.apply(kaiming_init)
#
#     def forward(self, x):
#         # Fully connected layer to expand dimensions
#         x = self.fc(x)
#         x = self.bn_fc(x)  # Apply BatchNorm
#         x = F.relu(x)
#
#         # LSTM layers for decoding
#         x = x.unsqueeze(2)  # (bs, seq_len, 1)
#         x, _ = self.lstm1(x)
#         x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # Apply BatchNorm
#         x = F.relu(x)
#
#         x, _ = self.lstm2(x)
#         x = x.squeeze(2)
#
#         # Final layer to reconstruct the original input size
#         x = self.fc_final(x)
#         return x

class Conv1DFeatureExtractor(nn.Module):
    def __init__(self, input_channels, fc_output_dim=512, kernel_size=3):
        super(Conv1DFeatureExtractor, self).__init__()
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
        self.lstm  = nn.LSTM(68,256,batch_first=True,bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.apply(kaiming_init)
    def forward(self, x):
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
        x,_ = self.lstm(x)
        x = x[:, -1, :]
        # x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x,audio_feature


class DeconvModule(nn.Module):
    def __init__(self, input_channels=64, output_channels=2, kernel_size=3):
        super(DeconvModule, self).__init__()
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
    def forward(self, x):
        bs,c,f = x.size()
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.relu(self.bn5(self.deconv5(x)))
        x =  self.deconv6(x)
        return x




# class DeconvModule(nn.Module):
#     def __init__(self, input_channels=64, output_channels=2, kernel_size=3, dropout_prob=0.3):
#         super(DeconvModule, self).__init__()
#         self.deconv1 = nn.ConvTranspose1d(input_channels, 128, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.deconv3 = nn.ConvTranspose1d(64, 64, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
#         self.bn3 = nn.BatchNorm1d(64)
#         self.deconv4 = nn.ConvTranspose1d(64, 32, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
#         self.bn4 = nn.BatchNorm1d(32)
#         self.deconv5 = nn.ConvTranspose1d(32, 32, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
#         self.bn5 = nn.BatchNorm1d(32)
#         self.deconv6 = nn.ConvTranspose1d(32, output_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
#         self.bn6 = nn.BatchNorm1d(output_channels)
#
#         # Define Dropout layer with given probability
#         self.dropout = nn.Dropout(p=dropout_prob)
#
#         self.relu = nn.ReLU()
#         self.apply(kaiming_init)
#
#     def forward(self, x):
#         bs, c, f = x.size()
#         x = self.relu(self.bn1(self.deconv1(x)))
#         x = self.dropout(x)  # Apply dropout after activation
#         x = self.relu(self.bn2(self.deconv2(x)))
#         x = self.dropout(x)
#         x = self.relu(self.bn3(self.deconv3(x)))
#         x = self.dropout(x)
#         x = self.relu(self.bn4(self.deconv4(x)))
#         x = self.dropout(x)
#         x = self.relu(self.bn5(self.deconv5(x)))
#         x = self.dropout(x)
#         x = self.deconv6(x)  # No dropout in the final layer
#         return x


class IMU_encoder(nn.Module):
    def __init__(self, fc_output_dim=512):
        super(IMU_encoder, self).__init__()
        self.lstm1  = nn.LSTM(1,256,batch_first=True,bidirectional=False)
        self.lstm2  = nn.LSTM(256,fc_output_dim,batch_first=True,bidirectional=False)
        self.fc_mu     = nn.Linear(fc_output_dim,fc_output_dim)
        self.fc_var     = nn.Linear(fc_output_dim,fc_output_dim)
        self.apply(kaiming_init)
    def forward(self, x):
        # Flatten before passing to fully connected layer
        x = x.unsqueeze(-1)
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        x   = x[:,-1,:]
        x = self.fc_mu(x)
        # var = self.fc_var(x)
        # print(x.shape)
        return x,x

class IMU_decoder(nn.Module):
    def __init__(self, fc_output_dim,input_dim=400,latent_dim=256):
        super(IMU_decoder, self).__init__()
        self.lstm1  = nn.LSTM(1,latent_dim,batch_first=True,bidirectional=False)
        self.lstm2  = nn.LSTM(latent_dim,1,batch_first=True,bidirectional=False)
        self.fc     = nn.Linear(fc_output_dim,input_dim)
        self.fc_final = nn.Linear(input_dim,input_dim)
        self.relu = nn.ReLU()
        self.apply(kaiming_init)
    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(2)
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        x = x.squeeze(2)
        x = self.relu(self.fc_final(x))
        return x
    
# class IMU_decoder(nn.Module):
#     def __init__(self, fc_output_dim,input_dim=400,latent_dim=256):
#         super(IMU_decoder, self).__init__()
#         self.fc     = nn.Linear(fc_output_dim,input_dim)
#         self.fc_final = nn.Linear(input_dim,input_dim)
#         self.relu = nn.ReLU()
#         self.apply(kaiming_init)
#     def forward(self, x):
#         x = self.fc(x)
#         x = self.relu(self.fc_final(x))
#         return x

