import torch.nn as nn
import torch.nn.functional as F
import torch
from eca_attention import eca_layer
from attentionLayer import attentionLayer

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
        self.fc_mu = nn.Linear(4800, fc_output_dim)
        self.fc_logvar = nn.Linear(4800, fc_output_dim)
        self.fc_bn = nn.BatchNorm1d(fc_output_dim)  

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

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, audio_feature


class DeconvModule(nn.Module):
    def __init__(self, input_channels=64, output_channels=4, kernel_size=3):
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

    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.relu(self.bn5(self.deconv5(x)))
        # x =  self.deconv6(x)
        x = torch.sigmoid(self.bn6(self.deconv6(x)))
        return x

class IMU_encoder(nn.Module):
    def __init__(self, input_channels, fc_output_dim=512, kernel_size=3, dropout_prob=0.3):
        super(IMU_encoder, self).__init__()
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
        self.conv6 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=2)
        self.bn6 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc_mu = nn.Linear( 768, fc_output_dim)
        self.fc_logvar = nn.Linear(768 , fc_output_dim)
        self.fc_bn = nn.BatchNorm1d(fc_output_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        imu_feature = x
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, imu_feature

class IMU_decoder(nn.Module):
    def __init__(self, input_channels=64, output_channels=1, kernel_size=3, dropout_prob=0.3):
        super(IMU_decoder, self).__init__()
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
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(192, 200)

    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.dropout(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x


class FusionNet(nn.Module):
    def __init__(self, use_crossattention=0, feature_dim=512, dropout_rate=0.3, kernel_num=16, classes=2):
        super(FusionNet, self).__init__()
        self.use_crossattention = use_crossattention

        self.audioconv = Conv1DFeatureExtractor(4,feature_dim)
        self.audiodeconv = DeconvModule()

        self.imuencoder = IMU_encoder(1,feature_dim)
        self.imudecoder = IMU_decoder()

        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        self.kernel_num = kernel_num
        self.classes = classes
        self.relu = nn.ReLU()
        self.cross_atten = attentionLayer(self.feature_dim, 8, 0.3)
        self.cross_atten2 = attentionLayer(self.feature_dim*2, 8, 0.3)
        self.eca = eca_layer(channel=1)

        self.fc1 = nn.Linear(self.feature_dim*2, self.feature_dim*2)
        self.fc1_bn = nn.BatchNorm1d(self.feature_dim*2)  
        self.fc2 = nn.Linear(self.feature_dim*2, self.feature_dim*2)
        self.fc2_bn = nn.BatchNorm1d(self.feature_dim*2) 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        [ba, ca, feature] = x.size()

        mu, logvar, recons_feature = self.audioconv(x)
        z = self.reparameterize(mu, logvar)
        z = z.view(z.size(0), z.size(1))

        imu_mu, imu_logvar, imu_recons = self.imuencoder(y.unsqueeze(1))
        # print('imu decoder',imu_recons.shape)
        imu_z = self.reparameterize(imu_mu, imu_logvar)
        imu_z = imu_z.view(imu_z.size(0), imu_z.size(1))
        # print('imu_z', imu_z.shape)

        if self.use_crossattention:
            fav = self.cross_atten(imu_z.unsqueeze(1), z.unsqueeze(1)).squeeze(1)
            fva = self.cross_atten(z.unsqueeze(1), imu_z.unsqueeze(1)).squeeze(1)
            f_all = torch.cat([fav, fva], dim=1).squeeze(1)
            f_all = self.eca(f_all.unsqueeze(1)).squeeze(1)
        else:
            f_all = torch.cat([z, imu_z], dim=1)
        
        # print('++++++++++++++++',f_all.shape)

        f_all = self.relu(self.fc1(f_all))
        f_all = self.fc2(f_all)

        reconstructed_audio = self.audiodeconv(recons_feature)
        reconstructed_imu   = self.imudecoder(imu_recons)
        # print('recon_imu shape is', reconstructed_imu.shape)

        return f_all, reconstructed_audio, reconstructed_imu, mu, logvar, imu_mu, imu_logvar
    