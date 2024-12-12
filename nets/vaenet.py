import torch.nn as nn
import torch.nn.functional as F
import torch
# from svdd_detectnet import DetectNet
from nets.eca_attention import eca_layer
from nets.attentionLayer import attentionLayer

def kaiming_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

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
        self.fc_bn = nn.BatchNorm1d(fc_output_dim)
        self.lstm  = nn.LSTM(68, 64, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.000000005)
        self.fc_mu = nn.Linear(64, fc_output_dim)
        self.fc_var = nn.Linear(64, fc_output_dim)
        self.apply(kaiming_init)
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
        x = self.dropout(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])

        mu = self.fc_mu(x)
        var =self.fc_var(x)

        return mu, var

class DeconvModule(nn.Module):
    def __init__(self, input_channels=64, output_channels=2, kernel_size=3, fc_output_dim=512):
        super(DeconvModule, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(64, 64, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.deconv2 = nn.ConvTranspose1d(64, 64, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.deconv3 = nn.ConvTranspose1d(64, 64, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.deconv4 = nn.ConvTranspose1d(64, 32, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.deconv5 = nn.ConvTranspose1d(32, 32, kernel_size=kernel_size, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm1d(32)
        self.deconv6 = nn.ConvTranspose1d(32, output_channels, kernel_size=kernel_size, stride=2, padding=2, output_padding=1)
        self.bn6 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.fc_audio = nn.Linear(fc_output_dim, 64)
        self.lstm = nn.LSTM(1, 69, batch_first=True)
        self.apply(kaiming_init)
    def forward(self, x):
        bs, c = x.size()
        x = self.relu(self.fc_audio(x))
        x, _ = self.lstm(x.unsqueeze(2))
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.relu(self.bn5(self.deconv5(x)))
        x = self.deconv6(x)
        return x

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
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu,var

class IMU_decoder(nn.Module):
    def __init__(self, fc_output_dim,input_dim=400,latent_dim=256):
        super(IMU_decoder, self).__init__()
        self.lstm1  = nn.LSTM(1,latent_dim,batch_first=True,bidirectional=False)
        self.lstm2  = nn.LSTM(latent_dim,1,batch_first=True,bidirectional=False)
        self.fc     = nn.Linear(fc_output_dim,input_dim)
        self.fc_final = nn.Linear(input_dim,input_dim)
        self.apply(kaiming_init)
    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(2)
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        x = x.squeeze(2)
        x = self.fc_final(x)
        return x


class FusionNet(nn.Module):
    def __init__(self, use_crossattention=0, feature_dim=128, dropout_rate=0.3, kernel_num=16, classes=2):
        super(FusionNet, self).__init__()
        self.use_crossattention = use_crossattention

        self.audioconv = Conv1DFeatureExtractor(2,feature_dim)
        self.audiodeconv = DeconvModule(fc_output_dim=feature_dim)

        self.imuencoder = IMU_encoder(feature_dim)
        self.imudecoder = IMU_decoder(feature_dim)

        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        self.kernel_num = kernel_num
        self.classes = classes
        self.relu = nn.ReLU()
        self.cross_atten1 = attentionLayer(self.feature_dim, 8, 0.3)
        self.cross_atten2 = attentionLayer(self.feature_dim, 8, 0.3)
        self.eca = eca_layer(channel=1)

        self.mu1_encoder = nn.Linear(feature_dim,feature_dim)
        self.mu2_encoder = nn.Linear(feature_dim,feature_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        [ba, ca, feature] = x.size()
        mu1, logvar1 = self.audioconv(x)
        mu2, logvar2  = self.imuencoder(y)
        # print(mu1,logvar1)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)

        if self.use_crossattention:
            # mu1 = self.mu1_encoder(mu1)
            # mu2 = self.mu2_encoder(mu2)
            fav = self.cross_atten1(mu1.unsqueeze(1), mu2.unsqueeze(1)).squeeze(1)
            fva = self.cross_atten2(mu2.unsqueeze(1), mu1.unsqueeze(1)).squeeze(1)
            f_all = torch.cat([fav, fva], dim=1).squeeze(1)
            z = f_all
            # self.cross_atten2(f_all.unsqueeze(1), f_all.unsqueeze(1)).squeeze(1)
            # z = self.eca(f_all.unsqueeze(1)).squeeze(1)
        else:
            mu1 = self.mu1_encoder(mu1)
            mu2 = self.mu2_encoder(mu2)
            z = torch.cat([mu1, mu2], dim=1)

        reconstructed_audio = self.audiodeconv(z1)
        reconstructed_imu   = self.imudecoder(z2)
        return reconstructed_audio, reconstructed_imu,mu1,logvar1,mu2,logvar2,z
    


