import torch.nn as nn
import torch.nn.functional as F
import torch
# from svdd_detectnet import DetectNet
from eca_attention import eca_layer
from attentionLayer import attentionLayer


class Encoder_audio(nn.Module):
    def __init__(self, input_channels, feature_dim):
        super(Encoder_audio, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm2d added
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm2d added
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # BatchNorm2d added
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)  # BatchNorm2d added
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 5 * 5, feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        b, c, w, h = x.size()
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # BatchNorm2d applied
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # BatchNorm2d applied
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # BatchNorm2d applied
        x = self.pool(self.relu(self.bn4(self.conv4(x))))  # BatchNorm2d applied

        recon_f = x
        x = x.view(b, -1)
        x = self.relu(self.fc(x))
        x = self.fc2(x)
        return x, recon_f

class Decoder_audio(nn.Module):
    def __init__(self, output_channels):
        super(Decoder_audio, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)  # BatchNorm2d added
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm2d added
        self.deconv3 = nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(4)  # BatchNorm2d added
        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(1764, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4800)
        )

    def forward(self, x):
        # Deconvolution layers
        x = self.relu(self.bn1(self.deconv1(x)))  # BatchNorm2d applied
        x = self.relu(self.bn2(self.deconv2(x)))  # BatchNorm2d applied
        x = self.relu(self.bn3(self.deconv3(x)))  # BatchNorm2d applied

        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1)
        x = self.mlp(x)

        return x
    

class Conv1DFeatureExtractor(nn.Module):
    def __init__(self, input_channels, fc_output_dim=512, kernel_size=3):
        super(Conv1DFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)  # BatchNorm1d added
        self.conv2 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)  # BatchNorm1d added
        self.conv3 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)  # BatchNorm1d added
        self.conv4 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(64)  # BatchNorm1d added
        self.conv5 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(64)  # BatchNorm1d added
        self.conv6 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(64)  # BatchNorm1d added
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64*75, fc_output_dim)
        self.fc_bn = nn.BatchNorm1d(fc_output_dim)  # BatchNorm1d added

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
        # Flatten before passing to fully connected layer
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc_bn(self.fc(x)))
        return x,audio_feature


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
        x =  self.deconv6(x)
        return x

class FusionNet(nn.Module):
    def __init__(self, use_crossattention=0, feature_dim=512, dropout_rate=0.3, kernel_num=16, classes=2):
        super(FusionNet, self).__init__()
        self.use_crossattention = use_crossattention

        self.audioconv = Conv1DFeatureExtractor(4,feature_dim)
        self.audiodeconv = DeconvModule()

        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        self.kernel_num = kernel_num
        self.classes = classes
        self.relu = nn.ReLU()
        self.cross_atten = attentionLayer(self.feature_dim, 8, 0.3)
        self.eca = eca_layer(channel=1)

        self.encoder_audio = Encoder_audio(4, self.feature_dim)
        self.decoder_audio = Decoder_audio(self.feature_dim)

        self.imufc1 = nn.Linear(20, 128)
        self.imufc1_bn = nn.BatchNorm1d(128)  # BatchNorm1d added
        self.imufc2 = nn.Linear(128, feature_dim)
        self.imudefc1 = nn.Linear(feature_dim, 128)
        self.imudefc1_bn = nn.BatchNorm1d(128)  # BatchNorm1d added
        self.imudefc2 = nn.Linear(128, 20)

        self.fc1 = nn.Linear(self.feature_dim*2, self.feature_dim*2)
        self.fc1_bn = nn.BatchNorm1d(self.feature_dim*2)  # BatchNorm1d added
        self.fc2 = nn.Linear(self.feature_dim*2, self.feature_dim*2)
        self.fc2_bn = nn.BatchNorm1d(self.feature_dim*2)  # BatchNorm1d added

    def forward(self, x, y):
        [ba, ca, feature] = x.size()

        audio_feature,recons_feature = self.audioconv(x)
        audio_feature_flat = audio_feature.view(ba, -1).float()

        imu_feature =  self.relu(self.imufc1_bn(self.imufc1(y)))  # BatchNorm1d applied
        imu_feature =  self.imufc2(imu_feature)

        if self.use_crossattention:
            fav = self.cross_atten(imu_feature.unsqueeze(1), audio_feature_flat.unsqueeze(1)).squeeze(1)
            fva = self.cross_atten(audio_feature_flat.unsqueeze(1), imu_feature.unsqueeze(1)).squeeze(1)
            f_all = torch.cat([fav, fva], dim=1)
            f_all = self.eca(f_all.unsqueeze(1)).squeeze(1)
        else:
            f_all = torch.cat([audio_feature_flat, imu_feature], dim=1)


        f_all = self.fc1(f_all)



        # reconstructed_audio = self.audiodeconv(recons_feature)

        # reconstructed_imu = self.relu(self.imudefc1_bn(self.imudefc1(imu_feature)))  # BatchNorm1d applied
        # reconstructed_imu = self.imudefc2(reconstructed_imu)

        return f_all
    



    
