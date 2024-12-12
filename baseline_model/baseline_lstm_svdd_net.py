import torch.nn as nn
import torch
from eca_attention import eca_layer
from attentionLayer import attentionLayer


class SVDDLayer(nn.Module):
    def __init__(self, feature_dim):
        super(SVDDLayer, self).__init__()
        self.c = nn.Parameter(torch.randn(feature_dim))

    def forward(self, x):
        dist = torch.norm(x - self.c, dim=1)
        return dist

class FusionNet(nn.Module):
    def __init__(self, use_crossattention=0, feature_dim=512, dropout_rate=0.3, kernel_num=16, classes=2, threshold=0.5, lstm_input_dim=20, lstm_hidden_dim=64, lstm_num_layers=2):
        super(FusionNet, self).__init__()
        self.use_crossattention = use_crossattention
        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        self.kernel_num = kernel_num
        self.classes = classes
        self.threshold = threshold
        self.relu = nn.ReLU()
        self.cross_atten = attentionLayer(self.feature_dim, 8, 0.3)
        self.eca = eca_layer(channel=1)
        self.lstm1 = nn.LSTM(4800, self.feature_dim, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(1, self.feature_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(self.feature_dim, self.feature_dim, num_layers=1, batch_first=True)

        self.imufc1 = nn.Linear(20, 128)
        self.imufc1_bn = nn.BatchNorm1d(128)
        self.imufc2 = nn.Linear(128, feature_dim)
        self.imudefc1 = nn.Linear(feature_dim, 128)
        self.imudefc1_bn = nn.BatchNorm1d(128)
        self.imudefc2 = nn.Linear(128, 20)

        self.fc1 = nn.Linear(self.feature_dim*2, self.feature_dim * 2)
        self.fc1_bn = nn.BatchNorm1d(self.feature_dim * 2)
        self.fc2 = nn.Linear(self.feature_dim * 2, self.feature_dim * 2)
        self.fc2_bn = nn.BatchNorm1d(self.feature_dim * 2)
        self.svdd = SVDDLayer(self.feature_dim * 2)

    def forward(self, x, y):
        [ba, ca, feature] = x.size()


        audio_feature, _ = self.lstm1(x)
        _, (audio_feature, _) = self.lstm2(audio_feature)
        audio_feature = audio_feature.view(ba, -1).float()

        y = y.unsqueeze(2)  
        imu_feature, _ = self.lstm3(y)  
        imu_feature, (h_n, c_n) = self.lstm2(imu_feature)  
        
        # Extract the last time step output from the LSTM
        imu_feature = imu_feature[:, -1, :]
        imu_feature = imu_feature.view(ba, -1).float()

        if self.use_crossattention:
            fav = self.cross_atten(imu_feature.unsqueeze(1), audio_feature.unsqueeze(1)).squeeze(1)
            fva = self.cross_atten(audio_feature.unsqueeze(1), imu_feature.unsqueeze(1)).squeeze(1)
            f_all = torch.cat([fav, fva], dim=1)
            f_all = self.eca(f_all.unsqueeze(1)).squeeze(1)
        else:
            f_all = torch.cat([audio_feature, imu_feature], dim=1)

        f_all = self.relu(self.fc1(f_all))
        f_all = self.fc2(f_all)

        return f_all


