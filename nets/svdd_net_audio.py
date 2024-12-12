import torch.nn as nn
import torch.nn.functional as F
import torch
# from svdd_detectnet import DetectNet
from nets.eca_attention import eca_layer
from nets.attentionLayer import attentionLayer
from nets.transformer import TranAD_Basic


class Encoder_audio(nn.Module):
    def __init__(self, input_channels, feature_dim):
        super(Encoder_audio, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32) 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 4 * 4, feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x):
        b, c, w, h = x.size()
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  
        x = self.pool(self.relu(self.bn4(self.conv4(x))))  
        recon_f = x
        x = x.view(b, -1)
        x = self.relu(self.fc(x))
        x = self.fc2(x)
        return x, recon_f
    


class Decoder_audio_2d(nn.Module):
    def __init__(self, output_channels):
        super(Decoder_audio_2d, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)  
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)  
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32,output_channels, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(4)    
        self.relu = nn.ReLU()

    def forward(self, x):
  
        x = self.relu(self.deconv1(x))
        x = self.relu((self.deconv2(x)))  
        x = self.relu((self.deconv3(x)))
        x = (self.deconv4(x))
        return x


class Decoder_audio(nn.Module):
    def __init__(self, output_channels):
        super(Decoder_audio, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)  
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)  
        self.deconv3 = nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(4)  
        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(1764, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4800)
        )

    def forward(self, x):
  
        x = self.relu(self.bn1(self.deconv1(x)))  
        x = self.relu(self.bn2(self.deconv2(x)))  
        x = self.relu(self.bn3(self.deconv3(x)))  

        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1)
        x = self.mlp(x)

        return x
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


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
        self.fc = nn.Linear(64 * 64, fc_output_dim)  # Adjust 75 based on your input size
        self.fc_bn = nn.BatchNorm1d(fc_output_dim)
        self.lstm  = nn.LSTM(68,64,batch_first=True,bidirectional=False)
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
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc_bn(self.fc(x)))
        return x, audio_feature


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
        self.relu = nn.ReLU()

        # Apply Kaiming initialization
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.relu(self.bn5(self.deconv5(x)))
        x = self.deconv6(x)
        return x


class FusionNet_audio(nn.Module):
    def __init__(self, use_crossattention=0, feature_dim=512, dropout_rate=0.3, kernel_num=16, classes=2):
        super(FusionNet_audio, self).__init__()
        self.use_crossattention = use_crossattention

        self.audioconv = Conv1DFeatureExtractor(2)
        self.audiodeconv = DeconvModule()

        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        self.kernel_num = kernel_num
        self.classes = classes
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.feature_dim, self.feature_dim)
        self.fc1_bn = nn.BatchNorm1d(self.feature_dim)
        self.fc2 = nn.Linear(self.feature_dim, self.feature_dim)
        self.fc2_bn = nn.BatchNorm1d(self.feature_dim)

        self.fc_recons = nn.Linear(4352, 4410)  # Adjust this based on your input size

        # Apply Kaiming initialization
        self.apply(kaiming_init)

    def forward(self, x, y):
        ba, ca, feature = x.size()

        audio_feature, recons_feature = self.audioconv(x)

        audio_feature_flat = audio_feature.view(ba, -1).float()

        f_all = self.relu(self.fc1_bn(self.fc1(audio_feature_flat)))
        f_all = self.fc2_bn(self.fc2(f_all))

        reconstructed_audio = self.audiodeconv(recons_feature)
        reconstructed_audio = self.fc_recons(reconstructed_audio)

        return f_all, reconstructed_audio
    


class FusionNet_audio_2d(nn.Module):
    def __init__(self, use_crossattention=0, feature_dim=512, dropout_rate=0.3, kernel_num=16, classes=2):
        super(FusionNet_audio_2d, self).__init__()
        self.use_crossattention = use_crossattention
        # self.audioconv = Encoder_audio(4,feature_dim)
        # self.audiodeconv = Decoder_audio_2d(output_channels=4)

        self.audioconv = TranAD_Basic(48)
        
        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        self.kernel_num = kernel_num
        self.classes = classes
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(4800, self.feature_dim)
        self.fc1_bn = nn.BatchNorm1d(self.feature_dim)  
        self.fc2 = nn.Linear(self.feature_dim, self.feature_dim)
        self.fc2_bn = nn.BatchNorm1d(self.feature_dim) 

        self.fc_res1 = nn.Linear(4800, 1024)
        self.fc_res2 = nn.Linear(1024, 4800)

    def forward(self, x, y):

        [bs,feature,seq] = x.size()
        x = torch.permute(x,(2,0,1))

        audio_feature,recons_feature = self.audioconv(x,x)
        audio_feature = torch.permute(audio_feature,(1,0,2))
        audio_feature = audio_feature.reshape(bs,-1)

        f_all = self.relu(self.fc1_bn(self.fc1(audio_feature)))
        f_all = self.fc2(f_all)

        recons_feature = torch.permute(recons_feature,(1,0,2))
        recons_feature = recons_feature.reshape(bs,-1)
        recons_feature = self.relu(self.fc_res1(recons_feature))
        recons_feature = self.fc_res2(recons_feature)
        # print(recons_feature.shape)
        # recons_feature = recons_feature.unsqueeze(0)

        return f_all, recons_feature
