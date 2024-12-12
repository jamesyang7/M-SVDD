import torch.nn as nn
import torch.nn.functional as F
import torch
# from svdd_detectnet import DetectNet
from nets.eca_attention import eca_layer
from nets.attentionLayer import attentionLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
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
        self.conv5 = nn.Conv1d(64, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # self.apply(kaiming_init)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = x.permute(0,2,1)
        return x

class FFT_CNN_IFFT_Model(nn.Module):
    def __init__(self, input_channels=2, output_dim=512, kernel_size=3):
        super(FFT_CNN_IFFT_Model, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64,32, kernel_size=kernel_size, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32,32, kernel_size=kernel_size, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Step 1: Compute FFT
        x = x.permute(0,2,1)
        fft_data = torch.fft.rfft(x, dim=1) 
        peak_value = torch.max(torch.abs(x), dim=1, keepdim=True)[0]  
        peak_value = peak_value.permute(0, 2, 1)  
        magnitude = torch.abs(fft_data)  
        magnitude = magnitude.permute(0, 2, 1)
        magnitude = magnitude * peak_value
        x = self.relu(self.bn1(self.conv1(magnitude)))
        x = self.pool2(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = x.permute(0,2,1)
        return x  # Return reconstructed signal and features
    

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        # Ensure d_model matches feature_dim
        assert feature_dim == self.d_model, "Feature dimension and d_model must match."

        # Compute positional encodings
        position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=x.device) *
                             (-torch.log(torch.tensor(10000.0)) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add positional encoding to input
        return x + pe.unsqueeze(0)  # Add batch dimension


class CombinedFeatureExtractor(nn.Module):
    def __init__(self, fft_input_channels=2, conv_input_channels=2, embedding_dim=32, num_heads=4, num_layers=2, output_dim=512):
        super(CombinedFeatureExtractor, self).__init__()
        # FFT-based feature extractor
        self.fft_extractor = FFT_CNN_IFFT_Model(input_channels=fft_input_channels)

        # Conv1D-based feature extractor
        self.conv_extractor = Conv1DFeatureExtractor(input_channels=conv_input_channels)

        # Sinusoidal positional embedding
        self.positional_embedding = SinusoidalPositionalEmbedding(d_model=embedding_dim)

        # Transformer Encoder
        self.transformer_dim = embedding_dim
        encoder_layer = TransformerEncoderLayer(d_model=self.transformer_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layer to map transformer output to final features
        self.fc = nn.Linear(1024, output_dim)

    def forward(self, x):
        bs = x.shape[0]

        # Extract features from FFT-based extractor
        fft_features = self.fft_extractor(x)  # Shape: (batch_size, seq_len, feature_dim)

        # Extract features from Conv1D-based extractor
        conv_features = self.conv_extractor(x)  # Shape: (batch_size, seq_len, feature_dim)

        # Concatenate features
        combined_features = torch.cat((fft_features, conv_features), dim=1)  # Shape: (batch_size, seq_len, combined_feature_dim)

        # Add sinusoidal positional embeddings
        combined_features = self.positional_embedding(combined_features)

        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(combined_features)  # Shape: (batch_size, seq_len, embedding_dim)
        transformer_output = transformer_output.view(bs, -1)  # Flatten for fully connected layer

        # Final feature representation
        final_features = self.fc(transformer_output)  # Shape: (batch_size, output_dim)

        return final_features, conv_features.permute(0, 2, 1)

class DeconvModule(nn.Module):
    def __init__(self, input_channels=32, output_channels=2, kernel_size=3):
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
        self.deconv6 = nn.ConvTranspose1d(32, output_channels, kernel_size=5, stride=4, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        # self.apply(kaiming_init)
    def forward(self, x):
        bs,c,f = x.size()
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        x = self.relu(self.bn5(self.deconv5(x)))
        x =  self.deconv6(x)
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
    
