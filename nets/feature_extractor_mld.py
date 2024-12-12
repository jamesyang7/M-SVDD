import torch.nn as nn
import torch.nn.functional as F
import torch
# from svdd_detectnet import DetectNet
import torch.nn.functional as F
import importlib
import torch
import numpy as np
from torch.nn.utils import weight_norm


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


class Encoder(nn.Module):
    def __init__(self, input_feature=55,fc_output_dim=512):
        super(Encoder, self).__init__()
        self.lstm1  = nn.LSTM(input_feature,256,batch_first=True,bidirectional=False,num_layers=2)
        self.lstm2  = nn.LSTM(256,fc_output_dim,batch_first=True,bidirectional=False,num_layers=2)
        self.fc_mu     = nn.Linear(fc_output_dim,fc_output_dim)
        self.fc1     = nn.Linear(fc_output_dim,fc_output_dim)
        self.relu    = nn.ReLU()
        self.apply(kaiming_init)
    def forward(self, x):
        # Flatten before passing to fully connected layer
        # x = x.unsqueeze(-1)
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        reconstuct_x = x
        x   = x[:,-1,:]
        x = self.relu(self.fc1(x))
        x = self.fc_mu(x)
        # var = self.fc_var(x)
        # print(x.shape)
        return x,reconstuct_x

class Decoder(nn.Module):
    def __init__(self, input_dim=55,feature_dim=32,latent_dim=256):
        super(Decoder, self).__init__()
        self.lstm1  = nn.LSTM(feature_dim,latent_dim,batch_first=True,bidirectional=False,num_layers=2)
        self.lstm2  = nn.LSTM(latent_dim,input_dim,batch_first=True,bidirectional=False,num_layers=2)
        self.fc1    = nn.Linear(input_dim,input_dim)
        self.relu    = nn.ReLU()
        self.fc_final = nn.Linear(input_dim,input_dim)
        self.apply(kaiming_init)
    def forward(self, x):
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(x)
        x   = self.relu(self.fc1(x))
        x = self.fc_final(x)
        return x


# def _handle_n_hidden(n_hidden):
#     if type(n_hidden) == int:
#         n_layers = 1
#         hidden_dim = n_hidden
#     elif type(n_hidden) == str:
#         n_hidden = n_hidden.split(',')
#         n_hidden = [int(a) for a in n_hidden]
#         n_layers = len(n_hidden)
#         hidden_dim = int(n_hidden[0])

#     else:
#         raise TypeError('n_hidden should be a string or a int.')

#     return hidden_dim, n_layers

# def _instantiate_class(module_name: str, class_name: str):
#     module = importlib.import_module(module_name)
#     class_ = getattr(module, class_name)
#     return class_()

# class ConvSeqEncoder(torch.nn.Module):
#     """
#     this network architecture is from NeurTraL-AD
#     """
#     def __init__(self, n_features, n_hidden='100', n_output=128, n_layers=3, seq_len=100,
#                  bias=True, batch_norm=True, activation='ReLU'):
#         super(ConvSeqEncoder, self).__init__()

#         n_hidden, _ = _handle_n_hidden(n_hidden)

#         self.bias = bias
#         self.batch_norm = batch_norm
#         self.activation = activation

#         enc = [self._make_layer(n_features, n_hidden, (3,1,1))]
#         in_dim = n_hidden
#         window_size = seq_len
#         for i in range(n_layers - 2):
#             out_dim = n_hidden*2**i
#             enc.append(self._make_layer(in_dim, out_dim, (3,2,1)))
#             in_dim =out_dim
#             window_size = np.floor((window_size+2-3)/2)+1

#         self.enc = torch.nn.Sequential(*enc)
#         self.final_layer = torch.nn.Conv1d(in_dim, n_output, int(window_size), 1, 0)

#     def _make_layer(self, in_dim, out_dim, conv_param):
#         down_sample = None
#         if conv_param is not None:
#             down_sample = torch.nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
#                                           kernel_size=conv_param[0], stride=conv_param[1], padding=conv_param[2],
#                                           bias=self.bias)
#         elif in_dim != out_dim:
#             down_sample = torch.nn.Conv1d(in_channels=in_dim, out_channels=out_dim,
#                                           kernel_size=1, stride=1, padding=0,
#                                           bias=self.bias)

#         layer = ConvResBlock(in_dim, out_dim, conv_param, down_sample=down_sample,
#                              batch_norm=self.batch_norm, bias=self.bias, activation=self.activation)

#         return layer

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         z = self.enc(x)
#         z = self.final_layer(z)
#         return z.squeeze(-1)


# class ConvResBlock(torch.nn.Module):
#     """Convolutional Residual Block"""
#     def __init__(self, in_dim, out_dim, conv_param=None, down_sample=None,
#                  batch_norm=False, bias=False, activation='ReLU'):
#         super(ConvResBlock, self).__init__()

#         self.conv1 = torch.nn.Conv1d(in_dim, in_dim,
#                                      kernel_size=1, stride=1, padding=0, bias=bias)

#         if conv_param is not None:
#             self.conv2 = torch.nn.Conv1d(in_dim, in_dim,
#                                          conv_param[0], conv_param[1], conv_param[2],bias=bias)
#         else:
#             self.conv2 = torch.nn.Conv1d(in_dim, in_dim,
#                                          kernel_size=3, stride=1, padding=1, bias=bias)

#         self.conv3 = torch.nn.Conv1d(in_dim, out_dim,
#                                      kernel_size=1, stride=1, padding=0, bias=bias)

#         if batch_norm:
#             self.bn1 = torch.nn.BatchNorm1d(in_dim)
#             self.bn2 = torch.nn.BatchNorm1d(in_dim)
#             self.bn3 = torch.nn.BatchNorm1d(out_dim)
#             if down_sample:
#                 self.bn4 = torch.nn.BatchNorm1d(out_dim)

#         self.act = _instantiate_class("torch.nn.modules.activation", activation)
#         self.down_sample = down_sample
#         self.batch_norm = batch_norm

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         if self.batch_norm:
#             out = self.bn1(out)
#         out = self.act(out)

#         out = self.conv2(out)
#         if self.batch_norm:
#             out = self.bn2(out)
#         out = self.act(out)

#         out = self.conv3(out)
#         if self.batch_norm:
#             out = self.bn3(out)

#         if self.down_sample is not None:
#             residual = self.down_sample(x)
#             if self.batch_norm:
#                 residual = self.bn4(residual)

#         out += residual
#         out = self.act(out)

#         return out



# #
# if __name__ == '__main__':
#     model = ConvSeqEncoder(n_features=19, n_hidden='512', n_layers=4, seq_len=30, batch_norm=False,
#                            n_output=1, activation='LeakyReLU')
#     print(model)
#     a = torch.randn(32, 30, 19)

#     b =  model(a)
#     print(b.shape)
