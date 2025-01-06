from random import random
import matplotlib
import torch
matplotlib.use('Agg')
import torchaudio.transforms as T
import torchvision.transforms as trans

def normalization_processing(data):
    data_min = data.min()
    data_max = data.max()
    data = data - data_min
    data = data / (data_max-data_min)
    return data

def normalization_processing_torch(data):

    data_min = torch.min(data)
    data_max = torch.max(data)
    normalized_data = (data - data_min) / (data_max - data_min)

    return normalized_data

def normalization_processing_torch_all(data):
    for i in range(data.shape[0]):
        data[i,:] = normalization_processing_torch(data[i,:])
    return data


def Audio2Spectrogram(np_data,sr,num_audio=6,normarlization=0,min_frequency=8000,max_frequency=10000,eliminate=0,conv_2d=0):

    np_data   = torch.tensor(np_data,dtype=torch.float32)

    melspectrogram = T.MelSpectrogram(
        sample_rate = sr,
        n_fft = 2048,
        hop_length=512,
        n_mels=128,
        # f_min=min_frequency,
        # f_max=max_frequency,
        pad_mode='constant',
        norm='slaney',
        mel_scale='slaney',
        power=2,

    )
    spectrogram = melspectrogram(np_data)

    if normarlization!=0:
        spectrogram = normalization_processing_torch_all(spectrogram)

    if conv_2d==0:
        resize_transform = trans.Resize((64,64),antialias=True)
        spectrogram = resize_transform(spectrogram)
    elif conv_2d==1:
        resize_transform = trans.Resize((256,256),antialias=True)
        spectrogram = resize_transform(spectrogram)
    else:
        resize_transform = trans.Resize((224,224),antialias=True)
        spectrogram = resize_transform(spectrogram)

    return spectrogram


def downsample(audio, old_sample_rate=48000, new_sample_rate=4800):
    """
    Downsample audio from old_sample_rate to new_sample_rate.
    """
    ratio = old_sample_rate // new_sample_rate
    downsampled_audio = audio[:, ::ratio]
    return downsampled_audio


class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.state = None

    def filter(self, value):
        if self.state is None:
            self.state = value
        else:
            self.state = self.alpha * value + (1 - self.alpha) * self.state
        return self.state