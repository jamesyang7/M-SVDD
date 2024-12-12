import os
from random import random
import librosa
import matplotlib.pyplot as plt
import matplotlib
import torch
matplotlib.use('Agg')
from scipy import signal
import numpy as np
from scipy.signal import butter, lfilter
import torchaudio.transforms as T
import torchvision.transforms as trans

def normalization_processing(data):

    data_min = data.min()
    data_max = data.max()

    data = data - data_min
    data = data / (data_max-data_min)

    return data

def normalization_processing_torch(data):
    # Assuming data is a PyTorch tensor
    data_min = torch.min(data)
    data_max = torch.max(data)

    # Normalizing the data
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
        n_mels=20,
        # f_min=min_frequency,
        # f_max=max_frequency,
        pad_mode='constant',
        norm='slaney',
        mel_scale='slaney',
        power=2,

    )
    spectrogram = melspectrogram(np_data)

    if normarlization!=0:
        # spectrogram = spectrogram/np.linalg.norm(spectrogram,axis=0,keepdims=True)
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

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    # 设计巴特沃斯带通滤波器
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    # 应用滤波器
    filtered_data = lfilter(b, a, data)
    return filtered_data


def make_seq_audio(audio_path,name):
    parts = name.split("/")
    index = parts[-1][:-4]
    past_audio = np.load(os.path.join(audio_path,name))
    for f in range(1,3):
        padded_index = str(int(index) - f).zfill(4)
        file_name = f"{parts[0]}/{parts[1]}/{padded_index}.npy"
        current_pos = np.load(os.path.join(audio_path,file_name))
        past_audio = np.concatenate((past_audio,current_pos),1)
    return past_audio

def exponential_smoothing(data, alpha):
    smoothed_data = [data[0]]
    for i in range(1, len(data)):
        smoothed_value = alpha * data[i] + (1 - alpha) * smoothed_data[-1]
        smoothed_data.append(smoothed_value)
    return smoothed_data

def make_seq_imu(audio_path,name):
    parts = name.split("/")
    index = parts[-1][:-4]
    past_audio = np.load(os.path.join(audio_path,name))
    data_all = []
    data_all.append(past_audio)
    for f in range(1,3):
        padded_index = str(int(index) - f).zfill(4)
        file_name = f"{parts[0]}/{parts[1]}/{padded_index}.npy"
        current_pos = np.load(os.path.join(audio_path,file_name))
        current_pos = exponential_smoothing(current_pos,0.2)
        data_all.append(current_pos)
    data_all = np.stack(data_all,axis=0)
    return data_all


def make_few_shot(anotation_lines,cls_dir,Nshot):
    list_all = []
    for i in range(5):
        num = 0
        for anotation in anotation_lines:
            class_path = os.path.join(cls_dir,anotation[:-1])
            class_numpy = np.load(class_path)[0]
            if class_numpy==i and num<Nshot:
                list_all.append((anotation))
                num+=1
    return list_all

def make_few_shot_collision(anotation_lines,cls_dir,Nshot):
    list_all = []
    num = 0
    for anotation in anotation_lines:
        class_path = os.path.join(cls_dir,anotation[:-1])
        class_numpy = np.load(class_path)[0]
        if class_numpy==0 and num<50:
            list_all.append((anotation))
            num+=1
    num=0
    for anotation in anotation_lines:
        class_path = os.path.join(cls_dir,anotation[:-1])
        class_numpy = np.load(class_path)[0]
        if class_numpy!=0 and num<Nshot:
            list_all.append((anotation))
            num+=1
    return list_all

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