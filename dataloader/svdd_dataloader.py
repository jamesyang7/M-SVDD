import os
import math
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import cv2
from torch.utils.data.dataset import Dataset
from preprocess.audio_process import *
from preprocess.image_process import *
import random
from .utlis import *
np.random.seed(42)

class CollisionLoader(Dataset):
    def __init__(self, data_path,frequency=[0,10000],train=True,augment = False):
        super(CollisionLoader, self).__init__()
        self.class_path         = os.path.join(data_path,'class')
        self.imu_path           = os.path.join(data_path,'imu')
        self.audio_path         = os.path.join(data_path,'audio_numpy')
        self.frequency          = frequency
        self.train  = train
        self.augment = augment
        self.imu_list = []
        self.audio_list = []
        self.class_list = []
        self.lpf = LowPassFilter(alpha=0.1)

        subject_dirs = os.listdir(self.audio_path)
        for subject in subject_dirs:
            subject_path = os.path.join(self.audio_path, subject)
            seq_list = os.listdir(subject_path)
            seq_list.sort()
            seq_len = len(seq_list)

            if self.train:
                end = int(seq_len*0.1)
            else:
                end = 0

            a = 0
            for seq in seq_list[end:]:
                seq_path = os.path.join(self.audio_path, subject, seq)
                data_list = os.listdir(seq_path)
                data_list.sort()
                for data in data_list[:]:
                    class_path = os.path.join(self.class_path, subject, seq, data)
                    imu_path = os.path.join(self.imu_path, subject, seq, data)
                    audio_path = os.path.join(self.audio_path, subject, seq, data)
                    
                    if os.path.exists(class_path) and os.path.exists(imu_path) and os.path.exists(audio_path):
                        if self.train:
                            class_data = np.load(class_path)
                            if class_data == 0:
                                self.class_list.append(class_path)
                                self.imu_list.append(imu_path)
                                self.audio_list.append(audio_path)
                        else:
                            class_data = np.load(class_path)
                            if np.load(imu_path).shape[0]==20:
                                if class_data != 0:
                                        self.class_list.append(class_path)
                                        self.imu_list.append(imu_path)
                                        self.audio_list.append(audio_path)
                                else:
                                    if a<=int(seq_len*0.1):
                                        self.class_list.append(class_path)
                                        self.imu_list.append(imu_path)
                                        self.audio_list.append(audio_path)
                a+=1


    def __len__(self):
        return len(self.class_list)

    def __getitem__(self, index):

        audio      = np.load(self.audio_list[index])
        spec       = Audio2Spectrogram(audio,sr=48000,min_frequency=self.frequency[0],max_frequency=self.frequency[1],conv_2d=0)
        audio_tensor = torch.from_numpy(audio).float()

        imu        = LowPassFilter(np.load(self.imu_list[index]))
        imu        = torch.from_numpy(imu).float()

        cls      = np.load(self.class_list[index])
        cls      = cls[0]
        cls      = torch.tensor(cls)
        # print(spec.shape,imu.shape,audio_tensor.shape,cls)
        return spec, imu, audio_tensor, cls


class CollisionLoader_new(Dataset):
    def __init__(self,imu_path ,audio_path ,frequency=[0,10000],train=True,augment = False):
        super(CollisionLoader_new, self).__init__()
        self.imu_path           = imu_path
        self.audio_path         = audio_path 
        self.frequency          = frequency
        self.augment = augment
        self.imu_list = []
        self.audio_list = []
        self.lpf = LowPassFilter(alpha=0.2)
        subject_dirs = os.listdir(self.imu_path)
        for subject in subject_dirs:
            subject_path = os.path.join(self.imu_path, subject)
            seq_list = os.listdir(subject_path)
            seq_list.sort()
        
            for seq in seq_list:
                seq_path = os.path.join(self.imu_path, subject, seq)
                self.imu_list.append(seq_path)
                self.audio_list.append(os.path.join(self.audio_path, subject, seq))


    def __len__(self):
        return len(self.imu_list)


    def __getitem__(self, index):
        audio      = np.load(self.audio_list[index])
        audio      = downsample(audio)
        # audio = normalization_processing(audio)
        if self.augment and np.random.rand() < 0.2:  
            pulse_signal = self.pulse_audio()
            audio += pulse_signal
        elif self.augment and np.random.rand() > 0.95:
            continuous_audio = self.continuous_audio()
            audio += continuous_audio


        spec       = Audio2Spectrogram(audio,sr=48000,min_frequency=self.frequency[0],max_frequency=self.frequency[1],conv_2d=0)
        audio_tensor = torch.from_numpy(audio).float()
        

        imu   = np.load(self.imu_list[index])[:,-1]
        # imu   = np.array([self.lpf.filter(value) for value in imu])
        imu = normalization_processing(imu)
        imu = imu
        
        if self.augment and np.random.rand() < 0.2:  
            pulse_signal = self.pulse_imu()
            imu += pulse_signal

        elif self.augment and np.random.rand() > 0.95:
            continuous_imu = self.continuous_imu()
            imu += continuous_imu

        imu = torch.from_numpy(imu).float()
        
        
        if self.augment:
            augmented_spec, augemnted_imu, augmented_audio_tensor = self.augment_data(spec, imu, audio_tensor)
            return augmented_spec, augemnted_imu, augmented_audio_tensor
        

        # audio_tensor = torch.mean(audio_tensor,dim=0)
        
        return spec, imu, audio_tensor


class CollisionLoader_audio(Dataset):
    def __init__(self,imu_path ,audio_path ,frequency=[0,8000],train=True,augment = False,mask = False,twod=0):
        super(CollisionLoader_audio, self).__init__()
        self.imu_path           = imu_path
        self.audio_path         = audio_path
        self.frequency          = frequency
        self.twod               = twod
        self.augment = augment
        self.mask    = mask
        self.imu_list = []
        self.audio_list = []
        self.lpf = LowPassFilter(alpha=0.1)
        subject_dirs = os.listdir(self.imu_path)
        for subject in subject_dirs:
            subject_path = os.path.join(self.imu_path, subject)
            seq_list = os.listdir(subject_path)
            seq_list.sort()

            for seq in seq_list:
                seq_path = os.path.join(self.imu_path, subject, seq)
                self.imu_list.append(seq_path)
                self.audio_list.append(os.path.join(self.audio_path, subject, seq))

    def __len__(self):
        return len(self.imu_list)

    def __getitem__(self, index):

        audio      = np.load(self.audio_list[index])
        audio      = downsample(audio,old_sample_rate=88200, new_sample_rate=4410)
        # audio      = downsample(audio,old_sample_rate=88200, new_sample_rate=2048)
        # audio = normalization_processing(audio)
        if self.augment:
            rand_num = random.random()
            if rand_num > 0.7:
                overlap_index = self.randomly_select_index(index)
                overlap_audio = np.load(self.audio_list[overlap_index])
                overlap_audio = downsample(overlap_audio,old_sample_rate=88200, new_sample_rate=4410)
                audio = (audio + overlap_audio) / 2
            else:
                audio = audio

        if self.twod==1:
            audio      = np.mean(audio,axis=0)
            audio      = np.reshape(audio,(48,100))

        imu   = np.load(self.imu_list[index])
        imu = np.transpose(imu,[1,0])

        rand_num_mask = random.random()
        if self.mask and rand_num_mask>=0.6:
            audio_input,audio_reconstruct = apply_mask(audio)
            imu_input,imu_reconstruct     = apply_mask(imu)
            imu_input,imu_reconstruct = imu_input[-1,:],imu_reconstruct[-1,:]
        else:
            imu_input,imu_reconstruct    = imu[-1,:],imu[-1,:]
            audio_input,audio_reconstruct = audio,audio

        imu_input   = np.array([self.lpf.filter(value) for value in imu_input])
        # imu_input = normalization_processing(imu_input)
        current_mean = np.mean(imu_input)
        target_mean = 1
        difference = current_mean - target_mean
        imu_input = imu_input - difference

        spec       = Audio2Spectrogram(audio,sr=44100,min_frequency=self.frequency[0],max_frequency=self.frequency[1],conv_2d=0)


        imu,imu_reconstruct = torch.from_numpy(imu_input).float(),torch.from_numpy(imu_reconstruct).float()
        audio_tensor,audio_reconstruct = torch.from_numpy(audio_input).float(),torch.from_numpy(audio_reconstruct).float()

        return spec, imu, audio_tensor,imu_reconstruct,audio_reconstruct

    def randomly_select_index(self, exclude_index):
        indices = list(range(len(self.audio_list)))
        indices.remove(exclude_index)
        return np.random.choice(indices)
