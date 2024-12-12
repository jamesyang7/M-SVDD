import os
import math
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import cv2
from torch.utils.data.dataset import Dataset
from audio_process import *
from image_process import *

np.random.seed(42)

class CollisionLoader(Dataset):
    def __init__(self, data_path,frequency=[0,10000],train=True):
        super(CollisionLoader, self).__init__()
        self.class_path         = os.path.join(data_path,'class')
        self.imu_path           = os.path.join(data_path,'imu')
        self.audio_path         = os.path.join(data_path,'audio_numpy')
        self.frequency          = frequency
        self.train  = train
        self.imu_list = []
        self.audio_list = []
        self.class_list = []

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

        audio      = normalization_processing(np.load(self.audio_list[index]))
        spec       = Audio2Spectrogram(audio,sr=48000,min_frequency=self.frequency[0],max_frequency=self.frequency[1],conv_2d=0)
        audio_tensor = torch.from_numpy(audio).float()

        imu   = np.load(self.imu_list[index])
        imu = torch.from_numpy(imu).float()

        cls      = np.load(self.class_list[index])
        cls      = cls[0]
        cls      = torch.tensor(cls)
        # print(spec.shape,imu.shape,audio_tensor.shape,cls)
        return spec, imu, audio_tensor, cls

