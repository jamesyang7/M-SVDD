import os.path
import random

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torchaudio.transforms as T
import torchvision.transforms as trans
import cv2



def preprocess_input(image):
    image = image / 127.5-1
    return image


def make_traj(gt_path,name):
    parts = name.split("/")
    index = parts[1][:-4]
    future_traj = np.load(os.path.join(gt_path,name)).reshape(1,3)
    for f in range(1,10):
        file_name   =  f"{parts[0]}/{int(index)+f}.npy"
        current_pos = np.load(os.path.join(gt_path,file_name)).reshape(1,3)
        future_traj = np.concatenate((future_traj,current_pos),0)
    return future_traj

def make_img_seq(image_path,name,image):
    parts = name.split("/")
    index = parts[-1][:-4]
    past_image = image[np.newaxis,...]
    for f in range(1,4):
        padded_index = str(int(index) - f).zfill(4)
        file_name = f"{parts[0]}/{parts[1]}/{padded_index}.png"
        current_image_name = os.path.join(image_path,file_name)
        current_image  = cv2.imread(current_image_name,cv2.IMREAD_COLOR)
        # current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        current_image = cv2.resize(current_image,(224,224))
        current_image = preprocess_input(current_image)
        current_image = current_image[np.newaxis,...]
        past_image = np.concatenate([past_image,current_image],axis=0)
    return past_image
