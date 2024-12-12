import os
import torch.nn as nn
import numpy as np
from random import shuffle
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
# from preprocess.focal_loss import FocalLoss
# from utils.svdd_loss import SVDDLoss
# from svdd_dataloader_train import CollisionLoader
from svdd_dataloader import CollisionLoader_new  
# from nets.svdd_net import FusionNet
# from baseline_lstm_svdd_net import FusionNet
from baseline_reconstruct import FusionNet
from reconstruction_loss import ReconstructionLoss
from baseline_reconstruct_loss import VAELoss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
train_audio_path = '/home/iot/collision_detect/data/audio/normal_test'
train_imu_path = '/home/iot/collision_detect/data/imu/normal_test'

test_audio_path = '/home/iot/collision_detect/data/audio/abnormal'
test_imu_path = '/home/iot/collision_detect/data/imu/abnormal'
checkpoint_path = '/home/iot/collision_detect/output/reconstruct_baseline/svdd_0_/last_model.pth'
save_path = '/home/iot/collision_detect/output'
workers = 4
batchsize = 64
dropout_rate = 0.3
kernel_num = 32
feature_dim = 512
num_class = 2
use_attention = 0
Epoch = 200
hidden_dim = 64  # LSTM隐藏层维度
num_layers = 2  # LSTM层数
save_name = "svdd_{}_".format(use_attention)
save_dir = os.path.join(save_path, save_name)
os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
reconstruction_loss_fn = ReconstructionLoss()
# loss_function = nn.MSELoss()
loss_function = VAELoss()

train_data = CollisionLoader_new(train_imu_path,train_audio_path)
val_data   = CollisionLoader_new(test_imu_path,test_audio_path)
train_dataloader = DataLoader(train_data, batchsize, shuffle=True, num_workers=workers, drop_last=True)
val_dataloader   = DataLoader(val_data, batchsize, shuffle=True, num_workers=workers, drop_last=True)

random_tensor = torch.load('center.pth').to(device)
print(random_tensor)
model = FusionNet(use_crossattention=use_attention, feature_dim=feature_dim, dropout_rate=dropout_rate, kernel_num=kernel_num, classes=num_class)
model = model.to(device)

if checkpoint_path != '':
    model.load_state_dict(torch.load(checkpoint_path))
model.eval()
sample_idx = 405



# import matplotlib.pyplot as plt

# Initialize lists to store individual loss values
reconstruction_losses = []
svdd_losses = []
total_losses = []
random_tensor = torch.load('center.pth').to(device)
true_labels = []
predicted_labels = []
threshold = 0.45

with torch.no_grad():
    for sample_idx in range(len(val_data)):
        spec, image, audio = val_data[sample_idx]

        spec, image, audio = spec.unsqueeze(0).to(device), image.unsqueeze(0).to(device), audio.unsqueeze(0).to(device)

        # anomaly_score = model(audio, image)
        features, reconstruct_audio, reconstruct_imu, mu, logvar, imu_mu, imu_logvar = model(audio, image)  # 修改这里，使其仅传递audio

        # reconstruction_loss = reconstruction_loss_fn(image, reconstructed_imu, audio, reconstructed_audio) * 10

        target_zero = random_tensor.unsqueeze(0).expand(spec.size(0), -1)
        loss_audio = loss_function(reconstruct_audio,audio,mu,logvar)
        reconimu_resized = reconstruct_imu.view(256,-1)
        loss_imu = loss_function(reconimu_resized,image,imu_mu,imu_logvar)
        reconstruction_loss = loss_audio + loss_imu

        # Calculate SVDD loss for the sample
        # svdd_loss = loss_function(anomaly_score, target_zero)*100

        # Calculate total loss for the sample
        total_loss = reconstruction_loss
        # print(total_loss)
        # print('total_loss is', total_loss)

        # Predict class based on total loss
        if total_loss>10.5:
            predicted_class=1
        else:
            predicted_class=0

        # Handle ground truth class
        true_class = 1 

        # Update confusion matrix variables
        true_labels.append(true_class)
        predicted_labels.append(predicted_class)

        
    for sample_idx in range(len(train_data)):
        spec, image, audio = train_data[sample_idx]

        spec, image, audio = spec.unsqueeze(0).to(device), image.unsqueeze(0).to(device), audio.unsqueeze(0).to(device)
        features, reconstruct_audio, reconstruct_imu, mu, logvar, imu_mu, imu_logvar = model(audio, image)  # 修改这里，使其仅传递audio

        # reconstruction_loss = reconstruction_loss_fn(image, reconstructed_imu, audio, reconstructed_audio) * 10

        target_zero = random_tensor.unsqueeze(0).expand(spec.size(0), -1)
        loss_audio = loss_function(reconstruct_audio,audio,mu,logvar)
        reconimu_resized = reconstruct_imu.view(256,-1)
        loss_imu = loss_function(reconimu_resized,image,imu_mu,imu_logvar)
        reconstruction_loss = loss_audio + loss_imu
        total_loss = reconstruction_loss
        # print(total_loss)
        # print('total_loss is', total_loss)

        # Predict class based on total loss
        if total_loss>10.5:
            predicted_class=1
        else:
            predicted_class=0

        # Handle ground truth class
        true_class = 0

        # Update confusion matrix variables
        true_labels.append(true_class)
        predicted_labels.append(predicted_class)

# Construct confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("Confusion Matrix:")
print(conf_matrix)


class_names = ['Normal', 'Anomaly']

# Calculate accuracy
accuracy = np.trace(conf_matrix) / float(np.sum(conf_matrix))
print(' the accuracy is ', accuracy)

