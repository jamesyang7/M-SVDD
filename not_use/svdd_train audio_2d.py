import os
import torch.nn as nn
import numpy as np
from random import shuffle
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader.svdd_dataloader import CollisionLoader_audio as CollisionLoader_new
from nets.svdd_net_audio import FusionNet_audio_2d as FusionNet
from utils.reconstruction_loss import ReconstructionLoss,CompactFeatureLoss,ReconstructionLoss_audio
from utils.svdd_loss import DSVDDLoss
from sklearn import svm
import joblib



torch.manual_seed(42)
np.random.seed(42)

train_audio_path = '/home/iot/collision_detect/data/audio/normal_train'
train_imu_path = '/home/iot/collision_detect/data/imu/normal_train'

test_audio_path = '/home/iot/collision_detect/data/audio/abnormal'
test_imu_path = '/home/iot/collision_detect/data/imu/abnormal'

checkpoint_path = ''
save_path = '/home/iot/collision_detect/output'

workers = 4
twod=1
batchsize = 64
dropout_rate = 0.3
kernel_num = 32
feature_dim = 512
num_class = 2
use_attention = 1
Epoch = 200
save_name = "ours_fusion_{}".format(use_attention)
save_dir = os.path.join(save_path, save_name)
os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
svdd_losses = []
reconstruction_losses = []
train_data = CollisionLoader_new(train_imu_path,train_audio_path,augment =True,twod=1)
val_data   = CollisionLoader_new(test_imu_path,test_audio_path,augment = True,twod=1)

audio = val_data.audio_list
imu  = val_data.imu_list

train_dataloader = DataLoader(train_data, batchsize, shuffle=True, num_workers=workers, drop_last=True)
val_dataloader   = DataLoader(val_data, batchsize, shuffle=True, num_workers=workers, drop_last=True)


model = FusionNet(use_crossattention=use_attention, feature_dim=feature_dim, dropout_rate=dropout_rate, kernel_num=kernel_num, classes=num_class)
model = model.to(device)

if checkpoint_path != '':
    model.load_state_dict(torch.load(checkpoint_path))

optimizer  = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)


reconstruction_loss_fn = ReconstructionLoss_audio()
feature_loss = CompactFeatureLoss()
loss_function = nn.MSELoss()
random_tensor = torch.randn(feature_dim).to(device)

best_val_loss = float('inf')
best_model_state_dict = None

torch.save(random_tensor, 'center.pth')
print(random_tensor)
# Dsvdd = DSVDDLoss(random_tensor)

for epoch in range(Epoch):
    model.train()
    total_train_loss = 0
    total_svdd_loss = 0
    total_reconstruction_loss = 0
    current_lr = scheduler.get_last_lr()[0]
    print(f"-------------------------Epoch [{epoch+1}/{Epoch}], Current Learning Rate: {current_lr:.6f}------------------------------------")
    for i, data in enumerate(train_dataloader, 0):
        spec, image, audio = data
        spec, image, audio = spec.to(device), image.to(device), audio.to(device)

        optimizer.zero_grad()

        anomaly_score, reconstructed_audio = model(audio, image)

        all_feature_loss = feature_loss(anomaly_score)

        target_zero = random_tensor.unsqueeze(0).expand(batchsize, -1)

        svdd_loss = loss_function(anomaly_score, target_zero)
        # svdd_loss = Dsvdd(anomaly_score)

        total_svdd_loss += svdd_loss.item()
        svdd_losses.append(svdd_loss.item())

        reconstruction_loss = reconstruction_loss_fn(audio.view(batchsize,-1), reconstructed_audio)
        total_reconstruction_loss += reconstruction_loss.item()
        reconstruction_losses.append(reconstruction_loss.item())

        total_loss = reconstruction_loss+svdd_loss
        total_loss.backward()
        optimizer.step()

        total_train_loss += total_loss.item()

    scheduler.step()
    mean_train_loss = total_train_loss / len(train_dataloader)
    mean_svdd_loss = total_svdd_loss / len(train_dataloader)
    mean_reconstruction_loss = total_reconstruction_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{Epoch}], Train Loss: {mean_train_loss:.4f}, SVDD Loss: {mean_svdd_loss:.4f}, Reconstruction Loss: {mean_reconstruction_loss:.4f}, feature Loss: {all_feature_loss:.4f}")

    if (epoch + 1) % 1 == 0:
        model.eval()
        total_val_loss = 0
        total_val_svdd_loss = 0
        total_val_reconstruction_loss = 0

        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                spec, image, audio = data
                spec, image, audio = spec.to(device), image.to(device), audio.to(device)

                anomaly_score, reconstructed_audio = model(audio, image)
                # print(' recon_imu is ',reconstructed_imu.shape)

                all_feature_loss = feature_loss(anomaly_score)
                target_zero = random_tensor.unsqueeze(0).expand(batchsize, -1)
                svdd_loss = loss_function(anomaly_score, target_zero)
                # svdd_loss = Dsvdd(anomaly_score)
                total_val_svdd_loss += svdd_loss.item()

                # reconstruction_loss = reconstruction_loss_fn(audio, reconstructed_audio)
                reconstruction_loss = reconstruction_loss_fn(audio.view(batchsize,-1), reconstructed_audio)
                total_val_reconstruction_loss += reconstruction_loss.item()

                total_loss = reconstruction_loss+svdd_loss+all_feature_loss
                total_val_loss += total_loss.item()

        mean_val_loss = total_val_loss / len(val_dataloader)
        mean_val_svdd_loss = total_val_svdd_loss / len(val_dataloader)
        mean_val_reconstruction_loss = total_val_reconstruction_loss / len(val_dataloader)

    print(f"\033[94mEpoch [{epoch+1}/{Epoch}], Val Loss: {mean_val_loss:.4f}, Val SVDD Loss: {mean_val_svdd_loss:.4f}, Val Reconstruction Loss: {mean_val_reconstruction_loss:.4f}, feature Loss: {all_feature_loss:.4f}\033[0m")

    torch.save(model.state_dict(), os.path.join(save_dir,'last_model.pth'))
