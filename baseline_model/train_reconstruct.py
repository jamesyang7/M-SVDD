import os
import torch.nn as nn
import numpy as np
from random import shuffle
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
# from svdd_dataloader_train import CollisionLoader
from svdd_dataloader import CollisionLoader_new  
from baseline_reconstruct import FusionNet
from reconstruction_loss import ReconstructionLoss
from baseline_reconstruct_loss import VAELoss
import torch.nn.functional as F
torch.manual_seed(42)
np.random.seed(42)

train_audio_path = '/home/iot/collision_detect/data/audio/normal_train'
train_imu_path = '/home/iot/collision_detect/data/imu/normal_train'

test_audio_path = '/home/iot/collision_detect/data/audio/abnormal'
test_imu_path = '/home/iot/collision_detect/data/imu/abnormal'
checkpoint_path = ''
save_path = '/home/iot/collision_detect/output/reconstruct_baseline'
workers = 4
batchsize = 64
dropout_rate = 0.3
kernel_num = 32
feature_dim = 512
num_class = 2
use_attention = 0
Epoch = 200
save_name = "svdd_{}_".format(use_attention)
save_dir = os.path.join(save_path, save_name)
os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_data = CollisionLoader_new(train_imu_path,train_audio_path)
val_data   = CollisionLoader_new(test_imu_path,test_audio_path)
train_dataloader = DataLoader(train_data, batchsize, shuffle=True, num_workers=workers, drop_last=True)
val_dataloader = DataLoader(val_data, batchsize, shuffle=True, num_workers=workers, drop_last=True)

model = FusionNet(use_crossattention=use_attention, feature_dim=feature_dim, dropout_rate=dropout_rate, kernel_num=kernel_num, classes=num_class)
model = model.to(device)

if checkpoint_path != '':
    model.load_state_dict(torch.load(checkpoint_path))

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

reconstruction_loss_fn = ReconstructionLoss()
loss_function = VAELoss()
# loss_function = nn.MSELoss()
random_tensor = torch.randn(feature_dim * 2).to(device)
best_val_loss = float('inf')
best_model_state_dict = None

torch.save(random_tensor, 'center.pth')
for epoch in range(Epoch):
    model.train()
    total_train_loss = 0
    total_svdd_loss = 0
    total_reconstruction_loss = 0

    for i, data in enumerate(train_dataloader, 0):
        spec, image, audio = data
        spec, image, audio = spec.to(device), image.to(device), audio.to(device)
        # print('image size is ', image.shape)
        
        optimizer.zero_grad()
        features, reconstruct_audio, reconstruct_imu, mu, logvar, imu_mu, imu_logvar = model(audio, image)  # 修改这里，使其仅传递audio
        target_zero = random_tensor.unsqueeze(0).expand(batchsize, -1)
        loss_audio = loss_function(reconstruct_audio,audio,mu,logvar)
        # image_resized = F.interpolate(image.unsqueeze(1), size=reconstruct_imu.shape[2:], mode='nearest')
        # print('image size is ', image.shape)
        reconimu_resized = reconstruct_imu.view(256,-1)
        # print('recon_imu is ',reconimu_resized.shape)

        # print('--------',image_resized.shape)
        # image_resized = image_resized.view(reconstruct_imu.size())
        # print('--------',image_resized.shape)
        loss_imu = loss_function(reconimu_resized,image,imu_mu,imu_logvar)
        reconstruction_loss = loss_audio + loss_imu

        # svdd_loss = loss_function(features, target_zero)
        # total_svdd_loss += svdd_loss.item()

        # reconstruction_loss = reconstruction_loss_fn(image, features)  # 修改这里，使用features
        # total_reconstruction_loss += reconstruction_loss.item()

        total_loss =  reconstruction_loss
        total_loss.backward()
        optimizer.step()

        total_train_loss += total_loss.item()

    mean_train_loss = total_train_loss / len(train_dataloader)
    # mean_svdd_loss = total_svdd_loss / len(train_dataloader)
    # mean_reconstruction_loss = total_reconstruction_loss / len(train_dataloader)

    print(f"Epoch [{epoch+1}/{Epoch}], Train Loss: {mean_train_loss:.4f}")

    if (epoch + 1) % 1 == 0:
        model.eval()
        total_val_loss = 0
        total_val_svdd_loss = 0
        total_val_reconstruction_loss = 0

        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                spec, image, audio = data
                spec, image, audio = spec.to(device), image.to(device), audio.to(device)

                features, reconstruct_audio, reconstruct_imu, mu, logvar, imu_mu, imu_logvar = model(audio, image)  # 修改这里，使其仅传递audio
                target_zero = random_tensor.unsqueeze(0).expand(batchsize, -1)
                loss_audio = loss_function(reconstruct_audio,audio,mu,logvar)
                # image_resized = F.interpolate(image.unsqueeze(1), size=reconstruct_imu.shape[2:], mode='nearest')
                # image_resized = image_resized.squeeze(1)
                # image_resized = image_resized.view(reconstruct_imu.size())
                reconimu_resized = reconstruct_imu.view(256,-1)
                # print('recon_imu is ',reconimu_resized.shape)


                loss_imu = loss_function(reconimu_resized,image,imu_mu,imu_logvar)
                reconstruction_loss = loss_audio + loss_imu

                # features, anomaly_label, svdd_out = model(audio,image)  # 修改这里，使其仅传递audio

                # target_zero = random_tensor.unsqueeze(0).expand(batchsize, -1)
                # svdd_loss = loss_function(features, target_zero)
                total_val_svdd_loss += reconstruction_loss.item()

                # reconstruction_loss = reconstruction_loss_fn(image, features)  # 修改这里，使用features
                # total_val_reconstruction_loss += reconstruction_loss.item()

                # total_loss = reconstruction_loss + svdd_loss
                # total_val_loss += total_loss.item()

        mean_val_loss = total_val_loss / len(val_dataloader)
        # mean_val_svdd_loss = total_val_svdd_loss / len(val_dataloader)
        # mean_val_reconstruction_loss = total_val_reconstruction_loss / len(val_dataloader)

        print(f"Epoch [{epoch+1}/{Epoch}], Validation Loss: {mean_val_loss:.4f}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, os.path.join(save_dir, 'best_model.pth'))

        torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))
