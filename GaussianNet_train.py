import os
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
import matplotlib.pyplot as plt
from nets.svdd_net import FusionNet
from dataloader.svdd_dataloader import CollisionLoader_audio as CollisionLoader_new
from utils.reconstruction_loss import ReconstructionLoss, CompactFeatureLoss
from nets.gaussianNet import Trainer, GaussianSVDDModel

torch.manual_seed(42)
np.random.seed(42)

# Paths
train_audio_path = '/home/iot/collision_detect/new_data/audio_np/Normal_train'
train_imu_path = '/home/iot/collision_detect/new_data/imu_np/Normal_train'
test_audio_path = '/home/iot/collision_detect/new_data/audio_np/Normal_test'
test_imu_path = '/home/iot/collision_detect/new_data/imu_np/Normal_test'

checkpoint_path = ''
save_path = '/home/iot/GSVDD/output'
feature_dim = 2
print(f"The feature dim is {feature_dim}")
# save_name = "Gaussian_{}".format(feature_dim)
save_name = "test"
save_dir = os.path.join(save_path, save_name)
os.makedirs(save_dir, exist_ok=True)

# Hyperparameters and Settings
workers = 4
batchsize = 256  #32
Epoch = 50
a, b = 0.1, 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Loaders
train_data = CollisionLoader_new(train_imu_path, train_audio_path, augment=False, mask=False)
val_data = CollisionLoader_new(test_imu_path, test_audio_path, augment=False, mask=False)
train_dataloader = DataLoader(train_data, batchsize, shuffle=True, num_workers=workers, drop_last=True)
val_dataloader = DataLoader(val_data, batchsize, shuffle=True, num_workers=workers, drop_last=True)

# Model and Optimizer
model = GaussianSVDDModel(output_dim=feature_dim).to(device)
if checkpoint_path != '':
    model.load_state_dict(torch.load(checkpoint_path))

# optimizer = optim.Adam(model.parameters(), lr=0.00005, betas=(0.9, 0.999))

optimizer = optim.Adam([
    {'params':[param for name,param in model.named_parameters() if name!='radius'], 'lr':0.0001 },
    {'params':model.radius, 'lr':0.0005 },
                       ])

# optimizer = optim.Adam([
# #     {'params':[param for name,param in model.named_parameters() if name!='radius'], 'lr':0.0001 },
#     {'params':model.radius, 'lr':0.0001 },
#     {'params':model.audio_decoder.parameters(), 'lr':0.001,'weight_decay': 1e-4 },
#     {'params':model.imu_decoder.parameters(), 'lr':0.0001},
#                        ],lr=0.0001)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

# optimizer = optim.Adam([
#     # Audio decoder parameters with a specific learning rate and weight decay
#     {'params': model.audio_decoder.parameters(), 'lr': 0.0001},
    
#     # IMU decoder parameters with a specific learning rate (and no weight decay)
#     {'params': model.imu_decoder.parameters(), 'lr': 0.0001},
    
#     # All other parameters with default learning rate and weight decay
#     {'params': [param for name, param in model.named_parameters() if 'audio_decoder' not in name and 'imu_decoder' not in name]},
# ], lr=0.0001) 

trainer = Trainer(model, train_dataloader, optimizer, device, checkpoint_path=save_dir,log_dir=save_dir)
trainer.train(num_epochs=Epoch,log_interval=5)

# # Loss Functions
# reconstruction_loss_fn = ReconstructionLoss()
#
# # TensorBoard Summary Writer
# writer = SummaryWriter(log_dir=save_dir)  # Initialize TensorBoard writer
#
# # Function to plot and save reconstructed audio and IMU
# def plot_to_tensorboard(writer, tag, audio, imu, audio_recons, imu_recons, epoch):
#     fig, axs = plt.subplots(3, 1, figsize=(10, 10))
#
#     # Plot audio channel 1 (Ground Truth and Reconstructed)
#     axs[0].plot(audio[0].cpu().numpy(), color='blue', label='Ground Truth')
#     axs[0].plot(audio_recons[0].cpu().numpy(), color='orange', linestyle='--', label='Reconstructed')
#     axs[0].set_title('Audio - Channel 1')
#     axs[0].legend()
#
#     # Plot audio channel 2 (Ground Truth and Reconstructed)
#     axs[1].plot(audio[1].cpu().numpy(), color='green', label='Ground Truth')
#     axs[1].plot(audio_recons[1].cpu().numpy(), color='orange', linestyle='--', label='Reconstructed')
#     axs[1].set_title('Audio - Channel 2')
#     axs[1].legend()
#
#     # Plot IMU (Ground Truth and Reconstructed)
#     axs[2].plot(imu.cpu().numpy(), color='red', label='Ground Truth')
#     axs[2].plot(imu_recons.cpu().numpy(), color='purple', linestyle='--', label='Reconstructed')
#     axs[2].set_title('IMU')
#     axs[2].legend()
#
#     # Save plot to TensorBoard
#     writer.add_figure(tag, fig, epoch)
#     plt.close(fig)
#
#
# # Training Loop
# for epoch in range(Epoch):
#     model.train()
#     total_train_loss = 0
#     total_svdd_loss = 0
#     total_reconstruction_loss = 0
#     current_lr = scheduler.get_last_lr()[0]
#     print(f"-------------------------Epoch [{epoch+1}/{Epoch}], Current Learning Rate: {current_lr:.6f}------------------------------------")
#
#     for i, data in enumerate(train_dataloader, 0):
#         spec, imu, audio, imu_recons, audio_recons = data
#         spec, imu, audio, imu_recons, audio_recons = spec.to(device), imu.to(device), audio.to(device), imu_recons.to(device), audio_recons.to(device)
#         optimizer.zero_grad()
#
#         # Forward pass
#         distances, radius, reconstructed_audio, reconstructed_imu = model(audio, imu)
#
#         # Calculate Loss
#         svdd_loss = torch.mean((distances - radius) ** 2)
#         reconstruction_loss = reconstruction_loss_fn(imu_recons, reconstructed_imu, audio_recons, reconstructed_audio)
#         total_loss = b * reconstruction_loss + a * svdd_loss
#
#         # Backpropagation and Optimization
#         total_loss.backward()
#         optimizer.step()
#
#         # Logging
#         total_train_loss += total_loss.item()
#         total_svdd_loss += svdd_loss.item()
#         total_reconstruction_loss += reconstruction_loss.item()
#
#     scheduler.step()
#     mean_train_loss = total_train_loss / len(train_dataloader)
#     mean_svdd_loss = total_svdd_loss / len(train_dataloader)
#     mean_reconstruction_loss = total_reconstruction_loss / len(train_dataloader)
#
#     print(f"Epoch [{epoch+1}/{Epoch}], Train Loss: {mean_train_loss:.4f}, SVDD Loss: {mean_svdd_loss:.4f}, Reconstruction Loss: {mean_reconstruction_loss:.4f}")
#
#     # Validation and TensorBoard Logging
#     if (epoch + 1) % 5 == 0:
#         model.eval()
#         total_val_loss = 0
#         total_val_svdd_loss = 0
#         total_val_reconstruction_loss = 0
#
#         with torch.no_grad():
#             for i, data in enumerate(val_dataloader, 0):
#                 spec, imu, audio, imu_recons, audio_recons = data
#                 spec, imu, audio, imu_recons, audio_recons = spec.to(device), imu.to(device), audio.to(device), imu_recons.to(device), audio_recons.to(device)
#
#                 distances, radius, reconstructed_audio, reconstructed_imu = model(audio, imu)
#
#                 svdd_loss = torch.mean((distances - radius) ** 2)
#                 reconstruction_loss = reconstruction_loss_fn(imu_recons, reconstructed_imu, audio_recons, reconstructed_audio)
#                 total_loss = b * reconstruction_loss + a * svdd_loss
#
#                 total_val_loss += total_loss.item()
#                 total_val_svdd_loss += svdd_loss.item()
#                 total_val_reconstruction_loss += reconstruction_loss.item()
#
#                 # Log reconstructed samples to TensorBoard
#                 if i == 0:  # Log only the first batch to avoid flooding TensorBoard
#                     plot_to_tensorboard(
#                         writer,
#                         f'Reconstructed vs Ground Truth Epoch {epoch+1}',
#                         audio[0],  # Ground truth audio
#                         imu[0],  # Ground truth IMU
#                         reconstructed_audio[0],  # Reconstructed audio
#                         reconstructed_imu[0],  # Reconstructed IMU
#                         epoch + 1
#                     )
#
#         mean_val_loss = total_val_loss / len(val_dataloader)
#         mean_val_svdd_loss = total_val_svdd_loss / len(val_dataloader)
#         mean_val_reconstruction_loss = total_val_reconstruction_loss / len(val_dataloader)
#
#         print(f"\033[94mEpoch [{epoch+1}/{Epoch}], Val Loss: {mean_val_loss:.4f}, Val SVDD Loss: {mean_val_svdd_loss:.4f}, Val Reconstruction Loss: {mean_val_reconstruction_loss:.4f}\033[0m")
#
#         # Save the model
#         torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))
#
# # Close TensorBoard writer
# writer.close()
