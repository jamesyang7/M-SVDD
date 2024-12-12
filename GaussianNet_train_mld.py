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
from dataloader.svdd_dataloader_mld import get_sub_seqs,get_sub_seqs_label
from utils.reconstruction_loss import ReconstructionLoss, CompactFeatureLoss
from nets.gaussianNet_mld import Trainer, GaussianSVDDModel

torch.manual_seed(42)
np.random.seed(42)

seq_len = 100
stride  = 10
# Paths
train_data = np.load('/home/iot/collision_detect/OmniAnomaly/processed/MSL_train.pkl',allow_pickle=True)
test_data  = np.load('/home/iot/collision_detect/OmniAnomaly/processed/MSL_test.pkl',allow_pickle=True)
test_label = np.load('/home/iot/collision_detect/OmniAnomaly/processed/MSL_test_label.pkl',allow_pickle=True)


checkpoint_path = ''
save_path = '/home/iot/collision_detect/output/open_dataset'
feature_dim = 64
save_name = "Gaussian_{}".format(feature_dim)
save_dir = os.path.join(save_path, save_name)
os.makedirs(save_dir, exist_ok=True)

# Hyperparameters and Settings
workers = 4
batchsize = 64
Epoch = 1000
a, b = 0.1, 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Loaders
train_seqs = get_sub_seqs(train_data, seq_len=seq_len, stride=stride)
train_dataloader = DataLoader(train_seqs, batch_size=batchsize,
                              shuffle=True, pin_memory=True)

# Model and Optimizer
model = GaussianSVDDModel(feature_dim=feature_dim).to(device)
if checkpoint_path != '':
    model.load_state_dict(torch.load(checkpoint_path))


optimizer = optim.Adam([
    {'params':[param for name,param in model.named_parameters() if name!='radius'], 'lr':0.0001 },
    {'params':model.radius, 'lr':0.001 }, 
    ])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

trainer = Trainer(model, train_dataloader, optimizer, device, checkpoint_path=save_dir,log_dir=save_dir)
trainer.train(num_epochs=Epoch)
