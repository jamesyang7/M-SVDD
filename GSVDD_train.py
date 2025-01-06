import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.svdd_dataloader import CollisionLoader_audio as CollisionLoader_new
from nets.gaussianNet import Trainer, GaussianSVDDModel
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

train_audio_path = '/home/iot/collision_detect/new_data/audio_np/Normal_train'
train_imu_path = '/home/iot/collision_detect/new_data/imu_np/Normal_train'
test_audio_path = '/home/iot/collision_detect/new_data/audio_np/Normal_test'
test_imu_path = '/home/iot/collision_detect/new_data/imu_np/Normal_test'
checkpoint_path = ''
save_path = './output'
feature_dim = 32
print(f"The feature dim is {feature_dim}")
save_name = "test"
save_dir = os.path.join(save_path, save_name)
os.makedirs(save_dir, exist_ok=True)
workers = 4
batchsize = 32  
Epoch = 50
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

optimizer = optim.Adam([
    {'params':[param for name,param in model.named_parameters() if name!='radius'], 'lr':0.0001 },
    {'params':model.radius, 'lr':0.0001 },
                       ])

trainer = Trainer(model, train_dataloader, optimizer, device, checkpoint_path=save_dir,log_dir=save_dir)
trainer.train(num_epochs=Epoch,log_interval=5)