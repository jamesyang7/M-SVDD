import os
import torch.nn as nn
import numpy as np
from random import shuffle
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from svdd_dataloader_train import CollisionLoader
from baseline_ocsvm_net import FusionNet
# from utils.reconstruction_loss import ReconstructionLoss
from sklearn.svm import OneClassSVM

torch.manual_seed(42)
np.random.seed(42)

data_path = '/home/iot/audio_visual_collision/Data'
checkpoint_path = ''
save_path = '/home/iot/collision_detect/output_new'
workers = 4
batchsize = 256
dropout_rate = 0.3
kernel_num = 32
feature_dim = 512
num_class = 2
use_attention = 1
Epoch = 200
save_name = "ocsvm_{}_".format(use_attention)
save_dir = os.path.join(save_path, save_name)
os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_data = CollisionLoader(data_path, train=True)
val_data = CollisionLoader(data_path, train=False)
train_dataloader = DataLoader(train_data, batchsize, shuffle=True, num_workers=workers, drop_last=True)
val_dataloader = DataLoader(val_data, batchsize, shuffle=True, num_workers=workers, drop_last=True)

model = FusionNet(use_crossattention=use_attention, feature_dim=feature_dim, dropout_rate=dropout_rate, kernel_num=kernel_num, classes=num_class)
model = model.to(device)

if checkpoint_path != '':
    model.load_state_dict(torch.load(checkpoint_path))

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
Oneclass_SVM = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.01)

loss_function = nn.MSELoss()
random_tensor = torch.randn(feature_dim * 2).to(device)
best_val_loss = float('inf')
best_model_state_dict = None

torch.save(random_tensor, 'center.pth')

for epoch in range(Epoch):
    model.train()
    total_train_loss = 0

    for i, data in enumerate(train_dataloader, 0):
        spec, image, audio, cls = data
        spec, image, audio, cls = spec.to(device), image.to(device), audio.to(device), cls.to(device)

        optimizer.zero_grad()
        f_all = model(audio, image)
        
        # Compute the loss within PyTorch domain
        oneclass_loss = torch.tensor(0.0, device=device)
        f_all_np = f_all.detach().cpu().numpy()  # Move to CPU and convert to numpy for OneClassSVM

        Oneclass_SVM.fit(f_all_np)
        decision_function = Oneclass_SVM.decision_function(f_all_np)
        oneclass_loss = torch.tensor(decision_function, dtype=torch.float32, device=device).mean()

        # Ensure the tensor requires gradient
        oneclass_loss.requires_grad = True

        oneclass_loss.backward()
        optimizer.step()

        total_train_loss += oneclass_loss.item()

    mean_train_loss = total_train_loss / len(train_dataloader)

    print(f"Epoch [{epoch + 1}/{Epoch}], Train Loss: {mean_train_loss:.4f}")

    if (epoch + 1) % 1 == 0:
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                spec, image, audio, cls = data
                spec, image, audio, cls = spec.to(device), image.to(device), audio.to(device), cls.to(device)

                f_all = model(audio, image)
                
                # Compute the loss within PyTorch domain
                oneclass_loss = torch.tensor(0.0, device=device)
                f_all_np = f_all.detach().cpu().numpy()  # Move to CPU and convert to numpy for OneClassSVM

                decision_function = Oneclass_SVM.decision_function(f_all_np)
                oneclass_loss = torch.tensor(decision_function, dtype=torch.float32, device=device).mean()

                total_val_loss += oneclass_loss.item()

            mean_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch [{epoch + 1}/{Epoch}], Validation Loss: {mean_val_loss:.4f}")

        torch.save(model.state_dict(), 'last_model.pth')
