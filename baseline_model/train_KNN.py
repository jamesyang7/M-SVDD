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
from sklearn.neighbors import KNeighborsClassifier

torch.manual_seed(42)
np.random.seed(42)

data_path = '/home/iot/audio_visual_collision/Data'
checkpoint_path = ''
save_path = '/home/iot/collision_detect/output/original/cnn'
workers = 4
batchsize = 256
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
knn = KNeighborsClassifier(n_neighbors=5)

loss_function = nn.MSELoss()
random_tensor = torch.randn(feature_dim * 2).to(device)
best_val_loss = float('inf')
best_model_state_dict = None

torch.save(random_tensor, 'center.pth')

for epoch in range(Epoch):
    model.train()
    total_train_loss = 0

    train_features = []
    train_labels = []

    for i, data in enumerate(train_dataloader, 0):
        spec, image, audio, cls = data
        spec, image, audio, cls = spec.to(device), image.to(device), audio.to(device), cls.to(device)

        optimizer.zero_grad()
        f_all = model(audio, image)
        
        
        train_features.append(f_all.detach().cpu().numpy())
        train_labels.append(cls.detach().cpu().numpy())

        
        reconstruction_loss = loss_function(f_all, f_all)   #直接看val的
        reconstruction_loss.backward()
        optimizer.step()

        total_train_loss += reconstruction_loss.item()

    # Train KNN with collected features
    train_features = np.vstack(train_features)
    train_labels = np.hstack(train_labels)
    knn.fit(train_features, train_labels)

    mean_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{Epoch}], Train Loss: {mean_train_loss:.4f}")

    if (epoch + 1) % 1 == 0:
        model.eval()
        total_val_loss = 0

        val_features = []
        val_labels = []

        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                spec, image, audio, cls = data
                spec, image, audio, cls = spec.to(device), image.to(device), audio.to(device), cls.to(device)

                f_all = model(audio, image)
                val_features.append(f_all.detach().cpu().numpy())
                val_labels.append(cls.detach().cpu().numpy())

            val_features = np.vstack(val_features)
            val_labels = np.hstack(val_labels)

            # Predict with KNN and compute some metric, e.g., accuracy
            val_predictions = knn.predict(val_features)
            val_accuracy = np.mean(val_predictions == val_labels)
            total_val_loss = 1 - val_accuracy

            mean_val_loss = total_val_loss
            print(f"Epoch [{epoch + 1}/{Epoch}], Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {mean_val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))
