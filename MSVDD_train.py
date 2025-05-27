import os
import json
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.svdd_dataloader import CollisionLoader_audio as CollisionLoader
from nets.gaussianNet import Trainer, GaussianSVDDModel
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# Load configuration
with open("config/config.json", "r") as f:
    cfg = json.load(f)

# Unpack configuration
data_cfg = cfg["data"]
model_cfg = cfg["model"]
train_cfg = cfg["training"]
runtime_cfg = cfg["runtime"]

# Device setup
device = torch.device(runtime_cfg["device"] if torch.cuda.is_available() else "cpu")

# Directory setup
os.makedirs(runtime_cfg["save_dir"], exist_ok=True)

# Data loaders
train_loader = DataLoader(
    CollisionLoader(data_cfg["train_imu"], data_cfg["train_audio"], augment=False, mask=False),
    batch_size=train_cfg["batch_size"],
    shuffle=True,
    num_workers=runtime_cfg["workers"],
    drop_last=True
)

val_loader = DataLoader(
    CollisionLoader(data_cfg["test_imu"], data_cfg["test_audio"], augment=False, mask=False),
    batch_size=train_cfg["batch_size"],
    shuffle=False,
    num_workers=runtime_cfg["workers"],
    drop_last=True
)

# Model and optimizer
model = GaussianSVDDModel(output_dim=model_cfg["feature_dim"]).to(device)
if model_cfg["checkpoint"]:
    model.load_state_dict(torch.load(model_cfg["checkpoint"]))

optimizer = optim.Adam([
    {"params": [p for n, p in model.named_parameters() if n != "radius"], "lr": train_cfg["learning_rate"]},
    {"params": model.radius, "lr": train_cfg["learning_rate_radius"]}
])

# Training
trainer = Trainer(model, train_loader, optimizer, device, checkpoint_path=runtime_cfg["save_dir"], log_dir=runtime_cfg["save_dir"])
trainer.train(num_epochs=train_cfg["epochs"], log_interval=5)
