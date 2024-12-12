import os

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from nets.feature_extractor import Conv1DFeatureExtractor, DeconvModule, IMU_encoder, IMU_decoder
from nets.eca_attention import eca_layer
from nets.attentionLayer import attentionLayer
from torch.utils.tensorboard import SummaryWriter

class GaussianSVDDModel(nn.Module):
    def __init__(self, feature_dim, confidence=0.95, reg_const=1e-3,is_train=1):
        super(GaussianSVDDModel, self).__init__()
        # Initialize audio and IMU encoders and decoders
        self.is_train = is_train
        self.audio_encoder = Conv1DFeatureExtractor(2, feature_dim)
        self.audio_decoder = DeconvModule()
        self.imu_encoder = IMU_encoder(fc_output_dim=feature_dim)
        self.imu_decoder = IMU_decoder(fc_output_dim=feature_dim)
        # Define the fusion module
        self.cross_atten = attentionLayer(feature_dim, 8, 0.3)
        self.cross_atten2 = attentionLayer(feature_dim*2, 8, 0.3)
        self.eca = eca_layer(channel=1)
        # Loss weights and confidence level for dynamic radius calculation
        self.confidence = confidence
        self.reg_const =  reg_const
        # Mean vector and covariance matrix inverse initialization
        self.mu = torch.zeros(feature_dim*2, requires_grad=False)
        self.sigma_inv = torch.eye(feature_dim*2, requires_grad=False)

        self.fc_audio = nn.Linear(4352,4410)
        self.fc_imu = nn.Linear(400,400)
        self.fc1 = nn.Linear(feature_dim*2, feature_dim*2)

    def mahalanobis_distance(self, z):
        """
        Calculate Mahalanobis distance for each data point in the batch.
        :param z: Feature vectors (batch_size, feature_dim)
        :return: Mahalanobis distances (batch_size,)
        """
        diff = z - self.mu
        dist = torch.sqrt(torch.sum(diff @ self.sigma_inv * diff, dim=1))
        return dist

    def dynamic_radius(self, distances):
        """
        Estimate dynamic radius for the Gaussian sphere based on confidence level.
        :param distances: Mahalanobis distances for normal data points (batch_size,)
        :return: Estimated radius (scalar)
        """
        radius = torch.quantile(distances, self.confidence)
        return radius

    def forward(self, x_audio, x_imu):
        """
        Perform a forward pass, compute SVDD loss and reconstruction loss.
        :param x_audio: Audio input data (batch_size, input_dim)
        :param x_imu: IMU input data (batch_size, input_dim)
        :return: Total loss combining SVDD and reconstruction losses
        """
        # Encode audio and IMU features
        [ba, ca, feature] = x_audio.size()

        audio_feature,recons_feature = self.audio_encoder(x_audio)
        audio_feature_flat = audio_feature.view(ba, -1).float()

        imu_feature,imu_recons = self.imu_encoder(x_imu)
        imu_feature_flat   = imu_feature.view(ba, -1).float()

        # Combine both features for SVDD computation
        fav = self.cross_atten(imu_feature_flat.unsqueeze(1), audio_feature_flat.unsqueeze(1)).squeeze(1)
        fva = self.cross_atten(audio_feature_flat.unsqueeze(1), imu_feature_flat.unsqueeze(1)).squeeze(1)
        f_all = torch.cat([fav, fva], dim=1).squeeze(1)
        z_combined = self.eca(f_all.unsqueeze(1)).squeeze(1)
        z_combined = self.fc1(z_combined)

        # Update mean and covariance matrix inverse
        if self.is_train:
            self.mu = torch.mean(z_combined, dim=0)
            cov_matrix = torch.cov(z_combined.T) + self.reg_const * torch.eye(z_combined.size(1)).to(z_combined.device)
            self.sigma_inv = torch.inverse(cov_matrix)

        # self.sigma_inv = torch.inverse(torch.cov(z_combined.T) + 1e-6 * torch.eye(z_combined.size(1)).to(z_combined.device))
        # Compute Mahalanobis distances
        distances = self.mahalanobis_distance(z_combined)
        # Dynamic radius estimation
        if self.is_train:
            radius = self.dynamic_radius(distances)
            # print(radius,self.mu[:10],self.sigma_inv[0][:10])

        # Decode audio and IMU data
        x_audio_recon  = self.audio_decoder(recons_feature)
        x_audio_recon  = self.fc_audio(x_audio_recon)
        x_imu_recon    = self.imu_decoder(imu_recons)
        x_imu_recon    = self.fc_imu(x_imu_recon)
        if self.is_train:
            return distances,radius,x_audio_recon,x_imu_recon
        else:
            return distances,x_audio_recon,x_imu_recon

    def load_checkpoint(self, checkpoint_path):
        """
        Load the model, mean vector, and covariance matrix inverse from a checkpoint file.
        """
        checkpoint = torch.load(os.path.join(checkpoint_path,"last_weights"))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.mu = checkpoint['mu']
        self.sigma_inv = checkpoint['sigma_inv']
        print(f"Model loaded from {checkpoint_path}.")

class Trainer:
    def __init__(self, model, train_loader, optimizer, device, checkpoint_path='checkpoint.pth', log_dir='logs'):
        """
        Initialize the trainer with model, data loader, optimizer, device, and TensorBoard writer.
        """
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.writer = SummaryWriter(log_dir=log_dir)  # Initialize TensorBoard writer

    def train(self, num_epochs, log_interval=10):
        """
        Train the model for a specified number of epochs.
        :param num_epochs: Number of epochs to train the model
        :param log_interval: Interval for logging reconstructed and ground truth data
        """
        self.model.to(self.device)
        # Use Cosine Annealing learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            running_gaussian_loss = 0.0
            running_reconstruction_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):
                spec, imu, audio, imu_recons, audio_recons = data
                spec, imu, audio, imu_recons, audio_recons = spec.to(self.device), imu.to(self.device), audio.to(self.device), imu_recons.to(self.device), audio_recons.to(self.device)

                self.optimizer.zero_grad()
                # Forward pass
                distances, radius, x_audio_recon, x_imu_recon = self.model(audio, imu)

                # Gaussian loss
                gaussian_loss = 0.2*torch.mean(torch.relu(distances**2-radius**2))+torch.mean(distances**2)

                # Reconstruction losses
                reconstruction_loss_audio = nn.SmoothL1Loss()(audio, x_audio_recon)  # Using Huber Loss
                reconstruction_loss_imu = nn.SmoothL1Loss()(imu, x_imu_recon)  # Using Huber Loss
                reconstruction_loss = ( reconstruction_loss_audio+reconstruction_loss_imu) / 2

                # Scale the reconstruction loss to balance the overall loss
                scaled_reconstruction_loss = 100000 * reconstruction_loss

                # Total loss
                total_loss = gaussian_loss+scaled_reconstruction_loss
                total_loss.backward()

                # Update parameters
                self.optimizer.step()

                # Accumulate losses
                running_loss += total_loss.item() * audio.size(0)
                running_gaussian_loss += gaussian_loss.item() * audio.size(0)
                running_reconstruction_loss += reconstruction_loss.item() * audio.size(0)

            # Calculate and print average losses for the epoch
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_gaussian_loss = running_gaussian_loss / len(self.train_loader.dataset)
            epoch_reconstruction_loss = running_reconstruction_loss / len(self.train_loader.dataset)

            print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, Gaussian Loss: {epoch_gaussian_loss:.4f}, Reconstruction Loss: {epoch_reconstruction_loss:.4f}')

            # Update the learning rate using the cosine annealing scheduler
            scheduler.step()

            # Log reconstructed and ground truth samples to TensorBoard every `log_interval` epochs
            if (epoch + 1) % log_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    # Take the first batch from the training loader for visualization
                    data = next(iter(self.train_loader))
                    spec, imu, audio, imu_recons, audio_recons = data
                    spec, imu, audio, imu_recons, audio_recons = spec.to(self.device), imu.to(self.device), audio.to(self.device), imu_recons.to(self.device), audio_recons.to(self.device)

                    # Forward pass for reconstruction
                    _, _, reconstructed_audio, reconstructed_imu = self.model(audio, imu)

                    # Plot and log to TensorBoard
                    self.plot_to_tensorboard(
                        f'Reconstructed vs Ground Truth Epoch {epoch+1}',
                        audio[0],  # Ground truth audio
                        imu[0],  # Ground truth IMU
                        reconstructed_audio[0],  # Reconstructed audio
                        reconstructed_imu[0],  # Reconstructed IMU
                        epoch + 1
                    )
                self.save_checkpoint()
                # Save the model checkpoint after the last epoch
                # torch.save(self.model.state_dict(),os.path.join(self.checkpoint_path,"last_epoch"))     # Close TensorBoard writer
        self.writer.close()


    def plot_to_tensorboard(self, tag, audio, imu, audio_recons, imu_recons, epoch):
        """
        Plot ground truth and reconstructed audio and IMU data to TensorBoard.
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        # Plot audio channel 1 (Ground Truth and Reconstructed)
        axs[0].plot(audio[0].cpu().numpy(), color='blue', label='Ground Truth')
        axs[0].plot(audio_recons[0].cpu().numpy(), color='orange', linestyle='--', label='Reconstructed')
        axs[0].set_title('Audio - Channel 1')
        axs[0].legend()

        # Plot audio channel 2 (Ground Truth and Reconstructed)
        axs[1].plot(audio[1].cpu().numpy(), color='green', label='Ground Truth')
        axs[1].plot(audio_recons[1].cpu().numpy(), color='orange', linestyle='--', label='Reconstructed')
        axs[1].set_title('Audio - Channel 2')
        axs[1].legend()

        # Plot IMU (Ground Truth and Reconstructed)
        axs[2].plot(imu.cpu().numpy(), color='red', label='Ground Truth')
        axs[2].plot(imu_recons.cpu().numpy(), color='purple', linestyle='--', label='Reconstructed')
        axs[2].set_title('IMU')
        axs[2].legend()

        # Save plot to TensorBoard
        self.writer.add_figure(tag, fig, epoch)
        plt.close(fig)

    def save_checkpoint(self):
        """
        Save the model, mean vector (mu), and covariance matrix inverse (sigma_inv) to a checkpoint file.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'mu': self.model.mu,  # Save the mean vector
            'sigma_inv': self.model.sigma_inv  # Save the covariance matrix inverse
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_path,"last_weights"))
        print(f'Model saved to {os.path.join(self.checkpoint_path,"last_weights")}.')




# def main():
#     # Hyperparameters and setup
#     feature_dim = 128
#     learning_rate = 0.001
#     num_epochs = 20
#     batch_size = 32
#
#     # Model, optimizer, and device setup
#     model = GaussianSVDDModel(feature_dim=feature_dim)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Assume train_loader is defined and provides batches of audio and IMU data
#     # Replace `YourDatasetClass` with the actual dataset class used
#     train_loader = DataLoader(YourDatasetClass(...), batch_size=batch_size, shuffle=True)
#
#     # Initialize and start training
#     trainer = Trainer(model, train_loader, optimizer, device, checkpoint_path='gaussian_svdd_checkpoint.pth')
#     trainer.train(num_epochs=num_epochs)
#
# if __name__ == "__main__":
#     main()
