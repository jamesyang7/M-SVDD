import os

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from nets.feature_extractor_mld import Encoder,Decoder
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GaussianSVDDModel(nn.Module):
    def __init__(self, feature_dim, confidence=0.95, reg_const=1e-3,is_train=1):
        super(GaussianSVDDModel, self).__init__()
        # Initialize audio and IMU encoders and decoders
        self.is_train = is_train
        self.encoder = Encoder(55,feature_dim)
        self.decoder = Decoder(feature_dim=feature_dim)
        self.confidence = confidence
        self.reg_const =  reg_const
        self.mu = torch.zeros(feature_dim, requires_grad=False)
        self.sigma_inv = torch.eye(feature_dim, requires_grad=False)
        self.radius = nn.Parameter(torch.ones(1))


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

    def forward(self, x,flag=0):
        """
        Perform a forward pass, compute SVDD loss and reconstruction loss.
        :param x_audio: Audio input data (batch_size, input_dim)
        :param x_imu: IMU input data (batch_size, input_dim)
        :return: Total loss combining SVDD and reconstruction losses
        """
        # Encode audio and IMU features
        [ba, ca, feature] = x.size()

        feature,feature_recons = self.encoder(x)
        feature_flat   = feature.view(ba, -1).float()
        z_combined = feature_flat

        # Update mean and covariance matrix inverse
        if self.is_train:
            self.mu = torch.mean(z_combined, dim=0)
            cov_matrix = torch.cov(z_combined.T) + self.reg_const * torch.eye(z_combined.size(1)).to(z_combined.device)
            self.sigma_inv = torch.inverse(cov_matrix)

        # self.sigma_inv = torch.inverse(torch.cov(z_combined.T) + 1e-6 * torch.eye(z_combined.size(1)).to(z_combined.device))
        # Compute Mahalanobis distances
        distances = self.mahalanobis_distance(z_combined)
        # Dynamic radius estimation
        if flag:
            radius = self.dynamic_radius(distances)
            self.radius.data = torch.tensor([radius]).to(device)
            # print(radius,self.mu[:10],self.sigma_inv[0][:10])
        # Decode audio and IMU data

        x_recon  = self.decoder(feature_recons)

        if self.is_train:
            return distances,self.radius,x_recon,z_combined
        else:
            return distances,x_recon,z_combined

    def load_checkpoint(self, checkpoint_path):
        """
        Load the model, mean vector, and covariance matrix inverse from a checkpoint file.
        """
        checkpoint = torch.load(os.path.join(checkpoint_path,"last_weights"))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.mu = checkpoint['mu']
        self.sigma_inv = checkpoint['sigma_inv']
        self.radius = checkpoint['radius']
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
        self.lamda = 0.001

    def train(self, num_epochs, log_interval=5):
        """
        Train the model for a specified number of epochs.
        :param num_epochs: Number of epochs to train the model
        :param log_interval: Interval for logging reconstructed and ground truth data
        """
        self.model.to(self.device)
        # Use Cosine Annealing learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        flag = 1
        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            running_gaussian_loss = 0.0
            running_reconstruction_loss = 0.0
            running_reg_loss = 0.0

            for i, data in enumerate(self.train_loader, 0):
                input_data = data
                input_data = input_data.float().to(self.device)

                self.optimizer.zero_grad()
                # Forward pass
                distances, radius,x_recon,z = self.model(input_data,flag)
                # Gaussian loss
                gaussian_loss = 5*torch.mean(torch.relu(distances**2-radius**2))+radius**2

                # Reconstruction losses
                reconstruction_loss = nn.SmoothL1Loss()(input_data, x_recon)  

                # Scale the reconstruction loss to balance the overall loss
                scaled_reconstruction_loss = 10000 * reconstruction_loss

                entropy_loss = self.compute_entropy_and_covariance_loss(z)
                reg_loss = self.lamda*(entropy_loss)

                # Total loss
                total_loss = gaussian_loss+scaled_reconstruction_loss+reg_loss
                total_loss.backward()
                # Update parameters
                self.optimizer.step()
                # Accumulate losses
                running_loss += total_loss.item() * input_data.size(0)
                running_gaussian_loss += gaussian_loss.item() * input_data.size(0)
                running_reconstruction_loss += scaled_reconstruction_loss.item() * input_data.size(0)
                running_reg_loss +=reg_loss.item()*input_data.size(0)

            # Calculate and print average losses for the epoch
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_gaussian_loss = running_gaussian_loss / len(self.train_loader.dataset)
            epoch_reconstruction_loss = running_reconstruction_loss / len(self.train_loader.dataset)
            epoch_reg_loss = running_reg_loss / len(self.train_loader.dataset)

            print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, Gaussian Loss: {epoch_gaussian_loss:.4f}, Reconstruction Loss: {epoch_reconstruction_loss:.4f}, Reg Loss: {epoch_reg_loss:.4f}, Radius: {radius.item():.4f}, Dis:{torch.mean(distances).detach().cpu().numpy():.4f}')
            flag=0
            # Update the learning rate using the cosine annealing scheduler
            scheduler.step()

            # Log reconstructed and ground truth samples to TensorBoard every `log_interval` epochs
            if (epoch + 1) % log_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    # Take the first batch from the training loader for visualization
                    data = next(iter(self.train_loader))
                    input_data = data
                    input_data = input_data.float().to(self.device)
                    # Forward pass for reconstruction
                    _, _, reconstructed_data,_ = self.model(input_data)

                    # Plot and log to TensorBoard
                    # self.plot_to_tensorboard(
                    #     f'Reconstructed vs Ground Truth Epoch {epoch+1}',
                    #     input_data[0],  # Ground truth audio
                    #     reconstructed_data[0],  # Reconstructed audio
                    #     epoch + 1
                    # )
                self.save_checkpoint()
                # Save the model checkpoint after the last epoch
                # torch.save(self.model.state_dict(),os.path.join(self.checkpoint_path,"last_epoch"))     # Close TensorBoard writer
        self.writer.close()


    def plot_to_tensorboard(self, tag, audio,audio_recons, epoch):
        """
        Plot ground truth and reconstructed audio and IMU data to TensorBoard.
        """
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))

        # Plot audio channel 1 (Ground Truth and Reconstructed)
        axs[0].plot(audio[0].cpu().numpy(), color='blue', label='Ground Truth')
        axs[0].plot(audio_recons[0].cpu().numpy(), color='orange', linestyle='--', label='Reconstructed')
        axs[0].set_title('Reconstructed')
        axs[0].legend()

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
            'sigma_inv': self.model.sigma_inv,  # Save the covariance matrix inverse
            "radius":self.model.radius
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_path,"last_weights"))
        print(f'Model saved to {os.path.join(self.checkpoint_path,"last_weights")}.')

    def compute_entropy_and_covariance_loss(self, z):
        """
        Compute both the entropy loss and the covariance regularization loss.
        """
        # Calculate covariance matrix
        cov_matrix = torch.cov(z.T)
        # Calculate determinant of covariance matrix
        logdet_cov = torch.logdet(cov_matrix + 1e-6 * torch.eye(cov_matrix.size(0)).to(cov_matrix.device))
        # Calculate entropy based on the covariance matrix
        entropy_loss = -0.5 * logdet_cov  # Negative to minimize the entropy
        # Calculate eigenvalues of the covariance matrix
        # eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        # max_eigenvalue = torch.max(eigenvalues)
        # min_eigenvalue = torch.min(eigenvalues)
        # # Calculate condition number (max eigenvalue / min eigenvalue) and regularization loss
        # epsilon = 1e-6  # To avoid division by zero
        # covariance_reg_loss = torch.log(max_eigenvalue / (min_eigenvalue + epsilon))

        return entropy_loss




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
