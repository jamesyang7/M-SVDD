import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from nets.feature_extractor import Conv1DFeatureExtractor, DeconvModule, IMU_encoder, IMU_decoder
from nets.eca_attention import eca_layer
from nets.attentionLayer import attentionLayer
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.linalg.cholesky(a + torch.eye(a.size(0)) * 0.001)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

class DAGMM(nn.Module):
    def __init__(self, feature_dim=32, confidence=0.95, reg_const=1e-3,n_gmm=4):
        super(DAGMM, self).__init__()
        self.audio_encoder = Conv1DFeatureExtractor(2, feature_dim)
        self.audio_decoder = DeconvModule()
        self.imu_encoder = IMU_encoder(fc_output_dim=feature_dim)
        self.imu_decoder = IMU_decoder(fc_output_dim=feature_dim)
        # Define the fusion module
        self.cross_atten = attentionLayer(feature_dim, 8, 0.3)
        self.eca = eca_layer(channel=1)
        # Loss weights and confidence level for dynamic radius calculation
        self.confidence = confidence
        self.reg_const =  reg_const
        # Mean vector and covariance matrix inverse initialization
        # self.mu = torch.zeros(feature_dim, requires_grad=False)
        # self.sigma_inv = torch.eye(feature_dim, requires_grad=False)
        # self.radius = nn.Parameter(torch.ones(1))

        self.fc_audio = nn.Linear(4352,4410)
        self.fc_imu = nn.Linear(400,400)
        self.fc1 = nn.Linear(feature_dim, feature_dim)

        layers = []
        layers += [nn.Linear(feature_dim,16)]
        layers += [nn.Tanh()]            
        layers += [nn.Linear(16,n_gmm)]
        layers += [nn.Softmax(dim=1)]


        self.estimation = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm,feature_dim))
        self.register_buffer("cov", torch.zeros(n_gmm,feature_dim,feature_dim))


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
        # fav =  audio_feature_flat
        # fva = imu_feature_flat
        fav = self.cross_atten(imu_feature_flat.unsqueeze(1), audio_feature_flat.unsqueeze(1)).squeeze(1)
        fva = self.cross_atten(audio_feature_flat.unsqueeze(1), imu_feature_flat.unsqueeze(1)).squeeze(1)
        f_all = fav+fva
        # f_all = torch.cat([fav, fva], dim=1)
        # print(f_all.shape)
        z_combined = f_all
        # z_combined = self.eca(f_all.unsqueeze(1)).squeeze(1)
        # z_combined = self.fc1(z_combined)

        # Decode audio and IMU data
        x_audio_recon  = self.audio_decoder(recons_feature)
        x_audio_recon  = self.fc_audio(x_audio_recon)
        x_imu_recon    = self.imu_decoder(imu_recons)
        x_imu_recon    = self.fc_imu(x_imu_recon)

        gamma = self.estimation(z_combined)

        return x_audio_recon,x_imu_recon,z_combined,gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

 
        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov
        
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D)*eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            #det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = torch.cat(det_cov).cuda()
        #det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)


        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag


    def loss_function(self, x, x_hat, y, y_hat, z, gamma, lambda_energy, lambda_cov_diag):

        recon_error = torch.mean((x - x_hat) ** 2)+torch.mean((y - y_hat) ** 2)

        phi, mu, cov = self.compute_gmm_params(z, gamma)

        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

        loss = 10000*recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag

    def load_checkpoint(self, checkpoint_path):
        """
        Load the model, mean vector, and covariance matrix inverse from a checkpoint file.
        """
        checkpoint = torch.load(os.path.join(checkpoint_path,"last_weights"))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.mu = checkpoint['mu']
        self.phi = checkpoint['phi']
        self.cov = checkpoint['cov']
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
        self.lamda = 0.001
        self.lambda_energy, self.lambda_cov_diag = 0.1,0.005

    def train(self, num_epochs, log_interval=5):
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
            running_engergy = 0.0
            running_recons  = 0.0

            for i, data in enumerate(self.train_loader, 0):
                spec, imu, audio, imu_recons, audio_recons = data
                spec, imu, audio, imu_recons, audio_recons = spec.to(self.device), imu.to(self.device), audio.to(self.device), imu_recons.to(self.device), audio_recons.to(self.device)

                self.optimizer.zero_grad()
                x_audio_recon,x_imu_recon,z_combined,gamma = self.model(audio, imu)
                total_loss, sample_energy, recon_error, cov_diag = self.model.loss_function(audio,x_audio_recon,imu,x_imu_recon, z_combined, gamma, self.lambda_energy, self.lambda_cov_diag)
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                # Update parameters
                self.optimizer.step()
                running_loss += total_loss.item() * audio.size(0)
                running_engergy+=sample_energy* audio.size(0)
                running_recons+=recon_error* audio.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            energy_loss = running_engergy/ len(self.train_loader.dataset)
            recon_loss  = running_recons/len(self.train_loader.dataset)

            print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}, Energy Loss: {energy_loss:.4f}, Recons Loss: {recon_loss:.4f}')
            # Update the learning rate using the cosine annealing scheduler
            scheduler.step()

            # Log reconstructed and ground truth samples to TensorBoard every `log_interval` epochs
            if (epoch + 1) % log_interval == 0:
                self.save_checkpoint()
                # Save the model checkpoint after the last epoch
                # torch.save(self.model.state_dict(),os.path.join(self.checkpoint_path,"last_epoch"))     # Close TensorBoard writer




    def save_checkpoint(self):
        """
        Save the model, mean vector (mu), and covariance matrix inverse (sigma_inv) to a checkpoint file.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'mu': self.model.mu,  # Save the mean vector
            'phi': self.model.phi,  # Save the covariance matrix inverse
            "cov":self.model.cov
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
