# -*- coding: utf-8 -*-
"""
One-class classification
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""
from deepod.core.networks.ts_network_tcn import TCNnet, TcnAE
from deepod.core.base_model_gsvdd import GDeepAD
from deepod.core.networks.base_networks import get_network
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class GDeepSVDDTS(GDeepAD):
    """
    Deep One-class Classification for Anomaly Detection (ICML'18)
     :cite:`ruff2018deepsvdd`
     

    Args:

        epochs (int, optional): 
            Number of training epochs. Default is 100.
            
        
        batch_size (int, optional): 
            Number of samples in a mini-batch. Default is 64.
        
        lr (float, optional): 
            Learning rate. Default is 1e-5.
        
        network (str, optional):
            Network structure for different data structures. Default is 'Transformer'.
        
        seq_len (int, optional): 
            Size of window used to create subsequences from the data. Default is 30.
        
        stride (int, optional): 
            Number of time points the window moves between subsequences. Default is 10.
        
        rep_dim (int, optional): 
            Dimensionality of the representation space. Default is 64.
        
        hidden_dims (Union[list, str, int], optional): 
            Dimensions for hidden layers. Default is '512'.
                - If list, each item is a layer
                - If str, neural units of hidden layers are split by comma
                - If int, number of neural units of single hidden layer
        
        act (str, optional): 
            Activation layer name. Choices are ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']. Default is 'GELU'.
        
        bias (bool, optional): 
            Whether to add a bias term in linear layers. Default is False.
        
        n_heads (int, optional): 
            Number of heads in multi-head attention. Default is 8.
        
        d_model (int, optional): 
            Number of dimensions in Transformer model. Default is 512.
        
        attn (str, optional): 
            Type of attention mechanism. Default is 'self_attn'.
        
        pos_encoding (str, optional): 
            Manner of positional encoding. Default is 'fixed'.
        
        norm (str, optional): 
            Manner of normalization in Transformer. Default is 'LayerNorm'.
        
        epoch_steps (int, optional): 
            Maximum steps in an epoch. Default is -1.
        
        prt_steps (int, optional): 
            Number of epoch intervals per printing. Default is 10.
        
        device (str, optional): 
            Torch device. Default is 'cuda'.
        
        verbose (int, optional): 
            Verbosity mode. Default is 2.
        
        random_state (int, optional): 
            Seed used by the random number generator. Default is 42.
    
    """
    
    def __init__(self, epochs=100, batch_size=64, lr=1e-5,
                 network='Transformer', seq_len=30, stride=10,
                 rep_dim=64, hidden_dims='512', act='GELU', bias=False,
                 n_heads=8, d_model=512, attn='self_attn', pos_encoding='fixed', norm='LayerNorm',
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        """
        Initializes the DeepSVDDTS model with the specified parameters.
        """
        
        super(GDeepSVDDTS, self).__init__(
            model_name='DeepSVDD', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            network=network, seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = attn
        self.pos_encoding = pos_encoding
        self.norm = norm
        
        self.c = None
        return

    def training_prepare(self, X, y):
        """
        Prepares the training process by setting up data loaders and initializing the network and loss criterion.

        Args:
        
            X (torch.Tensor): 
                Input tensor of the features.
            
            y (torch.Tensor): 
                Input tensor of the labels.

        Returns:
        
            train_loader (DataLoader):
                DataLoader for the training data.
            
            net (nn.Module): 
                Initialized neural network model.
            
            criterion (DSVDDLoss):
                Loss function for DeepSVDD.
                
        """
        
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }
        if self.network == 'Transformer':
            network_params['n_heads'] = self.n_heads
            network_params['d_model'] = self.d_model
            network_params['pos_encoding'] = self.pos_encoding
            network_params['norm'] = self.norm
            network_params['attn'] = self.attn
            network_params['seq_len'] = self.seq_len
        elif self.network == 'ConvSeq':
            network_params['seq_len'] = self.seq_len
        
        # network_class = get_network(self.network)
        network_class = TcnAE(n_features=self.n_features,n_hidden=512,n_output=32,activation=self.act,bias=self.bias)
        net = network_class.to(self.device)

        if self.verbose >= 2:
            print(net)

        return train_loader, net

    def inference_prepare(self, X):
        """
        Prepares the model for inference by setting up data loaders.

        Args:
        
            X (torch.Tensor): 
                Input tensor of the features for inference.

        Returns:
        
            test_loader (DataLoader):
                DataLoader for inference.
                
        """
        
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        return test_loader

    def training_forward(self, batch_x, net,flag):
        """
        Performs a forward pass during training.

        Args:
        
            batch_x (torch.Tensor): 
                Batch of input data.
            
            net (nn.Module): 
                The neural network model.
            
            criterion (DSVDDLoss):
                Loss function for DeepSVDD.

        Returns:
            
            loss (torch.Tensor):
                Computed loss for the batch.
            
        """
        
        batch_x = batch_x.float().to(self.device)
        distances,radius,rep,dec = net(batch_x,flag)
        gaussian_loss = torch.mean(torch.relu(distances**2-radius**2))+radius**2
        reconstruction_loss = nn.SmoothL1Loss()(batch_x,dec)
        reg_loss = self.compute_entropy_and_covariance_loss(rep)*0.001
        loss = gaussian_loss+reconstruction_loss+reg_loss
        # print(torch.mean(distances).cpu().detach().numpy(),radius.cpu().detach().numpy())
        return loss

    def inference_forward(self, batch_x, net):
        """
        Performs a forward pass during inference.

        Args:
            
            batch_x (torch.Tensor): 
                Batch of input data.
            
            net (nn.Module): 
                The neural network model.
            
            criterion (DSVDDLoss): 
                Loss function for DeepSVDD to calculate anomaly score.

        Returns:
            
            batch_z (torch.Tensor): 
                The encoded batch of data in the feature space.
            
            s (torch.Tensor): 
                The anomaly scores for the batch.
        """
        net.is_train=0
        batch_x = batch_x.float().to(self.device)
        distances,radius,rep,dec = net(batch_x)
        loss = distances
        return rep, loss
    
    def compute_entropy_and_covariance_loss(self, z):
        """
        Compute both the entropy loss and the covariance regularization loss.
        Includes a diagonal penalty term to prevent small diagonal entries.
        """
        # Calculate covariance matrix
        cov_matrix = torch.cov(z.T)
        
        cov_matrix += 1e-3 * torch.eye(cov_matrix.size(0)).to(cov_matrix.device)
        
        diagonal_entries = torch.diag(cov_matrix)
        diagonal_penalty = torch.sum(1.0 / (diagonal_entries + 1e-6))  # Avoid division by zero
        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        max_eigenvalue = torch.max(eigenvalues)
        min_eigenvalue = torch.min(eigenvalues)
        epsilon = 1e-6  # To avoid division by zero
        condition_number_penalty = (max_eigenvalue / (min_eigenvalue + epsilon))
        
        # Combine the losses with weights (tunable hyperparameters)
        diagonal_weight = 1
        condition_weight = 1
        total_loss = diagonal_weight * diagonal_penalty +condition_number_penalty
    
        
        return total_loss
 
