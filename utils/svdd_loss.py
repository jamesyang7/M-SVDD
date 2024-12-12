import torch
import torch.nn as nn

class SVDDLoss(nn.Module):
    def __init__(self, center, radius, nu=0.1):
        super(SVDDLoss, self).__init__()
        self.center = center
        self.radius = nn.Parameter(radius)
        self.nu = nu

    def forward(self, distances):
        distance_loss = torch.mean(distances)
        radius_loss = self.radius ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(distances), distances - self.radius ** 2))
        total_loss = radius_loss + distance_loss

        return total_loss
    



class DSVDDLoss(torch.nn.Module):
    """

    Custom loss function for Deep Support Vector Data Description (Deep SVDD).
    
    This loss function computes the distance between each data point in the representation
    space and the center of the hypersphere and aims to minimize this distance for normal data points.

    Args:
    
        c (torch.Tensor):
            The center of the hypersphere in the representation space.
            
        reduction (str, optional): 
            Specifies the reduction to apply to the output. Choices are 'none', 'mean', 'sum'. Default is 'mean'.
                - If ``'none'``: no reduction will be applied;
                - If ``'mean'``: the sum of the output will be divided by the number of elements in the output;
                - If ``'sum'``: the output will be summed

    """
    
    def __init__(self, c, reduction='mean'):
        """
        Initializes the DSVDDLoss with the hypersphere center and reduction method.
        """
        
        super(DSVDDLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep, reduction=None):
        """
        Calculates the Deep SVDD loss for a batch of representations.

        Args:
        
            rep (torch.Tensor): 
                The representation of the batch of data.
            
            reduction (str, optional): 
                The reduction method to apply. If None, will use the specified 'reduction' attribute. Default is None.

        Returns:
        
            loss (torch.Tensor): 
                The calculated loss based on the representations and the center 'c'.
                
        """
        
        loss = torch.sum((rep - self.c) ** 2, dim=1)

        if reduction is None:
            reduction = self.reduction

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss