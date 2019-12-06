import torch
import torch.nn
from torch.nn.modules.loss import _WeightedLoss

class TVLoss(_WeightedLoss):
    """
    Total variation loss.
    """
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(TVLoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, img, tv_weight):
        w_variance = (img[:,:,:,:-1] - img[:,:,:,1:]).abs().sum()
        h_variance = (img[:,:,:-1,:] - img[:,:,1:,:]).abs().sum()
        loss = tv_weight * (h_variance + w_variance)
        return loss