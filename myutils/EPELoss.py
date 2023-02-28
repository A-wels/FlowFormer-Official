import torch.nn as nn
import torch

class EPELoss(nn.Module):
    def __init__(self):
        super(EPELoss, self).__init__()

    def forward(self, pred_flow, gt_flow):
        # calculate the Euclidean distance between predicted and ground truth flow vectors
        epe = torch.sum((pred_flow - gt_flow)**2, dim=1).sqrt()
        return torch.mean(epe)
