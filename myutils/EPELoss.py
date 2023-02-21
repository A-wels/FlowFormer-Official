import torch.nn as nn
import torch

class EPELoss(nn.Module):
    def __init__(self):
        super(EPELoss, self).__init__()

    def forward(self, pred_flow, gt_flow):
        # calculate the Euclidean distance between predicted and ground truth flow vectors
        epe = torch.sum((pred_flow - gt_flow)**6, dim=1).sqrt()
        """print(pred_flow.shape)
        print(gt_flow.shape)
        print(pred_flow[0][0][60][150])
        print(gt_flow[0][0][60][150])
        print(epe[0][60][150])
        exit()"""
        return torch.mean(epe)
