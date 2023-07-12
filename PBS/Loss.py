import torch
from torch import nn
import torch.nn.functional as F


class PixWiseBCELoss(nn.Module):
    def __init__(self, beta=0.2):
        super().__init__()
        self.binary_criterion = nn.BCELoss()
        self.map_criterion = nn.SmoothL1Loss()
        self.beta = beta

    def forward(self, net_mask, net_label, target_mask, target_label):
        map_loss = self.map_criterion(net_mask, target_mask)
        binary_loss = self.binary_criterion(net_label.type(torch.FloatTensor), target_label.type(torch.FloatTensor))
        loss = map_loss * self.beta + binary_loss * (1 - self.beta)
        return loss