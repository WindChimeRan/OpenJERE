import torch
import torch.nn as nn


class MaskedBCE(nn.Module):
    def __init__(self):
        super(MaskedBCE, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, logits, gt, mask):
        loss = self.BCE(logits, gt)
        loss = torch.sum(loss.mul(mask)) / torch.sum(mask)
        return loss
