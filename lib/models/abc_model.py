from abc import ABC, abstractmethod, abstractstaticmethod
from torch import nn

import torch


class Model(nn.Module):
    # def __init__(self, hyper) -> None:

    @abstractmethod
    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def run_metrics(self, output):
        pass

    @abstractstaticmethod
    def description(epoch, epoch_num, output):
        pass
