from typing import Dict, List, Tuple, Set, Optional
from abc import ABC, abstractmethod, abstractstaticmethod
from torch import nn

import torch


class ABCModel(nn.Module):

    @abstractmethod
    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def run_metrics(self, output):
        pass

    @abstractstaticmethod
    def description(epoch, epoch_num, output):
        pass
