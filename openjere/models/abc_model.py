from typing import Dict, List, Tuple, Set, Optional
from abc import ABC, abstractmethod, abstractstaticmethod
from torch import nn

import torch


class ABCModel(ABC, nn.Module):
    @abstractmethod
    def run_metrics(self, output):
        pass

    @abstractstaticmethod
    def description(epoch, epoch_num, output):
        pass
