import os
import json

import torch

from torch.utils.data import Dataset

from functools import partial
from typing import Dict, List, Tuple, Set, Optional

from abc import ABC, abstractmethod


class Abstract_dataset(ABC, Dataset):
    def __init__(self, hyper, dataset):
        self.hyper = hyper
        self.data_root = hyper.data_root

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass
