import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os
import copy

from typing import Dict, List, Tuple, Set, Optional
from functools import partial

from torchcrf import CRF

class MultiHeadSelection(nn.Module):
    def __init__(self, hyper) -> None:
        super(MultiHeadSelection, self).__init__()

        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))

        self.word_embeddings = nn.Embedding(num_embeddings=len(
            self.word_vocab),
                                            embedding_dim=hyper.emb_size)

        self.relation_emb = nn.Embedding(num_embeddings=len(
            self.relation_vocab),
                                         embedding_dim=hyper.rel_emb_size)
        # bio + pad
        self.bio_emb = nn.Embedding(num_embeddings=len(self.bio_vocab),
                                    embedding_dim=hyper.rel_emb_size)

        if hyper.cell_name == 'gru':
            self.encoder = nn.GRU(hyper.emb_size,
                                  hyper.hidden_size,
                                  bidirectional=True,
                                  batch_first=True)
        elif hyper.cell_name == 'lstm':
            self.encoder = nn.LSTM(hyper.emb_size,
                                   hyper.hidden_size,
                                   bidirectional=True,
                                   batch_first=True)
        else:
            raise ValueError('cell name should be gru/lstm!')

        if hyper.activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif hyper.activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('unexpected activation!')

        self.tagger = CRF(len(self.bio_vocab) - 1, batch_first=True)

    def forward(self):
        # TODO
        pass