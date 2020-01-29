import os
import json

import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

from functools import partial
from typing import Dict, List, Tuple, Set, Optional

from .abc_dataset import Abstract_dataset


class Copymb_Dataset(Abstract_dataset):
    def __init__(self, hyper, dataset):

        super(Copymb_Dataset, self).__init__(hyper, dataset)

        self.seq_list = []
        self.text_list = []
        self.bio_list = []
        self.spo_list = []

        for line in open(os.path.join(self.data_root, dataset), 'r'):
            line = line.strip("\n")
            instance = json.loads(line)

            self.seq_list.append(instance['seq'])
            self.text_list.append(instance['text'])
            self.bio_list.append(instance['bio'])
            self.spo_list.append(instance['spo_list'])
        
    def __getitem__(self, index):

        seq = self.seq_list[index]
        text = self.text_list[index]
        bio = self.bio_list[index]
        spo = self.spo_list[index]

        tokens_id = self.text2tensor(text)
        bio_id = self.bio2tensor(bio)
        seq_id, mask_decode = self.seq2tensor(text, seq)

        return tokens_id, bio_id, seq_id, len(text), spo, text, bio, mask_decode

    def __len__(self):
        return len(self.text_list)

    def text2tensor(self, text: List[str]) -> torch.tensor:
        oov = self.word_vocab['oov']
        padded_list = list(map(lambda x: self.word_vocab.get(x, oov), self.tokenizer(text)))
        padded_list.extend([self.word_vocab['<pad>']] *
                           (self.hyper.max_text_len - len(text)))
        return torch.tensor(padded_list)

    def bio2tensor(self, bio):
        # here we pad bio with "O". Then, in our model, we will mask this "O" padding.
        # in multi-head selection, we will use "<pad>" token embedding instead.
        padded_list = list(map(lambda x: self.bio_vocab[x], bio))
        padded_list.extend([self.bio_vocab['O']] *
                           (self.hyper.max_text_len - len(bio)))
        return torch.tensor(padded_list)

    def seq2tensor(self, text, seq):
        # s p o
        seq_tensor = torch.zeros(
            (self.hyper.max_text_len, self.hyper.max_decode_len * 2 + 1)).long()

        mask_tensor = seq_tensor.bool()

        NA = self.relation_vocab['N']
        for i in range(self.hyper.max_text_len):
            if str(i) not in seq:
                seq_tensor[i, 0] = NA 
                mask_tensor[i, 0] = True
        for s, pair in seq.items():
            for i, ptr in enumerate(pair):
                seq_tensor[int(s), i] = ptr
                mask_tensor[int(s), i] = True
            seq_tensor[int(s), i + 1] = NA
            mask_tensor[int(s), i + 1] = True
        
        assert mask_tensor.sum() > 0

        return seq_tensor, mask_tensor


class Batch_reader(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))
        # tokens_id, bio_id, seq_id, spo, text, bio

        self.tokens_id = torch.stack(transposed_data[0], 0)
        self.bio_id = torch.stack(transposed_data[1], 0)
        self.seq_id = torch.stack(transposed_data[2], 0)
        self.mask_decode = torch.stack(transposed_data[7], 0)

        self.length = transposed_data[3]

        self.spo_gold = transposed_data[4]
        self.text = transposed_data[5]
        self.bio = transposed_data[6]

    def pin_memory(self):
        self.tokens_id = self.tokens_id.pin_memory()
        self.bio_id = self.bio_id.pin_memory()
        self.seq_id = self.seq_id.pin_memory()
        self.mask_decode = self.mask_decode.pin_memory()
        return self


def collate_fn(batch):
    return Batch_reader(batch)


Copymb_loader = partial(
    DataLoader, collate_fn=collate_fn, pin_memory=True)
