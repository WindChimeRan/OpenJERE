from recordclass import recordclass
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from functools import partial
from typing import Dict, List, Tuple, Set, Optional

from .abc_dataset import Abstract_dataset
from lib.config.const import seq_padding


import numpy as np
import os
import json
import random

Sample = recordclass("Sample", "Id SrcLen SrcWords TrgLen TrgWords AdjMat")


def get_target_vocab_mask(src_words, word_vocab):
    mask = []
    for i in range(0, len(word_vocab)):
        mask.append(1)
    for word in src_words:
        if word in word_vocab:
            mask[word_vocab[word]] = 0

    mask[word_vocab["<oov>"]] = 0
    mask[word_vocab["<eos>"]] = 0
    mask[word_vocab["<;>"]] = 0
    mask[word_vocab["<|>"]] = 0
    return mask


class WDec_Dataset(Abstract_dataset):
    def __init__(self, hyper, dataset):

        super(WDec_Dataset, self).__init__(hyper, dataset)

        self.seq_list = []
        self.text_list = []
        self.spo_list = []

        for line in open(os.path.join(self.data_root, dataset), "r"):
            line = line.strip("\n")
            instance = json.loads(line)

            self.seq_list.append(instance["seq"])
            self.text_list.append(self.hyper.tokenizer(instance["text"]))
            self.spo_list.append(instance["spo_list"])

    def __getitem__(self, index):

        seq = self.seq_list[index]
        text = self.text_list[index]
        spo = self.spo_list[index]

        tokens_id = self.text2id(text)
        seq_id = self.seq2id(seq)

        # TODO: check
        trg_vocab_mask = get_target_vocab_mask(text)

        return tokens_id, seq_id, len(tokens_id), len(seq_id), spo, text

    def __len__(self):
        return len(self.text_list)

    def text2id(self, text: List[str]) -> torch.tensor:
        oov = self.word_vocab["<oov>"]
        text_list = list(
            map(lambda x: self.word_vocab.get(x, oov), self.tokenizer(text))
        )
        return text_list

    def seq2id(self, seq):
        oov = self.word_vocab["<oov>"]
        tuples = seq.strip().split("<|>")
        random.shuffle(tuples)
        new_trg_line = " <|> ".join(tuples)
        assert len(seq.split()) == len(new_trg_line.split())
        trg_line = new_trg_line

        trg_words = trg_line.split()
        trg_words.append("<EOS>")
        trg_words = list(map(lambda x: self.word_vocab.get(x, oov), trg_words))
        return trg_words


class Batch_reader(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))

        self.tokens_id = torch.tensor(seq_padding(transposed_data[0]))
        self.seq_id = torch.tensor(seq_padding(transposed_data[1]))

        self.src_words_mask = torch.gt(self.tokens_id, 0)
        self.src_words_mask = torch.gt(self.seq_id, 0)

        self.en_len = transposed_data[2]
        self.de_len = transposed_data[3]

        self.spo_gold = transposed_data[4]
        self.text = transposed_data[5]

    def pin_memory(self):
        self.tokens_id = self.tokens_id.pin_memory()
        self.seq_id = self.seq_id.pin_memory()
        self.mask_decode = self.mask_decode.pin_memory()
        return self


def collate_fn(batch):
    return Batch_reader(batch)


WDec_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)
