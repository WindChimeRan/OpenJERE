#! -*- coding:utf-8 -*-

import json
import numpy as np
from random import choice

import os
import torch.utils.data as Data
import torch.nn.functional as F

import time

# torch.backends.cudnn.benchmark = True

import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

from functools import partial
from typing import Dict, List, Tuple, Set, Optional

from .abc_dataset import Abstract_dataset


def get_now_time():
    a = time.time()
    return time.ctime(a)


def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]


def seq_padding_vec(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [[1, 0]] * (ML - len(x)) for x in X]


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


class Seq2umt_Dataset(Abstract_dataset):
    def __init__(self, hyper, dataset):

        super(Seq2umt_Dataset, self).__init__(hyper, dataset)

        self.spo_list = []
        self.text_list = []

        (
            self.T,
            self.S1,
            self.S2,
            self.S_K1_in,
            self.O_K1_in,
            self.S_K2_in,
            self.O_K2_in,
            self.O1,
            self.O2,
            self.R_in,
            self.R_gt,
        ) = ([] for _ in range(11))

        for line in open(os.path.join(self.data_root, dataset), "r", encoding="utf-8"):
            line = line.strip("\n")
            instance = json.loads(line)

            text = instance["text"]
            spo_list = instance["spo_list"]

            text_id = [self.word_vocab.get(c, self.word_vocab["oov"]) for c in text]
            self.T.append(text_id)

            # training
            r = instance.get("r", 0)
            s_k1 = instance.get("s_k1", 0)
            s_k2 = instance.get("s_k2", 0)
            o_k1 = instance.get("o_k1", 0)
            o_k2 = instance.get("o_k2", 0)

            rel_gt = instance.get("rel_gt", [])
            s1_gt = instance.get("s1_gt", [])
            s2_gt = instance.get("s2_gt", [])
            o1_gt = instance.get("o1_gt", [])
            o2_gt = instance.get("o2_gt", [])

            self.text_list.append(text)
            self.spo_list.append(spo_list)

            self.S1.append(s1_gt)
            self.S2.append(s2_gt)
            self.O1.append(o1_gt)
            self.O2.append(o2_gt)
            self.R_gt.append(rel_gt)

            self.R_in.append(r)
            self.S_K1_in.append([s_k1])
            self.S_K2_in.append([s_k2])
            self.O_K1_in.append([o_k1])
            self.O_K2_in.append([o_k2])

        self.T = np.array(seq_padding(self.T))

        # training
        self.S1 = np.array(seq_padding(self.S1))
        self.S2 = np.array(seq_padding(self.S2))
        self.O1 = np.array(seq_padding(self.O1))
        self.O2 = np.array(seq_padding(self.O2))
        self.R_gt = np.array(self.R_gt)

        # self.K1_in, self.K2_in = np.array(self.K1_in), np.array(self.K2_in)
        # only two time step are used for training
        self.S_K1_in = np.array(self.S_K1_in)
        self.S_K2_in = np.array(self.S_K2_in)
        self.O_K1_in = np.array(self.O_K1_in)
        self.O_K2_in = np.array(self.O_K2_in)
        self.R_in = np.array(self.R_in)

    def __getitem__(self, index):

        return (
            self.T[index],
            self.S1[index],
            self.S2[index],
            self.O1[index],
            self.O2[index],
            self.R_gt[index],
            self.R_in[index],
            self.S_K1_in[index],
            self.S_K2_in[index],
            self.O_K1_in[index],
            self.O_K2_in[index],
            self.text_list[index],
            len(self.text_list[index]),
            self.spo_list[index],
        )

    def __len__(self):
        return len(self.text_list)


class Batch_reader(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))

        lens = transposed_data[12]
        transposed_data, orig_idx = sort_all(transposed_data, lens)

        self.orig_idx = orig_idx
        self.length = transposed_data[10]

        self.T = torch.LongTensor(transposed_data[0])
        self.S1 = torch.FloatTensor(transposed_data[1])
        self.S2 = torch.FloatTensor(transposed_data[2])
        self.O1 = torch.FloatTensor(transposed_data[3])
        self.O2 = torch.FloatTensor(transposed_data[4])

        self.R_gt = torch.FloatTensor(transposed_data[5])
        self.R_in = torch.LongTensor(transposed_data[6])

        self.S_K1_in = torch.LongTensor(transposed_data[7])
        self.S_K2_in = torch.LongTensor(transposed_data[8])
        self.O_K1_in = torch.LongTensor(transposed_data[9])
        self.O_K2_in = torch.LongTensor(transposed_data[10])
        self.text = transposed_data[11]
        self.length = transposed_data[12]

        self.spo_gold = transposed_data[-1]

    def pin_memory(self):

        self.T = self.T.pin_memory()
        self.S1 = self.S1.pin_memory()
        self.S2 = self.S2.pin_memory()
        self.O1 = self.O1.pin_memory()
        self.O2 = self.O2.pin_memory()

        self.R_gt = self.R_gt.pin_memory()
        self.R_in = self.R_in.pin_memory()

        self.S_K1_in = self.S_K1_in.pin_memory()
        self.S_K2_in = self.S_K2_in.pin_memory()
        self.O_K1_in = self.O_K1_in.pin_memory()
        self.O_K2_in = self.O_K2_in.pin_memory()

        return self


def collate_fn(batch):
    return Batch_reader(batch)


Seq2umt_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)
