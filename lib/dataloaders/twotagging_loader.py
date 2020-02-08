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


class Twotagging_Dataset(Abstract_dataset):
    """
    T:    text 

    # model 1 ground truth
    S1:      subject_begin
    S2:      subject_end

    # model 2 ground truth
    K1, K2:  sample one of (S1, S2)
    O1, O2:  corresponding object and relation        
    """

    def __init__(self, hyper, dataset):

        super(Twotagging_Dataset, self).__init__(hyper, dataset)

        self.spo_list = []
        self.text_list = []

        self.T, self.S1, self.S2, self.K1, self.K2, self.O1, self.O2, = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for line in open(os.path.join(self.data_root, dataset), "r"):
            line = line.strip("\n")
            instance = json.loads(line)

            self.text_list.append(instance["text"])
            self.spo_list.append(instance["spo_list"])

            text = instance["text"]
            items = {}
            for sp in instance["spo_list"]:
                subjectid = text.find(sp["subject"])
                objectid = text.find(sp["object"])
                if subjectid != -1 and objectid != -1:
                    key = (subjectid, subjectid + len(sp["subject"]))
                    if key not in items:
                        items[key] = []
                    items[key].append(
                        (
                            objectid,
                            objectid + len(sp["object"]),
                            self.relation_vocab[sp["predicate"]],
                        )
                    )
            if items:
                text_id = [self.word_vocab.get(c, self.word_vocab["oov"]) for c in text]
                self.T.append(text_id)
                s1, s2 = [0] * len(text), [0] * len(text)
                for j in items:

                    s1[j[0]] = 1
                    s2[j[1] - 1] = 1

                k1, k2 = choice(list(items.keys()))
                # TODO: not sure about unk class
                o1, o2 = [0] * len(text), [0] * len(text)  # 0是unk类（共49+1个类）
                for j in items[(k1, k2)]:
                    o1[j[0]] = j[2]
                    o2[j[1] - 1] = j[2]
                self.S1.append(s1)
                self.S2.append(s2)
                self.K1.append([k1])
                self.K2.append([k2 - 1])
                self.O1.append(o1)
                self.O2.append(o2)

        self.T = np.array(seq_padding(self.T))
        self.S1 = np.array(seq_padding(self.S1))
        self.S2 = np.array(seq_padding(self.S2))
        self.O1 = np.array(seq_padding(self.O1))
        self.O2 = np.array(seq_padding(self.O2))
        self.K1, self.K2 = np.array(self.K1), np.array(self.K2)

    def __getitem__(self, index):

        return (
            self.T[index],
            self.S1[index],
            self.S2[index],
            self.K1[index],
            self.K2[index],
            self.O1[index],
            self.O2[index],
            self.text_list[index],
            len(self.text_list[index]),
            self.spo_list[index],
        )

    def __len__(self):
        return len(self.text_list)


class Batch_reader(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))

        self.T = torch.LongTensor(transposed_data[0])
        self.S1 = torch.FloatTensor(transposed_data[1])
        self.S2 = torch.FloatTensor(transposed_data[2])
        self.K1 = torch.LongTensor(transposed_data[3])
        self.K2 = torch.LongTensor(transposed_data[4])
        self.O1 = torch.LongTensor(transposed_data[5])
        self.O2 = torch.LongTensor(transposed_data[6])

        self.text = transposed_data[7]
        self.length = transposed_data[8]
        self.spo_gold = transposed_data[-1]

    def pin_memory(self):

        self.T = self.T.pin_memory()
        self.S1 = self.S1.pin_memory()
        self.S2 = self.S2.pin_memory()
        self.K1 = self.K1.pin_memory()
        self.K2 = self.K2.pin_memory()
        self.O1 = self.O1.pin_memory()
        self.O2 = self.O2.pin_memory()

        return self


def collate_fn(batch):
    return Batch_reader(batch)


Twotagging_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)
