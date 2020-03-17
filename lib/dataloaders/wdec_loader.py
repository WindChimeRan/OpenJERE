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


# def get_max_len(sample_batch):
#     src_max_len = len(sample_batch[0].SrcWords)
#     for idx in range(1, len(sample_batch)):
#         if len(sample_batch[idx].SrcWords) > src_max_len:
#             src_max_len = len(sample_batch[idx].SrcWords)

#     trg_max_len = len(sample_batch[0].TrgWords)
#     for idx in range(1, len(sample_batch)):
#         if len(sample_batch[idx].TrgWords) > trg_max_len:
#             trg_max_len = len(sample_batch[idx].TrgWords)

#     return src_max_len, trg_max_len


# def get_words_index_seq(words, max_len):
#     seq = list()
#     for word in words:
#         if word in word_vocab:
#             seq.append(word_vocab[word])
#         else:
#             seq.append(word_vocab["<UNK>"])
#     pad_len = max_len - len(words)
#     for i in range(0, pad_len):
#         seq.append(word_vocab["<PAD>"])
#     return seq


# def get_target_words_index_seq(words, max_len):
#     seq = list()
#     for word in words:
#         if word in word_vocab:
#             seq.append(word_vocab[word])
#         else:
#             seq.append(word_vocab["<UNK>"])
#     pad_len = max_len - len(words)
#     for i in range(0, pad_len):
#         seq.append(word_vocab["<EOS>"])
#     return seq


# def get_padded_mask(cur_len, max_len):
#     mask_seq = list()
#     for i in range(0, cur_len):
#         mask_seq.append(0)
#     pad_len = max_len - cur_len
#     for i in range(0, pad_len):
#         mask_seq.append(1)
#     return mask_seq


# def get_rel_mask(trg_words, max_len):
#     mask_seq = list()
#     for word in trg_words:
#         mask_seq.append(0)
#         # if word in relations:
#         #     mask_seq.append(0)
#         # else:
#         #     mask_seq.append(1)
#     pad_len = max_len - len(trg_words)
#     for i in range(0, pad_len):
#         mask_seq.append(1)
#     return mask_seq


# def get_char_seq(words, max_len):
#     char_seq = list()
#     for i in range(0, conv_filter_size - 1):
#         char_seq.append(char_vocab['<PAD>'])
#     for word in words:
#         for c in word[0:min(len(word), max_word_len)]:
#             if c in char_vocab:
#                 char_seq.append(char_vocab[c])
#             else:
#                 char_seq.append(char_vocab['<UNK>'])
#         pad_len = max_word_len - len(word)
#         for i in range(0, pad_len):
#             char_seq.append(char_vocab['<PAD>'])
#         for i in range(0, conv_filter_size - 1):
#             char_seq.append(char_vocab['<PAD>'])

#     pad_len = max_len - len(words)
#     for i in range(0, pad_len):
#         for i in range(0, max_word_len + conv_filter_size - 1):
#             char_seq.append(char_vocab['<PAD>'])
#     return char_seq


def get_target_vocab_mask(src_words):
    mask = []
    for i in range(0, len(word_vocab)):
        mask.append(1)
    for word in src_words:
        if word in word_vocab:
            mask[word_vocab[word]] = 0
    for rel in relations:
        mask[word_vocab[rel]] = 0

    mask[word_vocab["<UNK>"]] = 0
    mask[word_vocab["<EOS>"]] = 0
    mask[word_vocab[";"]] = 0
    mask[word_vocab["|"]] = 0
    return mask


def get_batch_data(cur_samples, is_training=False):
    """
    Returns the training samples and labels as numpy array
    """
    batch_src_max_len, batch_trg_max_len = get_max_len(cur_samples)
    src_words_list = list()
    src_words_mask_list = list()
    src_char_seq = list()

    trg_words_list = list()
    trg_vocab_mask = list()
    adj_lst = []

    target = list()
    cnt = 0
    for sample in cur_samples:
        src_words_list.append(get_words_index_seq(sample.SrcWords, batch_src_max_len))
        src_words_mask_list.append(get_padded_mask(sample.SrcLen, batch_src_max_len))
        src_char_seq.append(get_char_seq(sample.SrcWords, batch_src_max_len))
        trg_vocab_mask.append(get_target_vocab_mask(sample.SrcWords))

        cur_masked_adj = np.zeros(
            (batch_src_max_len, batch_src_max_len), dtype=np.float32
        )
        cur_masked_adj[: len(sample.SrcWords), : len(sample.SrcWords)] = sample.AdjMat
        adj_lst.append(cur_masked_adj)

        if is_training:
            padded_trg_words = get_words_index_seq(sample.TrgWords, batch_trg_max_len)
            trg_words_list.append(padded_trg_words)
            target.append(padded_trg_words[1:])
        else:
            trg_words_list.append(get_words_index_seq(["<SOS>"], 1))
        cnt += 1

    return {
        "src_words": np.array(src_words_list, dtype=np.float32),
        "src_chars": np.array(src_char_seq),
        "src_words_mask": np.array(src_words_mask_list),
        "adj": np.array(adj_lst),
        "trg_vocab_mask": np.array(trg_vocab_mask),
        "trg_words": np.array(trg_words_list, dtype=np.int32),
        "target": np.array(target),
    }


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

        # TODO: mask trg words
        trg_vocab_mask = None

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
