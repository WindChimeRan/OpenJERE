# https://128.84.21.199/pdf/1911.09886.pdf
# https://github.com/nusnlp/PtrNetDecoding4JERE
import sys
import os
import numpy as np
import random

import pickle
import datetime

from collections import OrderedDict
from tqdm import tqdm
from recordclass import recordclass
from typing import Dict, List, Tuple, Set, Optional
from functools import partial

from lib.metrics import F1_triplet
from lib.models.abc_model import ABCModel
from lib.config.const import *

import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json


# enc_type = ['LSTM', 'GCN', 'LSTM-GCN'][0]
# att_type = ['None', 'Unigram', 'N-Gram-Enc'][1]
def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
random_seed = 1234
n_gpu = torch.cuda.device_count()
set_random_seeds(random_seed)

# src_data_folder = sys.argv[3]
# trg_data_folder = sys.argv[4]
# if not os.path.exists(trg_data_folder):
#     os.mkdir(trg_data_folder)
# model_name = 1
# job_mode = sys.argv[5]
# run = int(sys.argv[5])
batch_size = 32
num_epoch = 30
max_src_len = 100
max_trg_len = 50
# embedding_file = os.path.join(src_data_folder, 'w2v.txt')
update_freq = 1

copy_on = True

gcn_num_layers = 3
word_embed_dim = 300
word_min_freq = 2
char_embed_dim = 50
char_feature_size = 50
conv_filter_size = 3
max_word_len = 10

# enc_inp_size = word_embed_dim + char_feature_size
enc_inp_size = word_embed_dim
enc_hidden_size = word_embed_dim
dec_inp_size = enc_hidden_size
dec_hidden_size = dec_inp_size

drop_rate = 0.3
layers = 1
early_stop_cnt = 5
sample_cnt = 0
Sample = recordclass("Sample", "Id SrcLen SrcWords TrgLen TrgWords AdjMat")

# rel_file = os.path.join(src_data_folder, 'relations.txt')
# relations = get_relations(rel_file)


class WordEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, pre_trained_embed_matrix, drop_out_rate):
        super(WordEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(pre_trained_embed_matrix))
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        word_embeds = self.embeddings(words_seq)
        word_embeds = self.dropout(word_embeds)
        return word_embeds

    def weight(self):
        return self.embeddings.weight


class CharEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, drop_out_rate):
        super(CharEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, words_seq):
        char_embeds = self.embeddings(words_seq)
        char_embeds = self.dropout(char_embeds)
        return char_embeds


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, is_bidirectional, drop_out_rate):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.is_bidirectional = is_bidirectional
        self.drop_rate = drop_out_rate
        # self.char_embeddings = CharEmbeddings(len(char_vocab), char_embed_dim, drop_rate)
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.layers,
            batch_first=True,
            bidirectional=self.is_bidirectional,
        )

        self.dropout = nn.Dropout(self.drop_rate)
        self.conv1d = nn.Conv1d(char_embed_dim, char_feature_size, conv_filter_size)
        self.max_pool = nn.MaxPool1d(
            max_word_len + conv_filter_size - 1, max_word_len + conv_filter_size - 1
        )

    def forward(self, words_input, char_seq, adj, is_training=False):
        # char_embeds = self.char_embeddings(char_seq)
        # char_embeds = char_embeds.permute(0, 2, 1)

        # char_feature = torch.tanh(self.max_pool(self.conv1d(char_embeds)))
        # char_feature = char_feature.permute(0, 2, 1)

        # words_input = torch.cat((words_input, char_feature), -1)
        words_input = words_input
        outputs, hc = self.lstm(words_input)
        outputs = self.dropout(outputs)

        return outputs


def mean_over_time(x, mask):
    x.data.masked_fill_(mask.unsqueeze(2).data, 0)
    x = torch.sum(x, dim=1)
    time_steps = torch.sum(mask.eq(0), dim=1, keepdim=True).float()
    x /= time_steps
    return x


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.linear_ctx = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.linear_query = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.v = nn.Linear(self.input_dim, 1)

    def forward(self, s_prev, enc_hs, src_mask):
        uh = self.linear_ctx(enc_hs)
        wq = self.linear_query(s_prev)
        wquh = torch.tanh(wq + uh)
        attn_weights = self.v(wquh).squeeze()
        attn_weights.data.masked_fill_(src_mask.data, -float("inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        ctx = torch.bmm(attn_weights.unsqueeze(1), enc_hs).squeeze()
        return ctx, attn_weights


class Decoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, layers, drop_out_rate, max_length, vocab_length
    ):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.drop_rate = drop_out_rate
        self.max_length = max_length

        self.attention = Attention(input_dim)
        self.lstm = nn.LSTMCell(2 * self.input_dim, self.hidden_dim, self.layers)

        self.dropout = nn.Dropout(self.drop_rate)
        self.ent_out = nn.Linear(self.input_dim, vocab_length)

    def forward(
        self, y_prev, h_prev, enc_hs, src_word_embeds, src_mask, is_training=False
    ):
        src_time_steps = enc_hs.size()[1]

        s_prev = h_prev[0]
        s_prev = s_prev.unsqueeze(1)
        s_prev = s_prev.repeat(1, src_time_steps, 1)
        ctx, attn_weights = self.attention(s_prev, enc_hs, src_mask)

        y_prev = y_prev.squeeze()
        s_cur = torch.cat((y_prev, ctx), 1)
        hidden, cell_state = self.lstm(s_cur, h_prev)
        hidden = self.dropout(hidden)
        output = self.ent_out(hidden)
        return output, (hidden, cell_state), attn_weights


class WDec(ABCModel):
    def __init__(self, hyper):
        super(WDec, self).__init__()
        self.hyper = hyper
        self.order = hyper.order
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, "word_vocab.json"), "r", encoding="utf-8")
        )
        self.relation_vocab = json.load(
            open(
                os.path.join(self.data_root, "relation_vocab.json"),
                "r",
                encoding="utf-8",
            )
        )
        self.word_embeddings = nn.Embedding(len(self.word_vocab), self.hyper.emb_size)

        # self.word_embeddings = WordEmbeddings(len(word_vocab), word_embed_dim, word_embed_matrix, drop_rate)
        self.encoder = Encoder(
            enc_inp_size, int(enc_hidden_size / 2), layers, True, drop_rate
        )
        self.decoder = Decoder(
            dec_inp_size,
            dec_hidden_size,
            layers,
            drop_rate,
            max_trg_len,
            len(self.word_vocab),
        )
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.metrics = F1_triplet()
        self.get_metric = self.metrics.get_metric

    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        src_words_seq = sample.tokens_id.cuda(self.gpu)
        # src_chars_seq = sample.src_chars_seq
        src_mask = sample.src_words_mask.cuda(self.gpu)
        trg_vocab_mask = sample.trg_vocab_mask.cuda(self.gpu)

        if is_train:
            trg_words_seq = sample.seq_id.cuda(self.gpu)
            trg_word_embeds = self.word_embeddings(trg_words_seq)

        src_word_embeds = self.word_embeddings(src_words_seq)

        batch_len = src_word_embeds.size()[0]
        if is_train:
            time_steps = trg_word_embeds.size()[1] - 1
        else:
            time_steps = max_trg_len

        encoder_output = self.encoder(src_word_embeds, None, None, is_train)

        h0 = torch.FloatTensor(torch.zeros(batch_len, word_embed_dim))
        h0 = h0.cuda()
        c0 = torch.FloatTensor(torch.zeros(batch_len, word_embed_dim))
        c0 = c0.cuda()
        dec_hid = (h0, c0)

        output = {}

        if is_train:
            dec_inp = trg_word_embeds[:, 0, :]
            dec_out, dec_hid, dec_attn = self.decoder(
                dec_inp, dec_hid, encoder_output, src_word_embeds, src_mask, is_train
            )
            dec_out = dec_out.view(-1, len(self.word_vocab))
            dec_out = F.log_softmax(dec_out, dim=-1)
            dec_out = dec_out.unsqueeze(1)
            for t in range(1, time_steps):
                dec_inp = trg_word_embeds[:, t, :]
                cur_dec_out, dec_hid, dec_attn = self.decoder(
                    dec_inp,
                    dec_hid,
                    encoder_output,
                    src_word_embeds,
                    src_mask,
                    is_train,
                )
                cur_dec_out = cur_dec_out.view(-1, len(self.word_vocab))
                dec_out = torch.cat(
                    (dec_out, F.log_softmax(cur_dec_out, dim=-1).unsqueeze(1)), 1
                )
        else:
            # TODO
            dec_inp = trg_word_embeds[:, 0, :]
            dec_out, dec_hid, dec_attn = self.decoder(
                dec_inp, dec_hid, encoder_output, src_word_embeds, src_mask, is_train
            )
            dec_out = dec_out.view(-1, len(self.word_vocab))
            if copy_on:
                dec_out.data.masked_fill_(trg_vocab_mask.data, -float("inf"))
            dec_out = F.log_softmax(dec_out, dim=-1)
            topv, topi = dec_out.topk(1)
            dec_out_v, dec_out_i = dec_out.topk(1)
            dec_attn_v, dec_attn_i = dec_attn.topk(1)

            for t in range(1, time_steps):
                dec_inp = self.word_embeddings(topi.squeeze().detach())
                cur_dec_out, dec_hid, cur_dec_attn = self.decoder(
                    dec_inp,
                    dec_hid,
                    encoder_output,
                    src_word_embeds,
                    src_mask,
                    is_train,
                )
                cur_dec_out = cur_dec_out.view(-1, len(self.word_vocab))
                if copy_on:
                    cur_dec_out.data.masked_fill_(trg_vocab_mask.data, -float("inf"))
                cur_dec_out = F.log_softmax(cur_dec_out, dim=-1)
                topv, topi = cur_dec_out.topk(1)
                cur_dec_out_v, cur_dec_out_i = cur_dec_out.topk(1)
                dec_out_i = torch.cat((dec_out_i, cur_dec_out_i), 1)
                cur_dec_attn_v, cur_dec_attn_i = cur_dec_attn.topk(1)
                dec_attn_i = torch.cat((dec_attn_i, cur_dec_attn_i), 1)

        if is_train:
            print(dec_out.size())
            print(trg_words_seq.size())
            dec_out = dec_out.view(-1, len(self.word_vocab))
            target = trg_words_seq.view(-1, 1).squeeze()
            loss = self.criterion(dec_out, target)

            output["loss"] = loss

            output["description"] = partial(self.description, output=output)

            return output
        else:
            return dec_out_i, dec_attn_i

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, epoch: {}/{}:".format(
            output["loss"].item(), epoch, epoch_num,
        )

    def run_metrics(self, output):
        self.metrics(output["decode_result"], output["spo_gold"])
