import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import json
import os
import copy

from typing import Dict, List, Tuple, Set, Optional
from functools import partial

from lib.metrics import F1_triplet
from lib.models.abc_model import ABCModel

from lib.layer import Attention

activation = F.gelu


def seq_max_pool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq = seq - (1 - mask) * 1e10
    return torch.max(seq, 1)


def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = torch.unsqueeze(vec, 1)

    vec = torch.zeros_like(seq[:, :, :1]) + vec
    return torch.cat([seq, vec], 2)


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    batch_idxs = torch.arange(0, seq.size(0)).cuda()

    batch_idxs = torch.unsqueeze(batch_idxs, 1)

    idxs = torch.cat([batch_idxs, idxs], 1)

    res = []
    for i in range(idxs.size(0)):
        vec = seq[idxs[i][0], idxs[i][1], :]
        res.append(torch.unsqueeze(vec, 0))

    res = torch.cat(res)
    return res


class Seq2umt(ABCModel):
    def __init__(self, hyper):
        super(Seq2umt, self).__init__()
        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, "word_vocab.json"), "r")
        )
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, "relation_vocab.json"), "r")
        )
        self.id2word = {v: k for k, v in self.word_vocab.items()}
        self.id2rel = {v: k for k, v in self.relation_vocab.items()}
        self.BCE = torch.nn.BCEWithLogitsLoss()
        self.metrics = F1_triplet()
        self.get_metric = self.metrics.get_metric
        self.encoder = Encoder(
            len(self.word_vocab), self.hyper.emb_size, self.hyper.hidden_size
        )
        self.decoder = Decoder(
            len(self.word_vocab), self.hyper.emb_size, self.hyper.hidden_size
        )
        self.sos = nn.Embedding(num_embeddings=1, embedding_dim=self.hyper.emb_size)


    def masked_BCEloss(self, logits, gt, mask):
        loss = self.BCE(logits, gt)
        loss = torch.sum(loss.mul(mask)) / torch.sum(mask)
        return loss

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, epoch: {}/{}:".format(
            output["loss"].item(), epoch, epoch_num,
        )

    def run_metrics(self, output):
        self.metrics(output["decode_result"], output["spo_gold"])

    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        output = {}

        t = text_id = sample.T.cuda(self.gpu)
        B = t.size(0)
        sos = (
            self.sos(torch.tensor(0).cuda(self.gpu))
            .unsqueeze(0)
            .expand(B, -1)
            .unsqueeze(1)
        )

        o, (h_n, c_n) = self.encoder(t)

        result = self.decoder(sos, o, (h_n, c_n))


class Encoder(nn.Module):
    def __init__(self, word_dict_length, word_emb_size, lstm_hidden_size):
        super(Encoder, self).__init__()

        self.embeds = nn.Embedding(word_dict_length, word_emb_size).cuda()
        self.fc1_dropout = nn.Sequential(
            nn.Dropout(0.25).cuda(),  # drop 20% of the neuron
        ).cuda()

        self.lstm1 = nn.LSTM(
            input_size=word_emb_size,
            hidden_size=int(word_emb_size / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=word_emb_size,
            hidden_size=int(word_emb_size / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size * 2,  # 输入的深度
                out_channels=word_emb_size,  # filter 的个数，输出的高度
                kernel_size=3,  # filter的长与宽
                stride=1,  # 每隔多少步跳一下
                padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ).cuda(),
            nn.ReLU().cuda(),
        )

        # self.comb = nn.Linear(word_emb_size * 2, word_emb_size)

    def forward(self, t):
        mask = torch.gt(torch.unsqueeze(t, 2), 0).type(
            torch.cuda.FloatTensor
        )  # (batch_size,sent_len,1)
        mask.requires_grad = False
        t = self.embeds(t)

        t = self.fc1_dropout(t)

        t = t.mul(mask)  # (batch_size,sent_len,char_size)

        t, (h_n, c_n) = self.lstm1(t, None)
        t, (h_n, c_n) = self.lstm2(t, None)



        t_max, t_max_index = seq_max_pool([t, mask])

        t_dim = list(t.size())[-1]
        o = seq_and_vec([t, t_max])


        o = o.permute(0, 2, 1)
        o = self.conv1(o)

        o = o.permute(0, 2, 1)

        h_n = torch.cat((h_n[0], h_n[1]), dim=-1).unsqueeze(0)
        c_n = torch.cat((c_n[0], c_n[1]), dim=-1).unsqueeze(0)
        return o, (h_n, c_n)


class Decoder(nn.Module):
    def __init__(self, word_dict_length, word_emb_size, lstm_hidden_size):
        super(Decoder, self).__init__()

        # self.embeds = nn.Embedding(word_dict_length, word_emb_size).cuda()
        self.fc1_dropout = nn.Sequential(
            nn.Dropout(0.25).cuda(),  # drop 20% of the neuron
        ).cuda()

        self.lstm1 = nn.LSTM(
            input_size=word_emb_size,
            hidden_size=word_emb_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.use_attention = True
        self.attention = Attention(word_emb_size)
    

    def forward_step(self, input_var, hidden, encoder_outputs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        # embedded = self.embedding(input_var)
        # embedded = self.input_dropout(embedded)


        output, hidden = self.lstm1(input_var, hidden)
        # output, hidden = self.lstm2(output, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)
        # output = output.contiguous().view(-1, self.word_emb_size)

        # print(output.size()) # 64,1,128
        # print(attn.size())  # 64,1,300
        # print('ok!')
        # exit()

        return output, attn
        # predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        # return predicted_softmax, hidden, attn

    def forward(self, input, o, h):
        output, attn = self.forward_step(input, h, o)


class Rel2Ent(nn.Module):
    def __init__(self):
        super(Rel2Ent, self).__init__()
        pass

    def forward(self, decoder):
        pass


class Ent2Ent(nn.Module):
    def __init__(self):
        super(Ent2Ent, self).__init__()
        pass

    def forward(self, decoder):
        pass


class Ent2Rel(nn.Module):
    def __init__(self):
        super(Ent2Rel, self).__init__()
        pass

    def forward(self, decoder):
        pass
