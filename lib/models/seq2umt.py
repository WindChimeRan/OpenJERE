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

from lib.layer import Attention, MaskedBCE


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
        # self.relation_vocab = json.load(
        #     open(os.path.join(self.data_root, "relation_vocab.json"), "r")
        # )
        # self.id2word = {v: k for k, v in self.word_vocab.items()}
        # self.id2rel = {v: k for k, v in self.relation_vocab.items()}
        self.mBCE = MaskedBCE()
        self.BCE = torch.nn.BCEWithLogitsLoss()
        self.metrics = F1_triplet()
        self.get_metric = self.metrics.get_metric
        self.encoder = Encoder(
            len(self.word_vocab), self.hyper.emb_size, self.hyper.hidden_size
        )
        self.decoder = Decoder(hyper)
        self.sos = nn.Embedding(num_embeddings=1, embedding_dim=self.hyper.emb_size)

    # def masked_BCEloss(self, logits, gt, mask):
    #     loss = self.BCE(logits, gt)
    #     loss = torch.sum(loss.mul(mask)) / torch.sum(mask)
    #     return loss

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, epoch: {}/{}:".format(
            output["loss"].item(), epoch, epoch_num,
        )

    def run_metrics(self, output):
        # # whole triplet 
        # self.metrics(output["decode_result"], output["spo_gold"])

        # # rel only
        # self.metrics(output["decode_result"], output["spo_gold"], get_seq=lambda dic: (dic["predicate"],))

        # rel + head
        self.metrics(output["decode_result"], output["spo_gold"], get_seq=lambda dic: (dic["predicate"], dic["subject"]))

    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        output = {}

        t = text_id = sample.T.cuda(self.gpu)
        mask = torch.gt(torch.unsqueeze(text_id, 2), 0).type(
            torch.cuda.FloatTensor
        )  # (batch_size,sent_len,1)
        mask.requires_grad = False

        t1_gt = sample.R_gt.cuda(self.gpu)

        t2_gt1 = sample.S1.cuda(self.gpu)
        t2_gt2 = sample.S2.cuda(self.gpu)

        t3_gt1 = sample.O1.cuda(self.gpu)
        t3_gt2 = sample.O2.cuda(self.gpu)

        o, h = self.encoder(t)
        if is_train:

            t1_out, t2_out, t3_out = self.decoder.train_forward(sample, o, h)
            (t2_out1, t2_out2), (t3_out1, t3_out2) = t2_out, t3_out

            t1_loss = self.BCE(t1_out, t1_gt)
            t2_loss = self.mBCE(t2_out1, t2_gt1, mask) + self.mBCE(
                t2_out2, t2_gt2, mask
            )
            t3_loss = self.mBCE(t3_out1, t3_gt1, mask) + self.mBCE(
                t3_out2, t3_gt2, mask
            )

            loss_sum = t1_loss + t2_loss + t3_loss
            output["loss"] = loss_sum
        else:
            # TODO
            result, result_t1, result_t2 = self.decoder.test_forward(sample, o, h)
            output["decode_result"] = result
            output["decode_t1"] = result_t1
            output["decode_t2"] = result_t2
            output["spo_gold"] = sample.spo_gold
        output["description"] = partial(self.description, output=output)
        return output


class Encoder(nn.Module):
    def __init__(self, word_dict_length, word_emb_size, lstm_hidden_size):
        super(Encoder, self).__init__()

        self.embeds = nn.Embedding(word_dict_length, word_emb_size)
        self.fc1_dropout = nn.Sequential(nn.Dropout(0.25),)  # drop 20% of the neuron

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
        emb = t = self.embeds(t)

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
    def __init__(self, hyper):
        super(Decoder, self).__init__()
        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu
        self.word_emb_size = self.hyper.emb_size
        self.word_vocab = json.load(
            open(os.path.join(self.data_root, "word_vocab.json"), "r")
        )
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, "relation_vocab.json"), "r")
        )
        self.rel_num = len(self.relation_vocab)
        self.id2word = {v: k for k, v in self.word_vocab.items()}
        self.id2rel = {v: k for k, v in self.relation_vocab.items()}

        self.fc1_dropout = nn.Sequential(
            nn.Dropout(0.25).cuda(),  # drop 20% of the neuron
        ).cuda()

        self.lstm1 = nn.LSTM(
            input_size=self.word_emb_size,
            hidden_size=self.word_emb_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.use_attention = True
        self.attention = Attention(self.word_emb_size)
        self.conv2_to_1_rel = nn.Conv1d(
                in_channels=self.word_emb_size * 2,  # 输入的深度
                out_channels=self.word_emb_size,  # filter 的个数，输出的高度
                kernel_size=3,  # filter的长与宽
                stride=1,  # 每隔多少步跳一下
                padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            )
        self.conv2_to_1_ent = nn.Conv1d(
                in_channels=self.word_emb_size * 2,  # 输入的深度
                out_channels=self.word_emb_size,  # filter 的个数，输出的高度
                kernel_size=3,  # filter的长与宽
                stride=1,  # 每隔多少步跳一下
                padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            )
        self.sos = nn.Embedding(num_embeddings=1, embedding_dim=self.word_emb_size)
        self.rel_emb = nn.Embedding(
            num_embeddings=self.rel_num, embedding_dim=self.word_emb_size
        )

        self.rel = nn.Linear(self.word_emb_size, self.rel_num)
        self.ent1 = nn.Linear(self.word_emb_size, 1)
        self.ent2 = nn.Linear(self.word_emb_size, 1)
        # self.rel_tag = nn.Linear(self.word_emb_size * 2, self.rel_num)

        self.t1_op = self.to_rel
        self.t2_op = self.to_ent
        self.t3_op = self.to_ent

    def forward_step(self, input_var, hidden, encoder_outputs):

        output, hidden = self.lstm1(input_var, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        return output, attn, hidden

    def to_rel(self, input, h, encoder_o, mask):
        output, attn, h = self.forward_step(input, h, encoder_o)
        ew_encoder_o = seq_and_vec([encoder_o, output.squeeze(1)])
        new_encoder_o = new_encoder_o.permute(0, 2, 1)
        new_encoder_o = self.conv2_to_1_rel(new_encoder_o)
        new_encoder_o = new_encoder_o.permute(0, 2, 1)

        output = activation(new_encoder_o)
        output = self.rel(output)
        output, _ = seq_max_pool([output, mask])

        return output, h, new_encoder_o, attn

    def to_ent(self, input, h, encoder_o, mask):
        output, attn, h = self.forward_step(input, h, encoder_o)
        output = output.squeeze(1)

        new_encoder_o = seq_and_vec([encoder_o, output])

        new_encoder_o = new_encoder_o.permute(0, 2, 1)
        new_encoder_o = self.conv2_to_1_ent(new_encoder_o)

        new_encoder_o = new_encoder_o.permute(0, 2, 1)

        output = activation(new_encoder_o)
        ent1 = self.ent1(output).squeeze(2)
        ent2 = self.ent2(output).squeeze(2)

        output = ent1, ent2

        return output, h, new_encoder_o, attn

    def t1(self, sos, encoder_o, h, mask):
        # sos - rel
        input = sos
        t1_out, h, new_encoder_o, attn = self.t1_op(input, h, encoder_o, mask)
        t1_out = t1_out.squeeze(1)

        return t1_out, h, new_encoder_o

    def t2(self, t2_in, encoder_o, h, mask):
        # rel - head
        input = self.rel_emb(t2_in)
        input = input.unsqueeze(1)
        t2_out, h, new_encoder_o, attn = self.t2_op(input, h, encoder_o, mask)
        return t2_out, h, new_encoder_o

    def t3(self, t3_in, encoder_o, h, mask):
        # head - tail
        k1, k2 = t3_in
        k1 = seq_gather([encoder_o, k1])
        k2 = seq_gather([encoder_o, k2])

        # TODO
        # print(k1.size())
        # k = torch.cat([k1,k2],1)
        input = k1 + k2
        input = input.unsqueeze(1)
        t3_out, h, new_encoder_o, attn = self.t3_op(input, h, encoder_o, mask)
        return t3_out, h, new_encoder_o

    def train_forward(self, sample, encoder_o, h):
        B = sample.T.size(0)
        sos = (
            self.sos(torch.tensor(0).cuda(self.gpu))
            .unsqueeze(0)
            .expand(B, -1)
            .unsqueeze(1)
        )
        t1_in = sos
        t2_in = sample.R_in.cuda(self.gpu)
        t3_in = sample.K1.cuda(self.gpu), sample.K2.cuda(self.gpu)
        t = text_id = sample.T.cuda(self.gpu)
        mask = torch.gt(torch.unsqueeze(text_id, 2), 0).type(
            torch.cuda.FloatTensor
        )  # (batch_size,sent_len,1)
        mask.requires_grad = False
        # only using encoder_o
        t1_out, h, new_encoder_o = self.t1(t1_in, encoder_o, h, mask)
        t2_out, h, new_encoder_o = self.t2(t2_in, new_encoder_o, h, mask)
        t3_out, h, new_encoder_o = self.t3(t3_in, new_encoder_o, h, mask)

        return t1_out, t2_out, t3_out

    def test_forward(self, sample, encoder_o, decoder_h) -> List[List[Dict[str, str]]]:
        text_id = sample.T.cuda(self.gpu)
        mask = torch.gt(torch.unsqueeze(text_id, 2), 0).type(
            torch.cuda.FloatTensor
        )  # (batch_size,sent_len,1)
        mask.requires_grad = False
        text = text_id.tolist()
        text = [[self.id2word[c] for c in sent] for sent in text]
        result = []
        result_t1 = []
        result_t2 = []
        for i, sent in enumerate(text):

            h, c = (
                decoder_h[0][:, i, :].unsqueeze(1).contiguous(),
                decoder_h[1][:, i, :].unsqueeze(1).contiguous(),
            )
            # TODO
            triplets, R_t1, R_t2 = self.extract_items(
                sent,
                text_id[i, :].unsqueeze(0).contiguous(),
                mask,
                encoder_o[i, :, :].unsqueeze(0).contiguous(),
                (h, c),
            )
            result.append(triplets)
            result_t1.append(R_t1)
            result_t2.append(R_t2)
        return result, result_t1, result_t2

    def _pos_2_entity(self, sent, t2_out):
        # extract t2 result from outs
        t2_out1, t2_out2 = t2_out
        _subject_name = []
        _subject_id = []
        for i, _kk1 in enumerate(t2_out1.squeeze().tolist()):
            if _kk1 > 0:
                for j, _kk2 in enumerate(t2_out2.squeeze().tolist()[i:]):
                    if _kk2 > 0:
                        _subject_name.append(self.hyper.join(sent[i : i + j + 1]))
                        _subject_id.append((i, i + j))
                        break
        return _subject_id, _subject_name

    def extract_items(
        self, sent, text_id, mask, encoder_o, h
    ) -> List[Dict[str, str]]:

        R = []
        R_t1 = []
        R_t2 = []

        sos = self.sos(torch.tensor(0).cuda(self.gpu)).unsqueeze(0).unsqueeze(1)
        # t1_out, h = self.t1(sos, encoder_o, h)
        t1_out, h, t1_encoder_o = self.t1(sos, encoder_o, h, mask)

        # t1_out, h, new_encoder_o = self.t1(t1_in, encoder_o, h, mask)
        # t2_out, h, new_encoder_o = self.t2(t2_in, new_encoder_o, h, mask)
        # t3_out, h, new_encoder_o = self.t3(t3_in, new_encoder_o, h, mask)

        # t1
        rels = t1_out.squeeze().tolist()
        rels_id = [i for i, r in enumerate(rels) if r > 0]
        rels_name = [self.id2rel[i] for i, r in enumerate(rels) if r > 0]

        for r_id, r_name in zip(rels_id, rels_name):
            # t2
            t2_in = torch.LongTensor([r_id]).cuda(self.gpu)
            # t2_out, h = self.t2(t2_in, encoder_o, h)
            t2_out, t2_h, t2_encoder_o = self.t2(t2_in, t1_encoder_o, h, mask)
            _subject_id, _subject_name = self._pos_2_entity(sent, t2_out)

            R_t1.append({"predicate": r_name})

            if len(_subject_name) > 0:
                for (s1, s2), s_name in zip(_subject_id, _subject_name):
                    t3_in = (
                        torch.LongTensor([[s1]]).cuda(self.gpu),
                        torch.LongTensor([[s2]]).cuda(self.gpu),
                    )
                    # t3_out, h = self.t3(t3_in, encoder_o, h)
                    t3_out, t3_h, t3_encoder_o = self.t2(t3_in, t2_encoder_o, t2_h, mask)

                    _object_id, _object_name = self._pos_2_entity(sent, t3_out)

                    R_t2.append({"subject": s_name, "predicate": r_name})
                    for o_name in _object_name:
                        R.append(
                            {"subject": s_name, "predicate": r_name, "object": o_name,}
                        )
        return R, R_t1, R_t2

    def forward(self, sample, encoder_o, h, is_train):
        pass
