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
from lib.config import seq_max_pool, seq_and_vec, seq_gather


activation = F.gelu


class Threetagging(ABCModel):
    def __init__(self, hyper):
        super(Threetagging, self).__init__()
        self.hyper = hyper
        self.order = hyper.order

        assert self.order == ["subject", "object", "predicate"]

        self.data_root = hyper.data_root
        self.gpu = hyper.gpu

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, "word_vocab.json"), "r")
        )

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
        self.metrics(
            output["decode_result"],
            output["spo_gold"],
            get_seq=lambda dic: (dic["object"], dic["predicate"], dic["subject"]),
        )

        # # rel only
        # self.metrics(output["decode_result"], output["spo_gold"], get_seq=lambda dic: (dic["predicate"],))

        # rel + head
        # self.metrics(output["decode_result"], output["spo_gold"], get_seq=lambda dic: (dic["predicate"], dic["subject"]))

    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        output = {}

        t = text_id = sample.T.cuda(self.gpu)
        length = sample.length
        mask = torch.gt(torch.unsqueeze(text_id, 2), 0).type(
            torch.cuda.FloatTensor
        )  # (batch_size,sent_len,1)
        mask.requires_grad = False

        rel_gt = sample.R_gt.cuda(self.gpu)

        head_gt1 = sample.S1.cuda(self.gpu)
        head_gt2 = sample.S2.cuda(self.gpu)

        tail_gt1 = sample.O1.cuda(self.gpu)
        tail_gt2 = sample.O2.cuda(self.gpu)

        o, t_max = self.encoder(t, length)

        if is_train:

            t_outs = self.decoder.train_forward(sample, o, t_max)

            out_map = dict(zip(self.order, (0, 1, 2)))

            rel_out = t_outs[out_map["predicate"]]
            head_out1, head_out2 = t_outs[out_map["subject"]]
            tail_out1, tail_out2 = t_outs[out_map["object"]]

            rel_loss = self.BCE(rel_out, rel_gt)
            head_loss = self.mBCE(head_out1, head_gt1, mask) + self.mBCE(
                head_out2, head_gt2, mask
            )
            tail_loss = self.mBCE(tail_out1, tail_gt1, mask) + self.mBCE(
                tail_out2, tail_gt2, mask
            )

            loss_sum = rel_loss + head_loss + tail_loss

            output["loss"] = loss_sum
        else:
            result = self.decoder.test_forward(sample, o, h)
            output["decode_result"] = result
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
            hidden_size=int(lstm_hidden_size / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=word_emb_size,
            hidden_size=int(lstm_hidden_size / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=lstm_hidden_size * 2,  # 输入的深度
                out_channels=lstm_hidden_size,  # filter 的个数，输出的高度
                kernel_size=3,  # filter的长与宽
                stride=1,  # 每隔多少步跳一下
                padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ).cuda(),
            nn.ReLU().cuda(),
        )

        # self.comb = nn.Linear(word_emb_size * 2, word_emb_size)

    def forward(self, t, length):
        mask = torch.gt(torch.unsqueeze(t, 2), 0).type(
            torch.cuda.FloatTensor
        )  # (batch_size,sent_len,1)
        mask.requires_grad = False
        SEQ = mask.size(1)

        emb = t = self.embeds(t)

        t = self.fc1_dropout(t)
        t = nn.utils.rnn.pack_padded_sequence(t, lengths=length, batch_first=True)
        # t = t.mul(mask)  # (batch_size,sent_len,char_size)

        t1, (h_n, c_n) = self.lstm1(t, None)
        # t2, (h_n, c_n) = self.lstm2(t1, None)
        t1, _ = nn.utils.rnn.pad_packed_sequence(t1, batch_first=True, total_length=SEQ)

        t_max, t_max_index = seq_max_pool([t1, mask])

        t_dim = list(t1.size())[-1]
        o = seq_and_vec([t1, t_max])

        o = o.permute(0, 2, 1)
        o = self.conv1(o)

        o = o.permute(0, 2, 1)

        h_n = torch.cat((h_n[0], h_n[1]), dim=-1).unsqueeze(0)
        c_n = torch.cat((c_n[0], c_n[1]), dim=-1).unsqueeze(0)
        return o, t_max


class Decoder(nn.Module):
    def __init__(self, hyper):
        super(Decoder, self).__init__()
        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu
        self.word_emb_size = self.hyper.emb_size
        self.hidden_size = self.hyper.hidden_size
        self.word_vocab = json.load(
            open(os.path.join(self.data_root, "word_vocab.json"), "r")
        )
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, "relation_vocab.json"), "r")
        )
        self.rel_num = len(self.relation_vocab)
        self.id2word = {v: k for k, v in self.word_vocab.items()}
        self.id2rel = {v: k for k, v in self.relation_vocab.items()}

        self.fc1_dropout = (nn.Dropout(0.25).cuda(),)  # drop 25% of the neuron

        self.lstm1 = nn.LSTM(
            input_size=self.word_emb_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.use_attention = True
        self.attention = Attention(self.word_emb_size)
        self.conv2_to_1_rel = nn.Conv1d(
            in_channels=self.hidden_size * 2,  # 输入的深度
            out_channels=self.word_emb_size,  # filter 的个数，输出的高度
            kernel_size=3,  # filter的长与宽
            stride=1,  # 每隔多少步跳一下
            padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
        )
        self.conv2_to_1_ent = nn.Conv1d(
            in_channels=self.hidden_size * 2,  # 输入的深度
            out_channels=self.word_emb_size,  # filter 的个数，输出的高度
            kernel_size=3,  # filter的长与宽
            stride=1,  # 每隔多少步跳一下
            padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
        )
        self.conv4_to_1_ent = nn.Conv1d(
            in_channels=self.hidden_size * 4,  # 输入的深度
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

        # order
        self.order = self.hyper.order

        self.state_map = {
            ("predicate", "subject", "object"): [
                self.sos2rel,
                self.rel2ent,
                self.ent2ent,
            ],
            ("predicate", "object", "subject"): [
                self.sos2rel,
                self.rel2ent,
                self.ent2ent,
            ],
            ("subject", "object", "predicate"): [
                self.sos2ent,
                self.ent2ent,
                self.ent2rel,
            ],
            ("subject", "predicate", "object"): [
                self.sos2ent,
                self.ent2rel,
                self.rel2ent,
            ],
            ("object", "subject", "predicate"): [
                self.sos2ent,
                self.ent2ent,
                self.ent2rel,
            ],
            ("object", "predicate", "subject"): [
                self.sos2ent,
                self.ent2rel,
                self.rel2ent,
            ],
        }[tuple(self.order)]

        self.decode_state_map = {
            ("predicate", "subject", "object"): [
                self._out2rel,
                self._out2entity,
                self._out2entity,
            ],
            ("predicate", "object", "subject"): [
                self._out2rel,
                self._out2entity,
                self._out2entity,
            ],
            ("subject", "object", "predicate"): [
                self._out2entity,
                self._out2entity,
                self._out2rel,
            ],
            ("subject", "predicate", "object"): [
                self._out2entity,
                self._out2rel,
                self._out2entity,
            ],
            ("object", "subject", "predicate"): [
                self._out2entity,
                self._out2entity,
                self._out2rel,
            ],
            ("object", "predicate", "subject"): [
                self._out2entity,
                self._out2rel,
                self._out2entity,
            ],
        }[tuple(self.order)]

    def forward_step(self, input_var, hidden, encoder_outputs):

        output, hidden = self.lstm1(input_var, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        return output, attn, hidden

    def to_rel(self, input, h, encoder_o, mask):
        output, attn, h = self.forward_step(input, h, encoder_o)
        new_encoder_o = seq_and_vec([encoder_o, output.squeeze(1)])

        new_encoder_o = new_encoder_o.permute(0, 2, 1)
        new_encoder_o = self.conv2_to_1_rel(new_encoder_o)
        new_encoder_o = new_encoder_o.permute(0, 2, 1)

        output = activation(new_encoder_o)
        output = self.rel(output)
        output, _ = seq_max_pool([output, mask])

        return output, h, new_encoder_o, attn

    def to_ent(self, input, h, encoder_o, mask):
        # TODO mask
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

    def sos2ent(self, sos, encoder_o, h, mask):
        # TODO: test
        input = sos
        out, h, new_encoder_o, attn = self.to_ent(input, h, encoder_o, mask)
        return out, h, new_encoder_o

    def no2ent(self, inp, encoder_o, t_max, mask):
        inp = None
        h = None

        # t_max, t_max_index = seq_max_pool([encoder_o, mask])

        t_dim = list(encoder_o.size())[-1]
        new_encoder_o = seq_and_vec([encoder_o, t_max])

        new_encoder_o = new_encoder_o.permute(0, 2, 1)
        new_encoder_o = self.conv2_to_1_ent(new_encoder_o)

        new_encoder_o = new_encoder_o.permute(0, 2, 1)

        ent1 = self.ent1(new_encoder_o).squeeze(2)
        ent2 = self.ent2(new_encoder_o).squeeze(2)
        output = ent1, ent2

        return output, t_max, new_encoder_o, mask

    def ent2ent_without_rnn(self, inp, encoder_o, t_max, mask):
        k1, k2 = inp
        k1 = seq_gather([encoder_o, k1])

        k2 = seq_gather([encoder_o, k2])

        k = torch.cat([k1, k2], 1)
        print(k.size())  # 64, 256
        print(encoder_o.size())  # 64, 300, 128
        print(t_max.size())  # 64, 64
        new_encoder_o = seq_and_vec([encoder_o, t_max])
        new_encoder_o = seq_and_vec([new_encoder_o, k])
        new_encoder_o = encoder_o.permute(0, 2, 1)
        print(new_encoder_o.size())
        new_encoder_o = self.conv4_to_1_ent(new_encoder_o)
        print(new_encoder_o.size())
        print("cnn ok")

        new_encoder_o = new_encoder_o.permute(0, 2, 1)

        ent1 = self.ent1(new_encoder_o).squeeze(2)
        ent2 = self.ent2(new_encoder_o).squeeze(2)
        output = ent1, ent2

        return output, t_max, new_encoder_o, mask

    def ent2rel(self, t_in, encoder_o, h, mask):
        # TODO: test
        k1, k2 = t_in
        k1 = seq_gather([encoder_o, k1])
        k2 = seq_gather([encoder_o, k2])

        # TODO
        # print(k1.size())
        # k = torch.cat([k1,k2],1)
        input = k1 + k2
        input = input.unsqueeze(1)
        out, h, new_encoder_o, attn = self.to_rel(input, h, encoder_o, mask)
        out = out.squeeze(1)

        return out, h, new_encoder_o

    def sos2rel(self, sos, encoder_o, h, mask):
        # t1
        input = sos
        t1_out, h, new_encoder_o, attn = self.to_rel(input, h, encoder_o, mask)
        t1_out = t1_out.squeeze(1)

        return t1_out, h, new_encoder_o

    def rel2ent(self, t2_in, encoder_o, h, mask):
        # t2
        input = self.rel_emb(t2_in)
        input = input.unsqueeze(1)
        t2_out, h, new_encoder_o, attn = self.to_ent(input, h, encoder_o, mask)
        return t2_out, h, new_encoder_o

    def ent2ent(self, t3_in, encoder_o, h, mask):
        # t3
        k1, k2 = t3_in
        k1 = seq_gather([encoder_o, k1])
        k2 = seq_gather([encoder_o, k2])

        # TODO
        # print(k1.size())
        # k = torch.cat([k1,k2],1)
        input = k1 + k2
        input = input.unsqueeze(1)
        t3_out, h, new_encoder_o, attn = self.to_ent(input, h, encoder_o, mask)
        return t3_out, h, new_encoder_o

    def train_forward(self, sample, encoder_o, t_max):
        B = sample.T.size(0)
        sos = (
            self.sos(torch.tensor(0).cuda(self.gpu))
            .unsqueeze(0)
            .expand(B, -1)
            .unsqueeze(1)
        )
        t1_in = sos

        r_in = sample.R_in.cuda(self.gpu)
        s_in = sample.S_K1_in.cuda(self.gpu), sample.S_K2_in.cuda(self.gpu)
        o_in = sample.O_K1_in.cuda(self.gpu), sample.O_K2_in.cuda(self.gpu)

        in_tuple = (r_in, s_in, o_in)

        in_map = {"predicate": 0, "subject": 1, "object": 2}

        t2_in = in_tuple[in_map[self.order[0]]]
        t3_in = in_tuple[in_map[self.order[1]]]

        t = text_id = sample.T.cuda(self.gpu)
        mask = torch.gt(torch.unsqueeze(text_id, 2), 0).type(
            torch.cuda.FloatTensor
        )  # (batch_size,sent_len,1)
        mask.requires_grad = False
        # only using encoder_o
        t1_out, t_max, new_encoder_o, mask = self.no2ent(None, encoder_o, t_max, mask)
        t2_out, t_max, new_encoder_o, mask = self.ent2ent_without_rnn(
            t2_in, new_encoder_o, t_max, mask
        )
        print(new_encoder_o.size())
        exit()
        t3_out = None
        # t1_out, h, new_encoder_o = self.state_map[0](t1_in, encoder_o, h, mask)
        # t2_out, h, new_encoder_o = self.state_map[1](t2_in, new_encoder_o, h, mask)
        # t3_out, h, new_encoder_o = self.state_map[2](t3_in, new_encoder_o, h, mask)

        return t1_out, t2_out, t3_out

    def test_forward(self, sample, encoder_o, decoder_h) -> List[List[Dict[str, str]]]:
        text_id = sample.T.cuda(self.gpu)
        mask = (
            torch.gt(torch.unsqueeze(text_id, 2), 0).float().cuda(self.gpu)
        )  # (batch_size,sent_len,1)
        mask.requires_grad = False
        text = text_id.tolist()
        text = [[self.id2word[c] for c in sent] for sent in text]
        result = []
        # result_t1 = []
        # result_t2 = []
        for i, sent in enumerate(text):

            h, c = (
                decoder_h[0][:, i, :].unsqueeze(1).contiguous(),
                decoder_h[1][:, i, :].unsqueeze(1).contiguous(),
            )
            triplets = self.extract_items(
                sent,
                text_id[i, :].unsqueeze(0).contiguous(),
                mask[i, :].unsqueeze(0).contiguous(),
                encoder_o[i, :, :].unsqueeze(0).contiguous(),
                (h, c),
            )
            result.append(triplets)
        return result

    def _out2entity(self, sent, out):
        # extract t2 result from outs
        out1, out2 = out
        _subject_name = []
        _subject_id = []
        for i, _kk1 in enumerate(out1.squeeze().tolist()):
            if _kk1 > 0:
                for j, _kk2 in enumerate(out2.squeeze().tolist()[i:]):
                    if _kk2 > 0:
                        _subject_name.append(self.hyper.join(sent[i : i + j + 1]))
                        _subject_id.append((i, i + j))
                        break
        return _subject_id, _subject_name

    def _out2rel(self, sent, out):
        rels = out.squeeze().tolist()

        rels_id = [i for i, r in enumerate(rels) if r > 0]
        rels_name = [self.id2rel[i] for i, r in enumerate(rels) if r > 0]
        return rels_id, rels_name

    def _out2in(self, out, order_i):
        if order_i == "predicate":
            inp = torch.LongTensor([out]).cuda(self.gpu)
        else:
            s1, s2 = out
            inp = (
                torch.LongTensor([[s1]]).cuda(self.gpu),
                torch.LongTensor([[s2]]).cuda(self.gpu),
            )
        return inp

    def extract_items(
        self, sent, text_id, mask, encoder_o, t1_h
    ) -> List[Dict[str, str]]:

        R = []

        sos = self.sos(torch.tensor(0).cuda(self.gpu)).unsqueeze(0).unsqueeze(1)

        t1_out, t1_h, t1_encoder_o = self.state_map[0](sos, encoder_o, t1_h, mask)

        t1_id, t1_name = self.decode_state_map[0](sent, t1_out)

        for id1, name1 in zip(t1_id, t1_name):

            t2_in = self._out2in(id1, self.order[0])
            t2_out, t2_h, t2_encoder_o = self.state_map[1](
                t2_in, t1_encoder_o, t1_h, mask
            )
            t2_id, t2_name = self.decode_state_map[1](sent, t2_out)

            if len(t2_name) > 0:
                for id2, name2 in zip(t2_id, t2_name):
                    t3_in = self._out2in(id2, self.order[1])
                    t3_out, _, _ = self.state_map[2](t3_in, t2_encoder_o, t2_h, mask)

                    _, t3_name = self.decode_state_map[2](sent, t3_out)

                    for name3 in t3_name:
                        R.append(
                            {
                                self.order[0]: name1,
                                self.order[1]: name2,
                                self.order[2]: name3,
                            }
                        )

        return R

    def forward(self, sample, encoder_o, h, is_train):
        pass
