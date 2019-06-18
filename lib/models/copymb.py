import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os
import copy

from typing import Dict, List, Tuple, Set, Optional
from functools import partial

from torchcrf import CRF

import torchsnooper


class CopyMB(nn.Module):
    def __init__(self, hyper) -> None:
        super(CopyMB, self).__init__()

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
            self.decoder = nn.GRU(hyper.emb_size,
                                  hyper.hidden_size,
                                  bidirectional=False,
                                  batch_first=True)
        elif hyper.cell_name == 'lstm':
            self.encoder = nn.LSTM(hyper.emb_size,
                                   hyper.hidden_size,
                                   bidirectional=True,
                                   batch_first=True)
            self.decoder = nn.LSTM(hyper.emb_size,
                                   hyper.hidden_size,
                                   bidirectional=False,
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
        # here the 'N' relation is used as <eos> in standard seq2seq
        self.rel_linear_1 = nn.Linear(
            hyper.hidden_size + hyper.rel_emb_size, len(self.relation_vocab))
        self.rel_linear_a = nn.Linear(
            hyper.hidden_size + hyper.rel_emb_size, hyper.hidden_size)
        self.rel_linear_b = nn.Linear(
            hyper.hidden_size, len(self.relation_vocab))

        self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab) - 1)

    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        tokens = sample.tokens_id.cuda(self.gpu)
        seq_gold = sample.seq_id.cuda(self.gpu)
        bio_gold = sample.bio_id.cuda(self.gpu)
        length = sample.length

        text_list = sample.text
        spo_gold = sample.spo_gold

        mask = tokens != self.word_vocab['<pad>']  # batch x seq

        embedded = self.word_embeddings(tokens)
        o, _ = self.encoder(embedded)

        o = (lambda a: sum(a) / 2)(torch.split(o,
                                               self.hyper.hidden_size,
                                               dim=2))

        emi = self.emission(o)

        output = {}

        crf_loss = 0

        if is_train:
            crf_loss = -self.tagger(emi, bio_gold, mask=mask, reduction='mean')
        else:
            decoded_tag = self.tagger.decode(emissions=emi, mask=mask)
            temp_tag = copy.deepcopy(decoded_tag)
            for line in temp_tag:
                line.extend([self.bio_vocab['<pad>']] *
                            (self.hyper.max_text_len - len(line)))
            bio_gold = torch.tensor(temp_tag).cuda(self.gpu)

        tag_emb = self.bio_emb(bio_gold)

        o = torch.cat((o, tag_emb), dim=2)

        o_hid_size = o.size(-1)

        hidden_idx = torch.tensor(list(map(lambda x: x-1, length)), dtype=torch.long).cuda(self.gpu)
        hidden_idx = torch.zeros_like(tokens).scatter_(
            1, hidden_idx.unsqueeze(1), 1).to(torch.uint8)

        # mask_for_hidden = torch.zeros_like(mask).unsqueeze(2)
        # for i, idx in enumerate(hidden_idx):
        #     mask_for_hidden[i, idx] = 1

        h = o[hidden_idx]
        # h = torch.masked_select(o, mask=mask_for_hidden).view(-1, o_hid_size)
        # h = h.unsqueeze(1).expand(-1, self.hyper.max_text_len, -1).contiguous().view(-1, o_hid_size)
        seq_gold = seq_gold.view(-1, 2 * self.hyper.max_decode_len + 1)
        decoder_input = o.view(-1, o_hid_size)

        print(h.size())
        # indirect
        # h = self.rel_linear_a(h)
        # decoder_o, decoder_state = self.decoder(decoder_input, h)
        # print(decoder_o.size())
        exit()
        # forward copymb decoder
        # if train
        # TODO
        # if not train
        # TODO
        # u = self.activation(self.selection_u(o)).unsqueeze(1)
        # v = self.activation(self.selection_v(o)).unsqueeze(2)
        # u = u + torch.zeros_like(v)
        # v = v + torch.zeros_like(u)
        # uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))
        # selection_logits = torch.einsum('bijh,rh->birj', uv,
        #                                 self.relation_emb.weight)

        # if not is_train:
        #     output['selection_triplets'] = self.inference(
        #         mask, text_list, decoded_tag, selection_logits)
        #     output['spo_gold'] = spo_gold

        # selection_loss = 0
        # if is_train:
        #     selection_loss = self.masked_BCEloss(mask, selection_logits,
        #                                          selection_gold)

        # loss = crf_loss + selection_loss
        # output['crf_loss'] = crf_loss
        # output['selection_loss'] = selection_loss
        # output['loss'] = loss

        # output['description'] = partial(self.description, output=output)
        return output

    def masked_BCEloss(self, mask, selection_logits, selection_gold):
        pass
