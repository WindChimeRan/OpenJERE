import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os
import copy

from typing import Dict, List, Tuple, Set, Optional
from functools import partial

# from torchcrf import CRF
from TorchCRF import CRF

import torchsnooper
from pytorch_memlab import profile


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
                                    embedding_dim=hyper.bio_emb_size)

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
            hyper.hidden_size + hyper.bio_emb_size, len(self.relation_vocab))
        self.rel_linear_a = nn.Linear(
            hyper.hidden_size + hyper.bio_emb_size, hyper.hidden_size)
        self.rel_linear_b = nn.Linear(
            hyper.hidden_size, len(self.relation_vocab))

        self.entity_linear_1 = nn.Linear(
            hyper.hidden_size * 2 + hyper.bio_emb_size, hyper.hidden_size)
        self.entity_linear_2 = nn.Linear(hyper.hidden_size, 1)

        self.cat_linear = nn.Linear(
            hyper.hidden_size + hyper.bio_emb_size, hyper.hidden_size)

        self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab) - 1)
        self.ce = nn.CrossEntropyLoss(reduction='none')

    # @profile
    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        tokens = sample.tokens_id.cuda(self.gpu)
        seq_gold = sample.seq_id.cuda(self.gpu)
        bio_gold = sample.bio_id.cuda(self.gpu)
        length = sample.length

        text_list = sample.text
        spo_gold = sample.spo_gold

        mask = tokens != self.word_vocab['<pad>']  # batch x seq

        mask_decode = sample.mask_decode.cuda(self.gpu)

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

        cat_o = torch.cat((o, tag_emb), dim=2)

        o_hid_size = cat_o.size(-1)

        hidden_idx = torch.tensor(
            list(map(lambda x: x-1, length)), dtype=torch.long).cuda(self.gpu)
        hidden_idx = torch.zeros_like(tokens).scatter_(
            1, hidden_idx.unsqueeze(1), 1).to(torch.uint8)

        h = o[hidden_idx]
        h = h.unsqueeze(1).expand(-1, self.hyper.max_text_len, -
                                  1).contiguous().view(1, -1, self.hyper.hidden_size).contiguous()
        decoder_sos = cat_o.view(-1, o_hid_size)
        decoder_input = self.cat_linear(self.activation(decoder_sos))

        copy_o = cat_o.unsqueeze(0).expand(self.hyper.max_text_len, -1, -1, -1).contiguous(
        ).view(-1, self.hyper.max_text_len, self.hyper.hidden_size + self.hyper.bio_emb_size)

        B, L, _ = copy_o.size()
        decoder_loss = 0
        if is_train:
            seq_gold = seq_gold.view(-1, 2 * self.hyper.max_decode_len + 1)
            for i in range(self.hyper.max_decode_len * 2 + 1):

                decoder_input = decoder_input.squeeze()
                decoder_output, h, output_logits = self._decode_step(
                    i, h, decoder_input, copy_o)

                # use groundtruth
                decoder_input = seq_gold[:, i]
                if i % 2 == 0:
                    decoder_input = self.relation_emb(decoder_input)
                else:

                    copy_index = torch.zeros((B, L)).scatter_(
                        1, decoder_input.unsqueeze(1).cpu(), 1).to(torch.uint8)
                    decoder_input = self.cat_linear(
                        self.activation(copy_o[copy_index]))

                step_loss = self.masked_NLLloss(
                    mask_decode[:, :, i], output_logits, seq_gold[:, i])

                decoder_loss += step_loss.sum()

            decoder_loss = decoder_loss / mask_decode.sum()
        # if evaluation
        else:
            for i in range(self.hyper.max_decode_len * 2 + 1):

                decoder_input = decoder_input.squeeze()
                decoder_output, h, output_logits = self._decode_step(
                    i, h, decoder_input, copy_o)

                decoder_input = torch.argmax(output_logits, dim=1).detach()
                if i % 2 == 0:
                    decoder_input = self.relation_emb(decoder_input)
                else:

                    copy_index = torch.zeros((B, L)).scatter_(
                        1, decoder_input.unsqueeze(1).cpu(), 1).to(torch.uint8)
                    decoder_input = self.cat_linear(
                        self.activation(copy_o[copy_index]))

        loss = crf_loss + decoder_loss
        output['crf_loss'] = crf_loss
        output['decoder_loss'] = decoder_loss
        output['loss'] = loss

        output['description'] = partial(self.description, output=output)

        if not is_train:
            # TODO inference
            pass

        return output

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, L_crf: {:.2f}, L_decode: {:.2f}, epoch: {}/{}:".format(
            output['loss'].item(), output['crf_loss'].item(),
            output['decoder_loss'].item(), epoch, epoch_num)

    def _decode_step(self, t: int, decoder_state, decoder_input, o):

        decoder_input = decoder_input.unsqueeze(dim=1)
        decoder_output, decoder_state = self.decoder(
            decoder_input, decoder_state)

        if t % 2 == 0:
            # relation logits
            output_logits = self.rel_linear_b(decoder_state.squeeze())
        else:
            output_logits = torch.cat(
                (decoder_state.permute(1, 0, 2).expand(-1, self.hyper.max_text_len, -1), o), dim=2)  # hidden 300 + 300 + 50
            output_logits = self.entity_linear_2(self.activation(
                self.entity_linear_1(self.activation(output_logits)))).squeeze()

        return decoder_output, decoder_state, output_logits

    def masked_NLLloss(self, mask, output_logits, seq_gold):

        loss = self.ce(output_logits, seq_gold).masked_select(mask.view(-1))
        return loss
