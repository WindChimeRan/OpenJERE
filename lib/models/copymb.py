import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os
import copy

import torchsnooper

from typing import Dict, List, Tuple, Set, Optional
from functools import partial
from pytorch_memlab import profile

from lib.tagger.crf import CRF
from lib.metrics import F1_triplet


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
        self.metrics = F1_triplet()
        self.get_metric = self.metrics.get_metric

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
            1, hidden_idx.unsqueeze(1), 1).bool()

        h = o[hidden_idx]
        h = h.unsqueeze(1).expand(-1, self.hyper.max_text_len, -
                                  1).contiguous().view(1, -1, self.hyper.hidden_size).contiguous()
        decoder_sos = cat_o.view(-1, o_hid_size)
        decoder_input = self.cat_linear(self.activation(decoder_sos))

        copy_o = cat_o.unsqueeze(0).expand(self.hyper.max_text_len, -1, -1, -1).contiguous(
        ).view(-1, self.hyper.max_text_len, self.hyper.hidden_size + self.hyper.bio_emb_size)

        B_stacked, L, _ = copy_o.size()
        # print(B, L)
        decoder_loss = 0
        decoder_result = []
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

                    copy_index = torch.zeros((B_stacked, L)).scatter_(
                        1, decoder_input.unsqueeze(1).cpu(), 1).bool()
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

                idx = torch.argmax(output_logits, dim=1).detach()
                if i % 2 == 0:
                    decoder_input = self.relation_emb(idx)
                    decoder_result.append(idx.cpu())
                else:
                    copy_index = torch.zeros((B_stacked, L)).scatter_(
                        1, idx.unsqueeze(1).cpu(), 1).bool()
                    decoder_result.append(copy_index.long())
                    decoder_input = self.cat_linear(
                        self.activation(copy_o[copy_index]))

        loss = crf_loss + decoder_loss
        output['crf_loss'] = crf_loss
        output['decoder_loss'] = decoder_loss
        output['loss'] = loss

        output['description'] = partial(self.description, output=output)

        if not is_train:
            output['spo_gold'] = spo_gold
            decoder_result = self.decodeid2triplet(decoder_result, tokens, bio_gold.tolist(), mask)
            output['decode_result'] = decoder_result

        return output

    def decodeid2triplet(self, decode_list, tokens, decoded_tag, mask):
        # 13 * 35 * 100
        text_len = self.hyper.max_text_len
        B = decode_list[0].size(0)//text_len
        decoded_tag = [[self.hyper.id2bio[t] for t in tt] for tt in decoded_tag]
        tokens = [[self.hyper.id2word[t] for t in tt] for tt in tokens.tolist()]
        result = [[] for i in range(B)] # batch = 35
        text_length = torch.sum(mask, dim=1).tolist()


        def find_entity(pos: int, tag: List[str], text: List[str]) -> List[str]:

            text_tag = list(zip(text, tag))
            init = pos

            r = []
            cnt = 0
            while pos >= 0:
                cnt += 1
                tok, tg = text_tag[pos]
                if tg == 'O':
                    r.append(tok)
                    break
                elif tg == 'B':
                    r.append(tok)
                    break
                elif tg == 'I':
                    # find pre
                    r.append(tok)
                    pos -= 1
                else:
                    raise ValueError('no <pad>! Should be BIO!')

            r = list(reversed(r))
            return r

        for b in range(B):
            for t in range(text_length[b]):
                text = tokens[b]
                tag = decoded_tag[b]
                head_pos = t
                head = find_entity(head_pos, tag, text)
                head = self.hyper.join(head)
                for i, step in enumerate(decode_list):
                    if i % 2 == 0: # rel
                        # print('rel', step.size())
                        # print(b, t)
                        mat = step.view(B, text_len)
                        rel = mat[b, t].item()
                        rel = self.hyper.id2rel[rel]
                        if rel == 'N':
                            break
                    else:          # ent
                        # 3500 x 100 = 35 x 100 x 100
                        # print('ent', step.size())
                        mat = step.view(B, text_len, text_len)
                        ent = mat[b, t].cpu()
                        
                        assert torch.sum(ent) == 1
                        tail_pos = torch.argmax(ent[:text_length[b]]).item()
                        tail = find_entity(tail_pos, tag, text)
                        tail = self.hyper.join(tail)
                        triplet = {'subject':head, 'predicate':rel, 'object': tail}
                        result[b].append(triplet)

        return result

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

    def run_metrics(self, output):
        self.metrics(output['decode_result'], output['spo_gold'])