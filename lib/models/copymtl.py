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

from lib.layer.crf import CRF
from lib.metrics import F1_triplet
from lib.models.abc_model import ABCModel
from lib.config import EOS, PAD, SOS, OOV, NO_RELATION


class CopyMTL(ABCModel):
    def __init__(self, hyper) -> None:
        super(CopyMTL, self).__init__()

        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, "word_vocab.json"), "r")
        )
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, "relation_vocab.json"), "r")
        )
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, "bio_vocab.json"), "r")
        )

        self.word_embeddings = nn.Embedding(
            num_embeddings=len(self.word_vocab), embedding_dim=hyper.emb_size
        )

        self.relation_emb = nn.Embedding(
            num_embeddings=len(self.relation_vocab), embedding_dim=hyper.rel_emb_size
        )
        # bio + pad
        self.bio_emb = nn.Embedding(
            num_embeddings=len(self.bio_vocab), embedding_dim=hyper.bio_emb_size
        )

        self.sos = nn.Embedding(num_embeddings=1, embedding_dim=hyper.emb_size)

        if hyper.cell_name == "gru":
            self.encoder = nn.GRU(
                hyper.emb_size, hyper.hidden_size, bidirectional=True, batch_first=True
            )
            self.decoder = nn.GRU(
                hyper.emb_size,
                hyper.hidden_size + hyper.bio_emb_size,
                bidirectional=False,
                batch_first=True,
            )
        elif hyper.cell_name == "lstm":
            self.encoder = nn.LSTM(
                hyper.emb_size, hyper.hidden_size, bidirectional=True, batch_first=True
            )
            self.decoder = nn.LSTM(
                hyper.emb_size,
                # hyper.hidden_size + hyper.bio_emb_size,
                hyper.hidden_size,
                bidirectional=False,
                batch_first=True,
            )
        else:
            raise ValueError("cell name should be gru/lstm!")

        if hyper.activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif hyper.activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif hyper.activation.lower() == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError("unexpected activation!")

        self.tagger = CRF(len(self.bio_vocab) - 1, batch_first=True)
        # here the 'N' relation is used as <eos> in standard seq2seq
        self.rel_linear_1 = nn.Linear(
            hyper.hidden_size + hyper.bio_emb_size, len(self.relation_vocab)
        )
        self.rel_linear_a = nn.Linear(
            hyper.hidden_size + hyper.bio_emb_size, hyper.hidden_size
        )
        self.rel_linear_b = nn.Linear(
            hyper.hidden_size + hyper.bio_emb_size, len(self.relation_vocab)
        )
        self.relation_embedding = nn.Embedding(
            len(self.relation_vocab) + 1, hyper.hidden_size
        )

        self.combine_inputs = nn.Linear(
            hyper.hidden_size + hyper.emb_size, hyper.emb_size
        )
        self.attn = nn.Linear(hyper.hidden_size * 2, 1)

        self.entity_linear_1 = nn.Linear(
            hyper.hidden_size * 3 + 3 * hyper.bio_emb_size, hyper.hidden_size
        )
        self.entity_linear_2 = nn.Linear(hyper.hidden_size, 1)

        self.cat_linear = nn.Linear(
            hyper.hidden_size + hyper.bio_emb_size, hyper.hidden_size
        )
        self.sos_embedding = nn.Embedding(1, hyper.emb_size)

        self.do_eos = nn.Linear(hyper.hidden_size, 1)
        self.do_predict = nn.Linear(hyper.hidden_size, len(self.relation_vocab))

        self.fuse = nn.Linear(hyper.hidden_size * 2, 100)
        self.do_copy_linear = nn.Linear(100, 1)

        self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab) - 1)
        # self.ce = nn.CrossEntropyLoss(reduction="none")
        self.loss = nn.NLLLoss(reduction="none")
        self.metrics = F1_triplet()
        self.get_metric = self.metrics.get_metric

    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        tokens = sample.tokens_id.cuda(self.gpu)
        sentence = tokens

        length = sample.length
        B = len(length)

        # if is_train:
        seq_gold = sample.seq_id.cuda(self.gpu)
        bio_gold = sample.bio_id.cuda(self.gpu)

        text_list = sample.text
        # spo_gold = sample.spo_gold

        mask = tokens != self.word_vocab["<pad>"]  # batch x seq

        mask_decode = sample.mask_decode.cuda(self.gpu)

        embedded = self.word_embeddings(tokens)
        o, h = self.encoder(embedded)

        o = (lambda a: sum(a) / 2)(torch.split(o, self.hyper.hidden_size, dim=2))
        h = tuple(map(lambda state: sum(torch.split(state, 1, dim=0)) / 2, h))

        emi = self.emission(o)

        output_dic = {}

        crf_loss = 0

        if is_train:
            crf_loss = -self.tagger(emi, bio_gold, mask=mask, reduction="mean")
            # print(crf_loss)
            # exit()
        else:
            decoded_tag = self.tagger.decode(emissions=emi, mask=mask)
            temp_tag = copy.deepcopy(decoded_tag)
            for line in temp_tag:
                line.extend(
                    [self.bio_vocab["<pad>"]] * (self.hyper.max_text_len - len(line))
                )
            bio_gold = torch.tensor(temp_tag).cuda(self.gpu)

        tag_emb = self.bio_emb(bio_gold)

        cat_o = o
        # cat_o = torch.cat((o, tag_emb), dim=2)

        o_hid_size = cat_o.size(-1)

        # copy_o = (
        #     cat_o.unsqueeze(0)
        #     .expand(self.hyper.max_text_len, -1, -1, -1)
        #     .contiguous()
        #     .view(
        #         -1,
        #         self.hyper.max_text_len,
        #         self.hyper.hidden_size + self.hyper.bio_emb_size,
        #     )
        # )

        sos = self.sos(torch.tensor(0).cuda(self.gpu)).unsqueeze(0).expand(B, -1)
        decoder_input = sos
        # h = fst_hidden = cat_o.view(
        #     -1, self.hyper.hidden_size + self.hyper.bio_emb_size
        # )

        # B_stacked, L, _ = copy_o.size()
        # print(B, L)
        decoder_loss = 0
        decoder_result = []
        decoder_state = h

        pred_action_list = []
        pred_logits_list = []

        go = torch.zeros(sentence.size()[0], dtype=torch.int64).to(self.gpu)
        output = self.sos_embedding(go)

        # first_entity_mask = torch.ones(go.size()[0], self.maxlen).to(self.gpu)
        # if is_train:
        # seq_gold = seq_gold.view(-1, 2 * self.hyper.max_decode_len + 1)
        actions = []
        for t in range(self.hyper.max_decode_len + 1):

            bag, decoder_state = self._decode_step(
                self.decoder, output, decoder_state, cat_o
            )
            predict_logits, copy_logits = bag

            if t % 3 == 0:
                action_logits = predict_logits
            else:
                action_logits = copy_logits - torch.zeros_like(
                    copy_logits
                ).masked_fill_(mask ^ True, -float("inf"))

            max_action = torch.argmax(action_logits, dim=1).detach()

            pred_action_list.append(max_action)
            pred_logits_list.append(action_logits)

            # next time step
            if t % 3 == 0:
                output = max_action
                output = self.relation_embedding(output)

            else:
                copy_index = (
                    torch.zeros_like(sentence)
                    .scatter_(1, max_action.unsqueeze(1), 1)
                    .bool()
                )
                output = sentence[copy_index]
                output = self.word_embeddings(output)

            step_loss = self.masked_NLLloss(
                mask_decode[:, t], action_logits, seq_gold[:, t]
            )

            # decoder_loss += step_loss.sum()
            decoder_loss += step_loss
            # print(step_loss.size())

        loss = crf_loss + decoder_loss
        output_dic["crf_loss"] = crf_loss
        output_dic["decoder_loss"] = decoder_loss
        output_dic["loss"] = loss

        output_dic["description"] = partial(self.description, output=output_dic)
        # print(crf_loss)
        # print(decoder_loss)
        # exit()
        if not is_train:
            spo_gold = sample.spo_gold
            output_dic["spo_gold"] = spo_gold
            decoder_result = self.decodeid2triplet(
                pred_action_list, sample.text, decoded_tag, mask
            )
            output_dic["decode_result"] = decoder_result

        return output_dic

    def decodeid2triplet(self, decode_list, tokens, decoded_tag, mask):
        # 13 * 35 * 100
        # print(list(map(lambda x : x[0], (decode_list, tokens, decoded_tag, mask))))
        # exit()
        # text_len = self.hyper.max_text_len

        # B = decode_list[0].size(0) // text_len
        decoded_tag = [[self.hyper.id2bio[t] for t in tt] for tt in decoded_tag]
        # tokens = [[self.hyper.id2word[t] for t in tt] for tt in tokens.tolist()]
        # result = [[] for i in range(B)]  # batch = 35
        # text_length = torch.sum(mask, dim=1).tolist()

        def find_entity(pos: int, tag: List[str], text: List[str]) -> List[str]:

            text_tag = list(zip(text, tag))
            init = pos

            r = []
            cnt = 0
            while pos >= 0:
                cnt += 1
                tok, tg = text_tag[pos]
                if tg == "O":
                    r.append(tok)
                    break
                elif tg == "B":
                    r.append(tok)
                    break
                elif tg == "I":
                    # find pre
                    r.append(tok)
                    pos -= 1
                else:
                    raise ValueError("no <pad>! Should be BIO!")

            r = list(reversed(r))
            return r

        batch_size = decode_list[0].size(0)
        result = [[] for i in range(batch_size)]  # batch = 35

        decode_list = torch.stack(decode_list).permute(1, 0)
        for b in range(batch_size):
            triplet = {}
            for t in range(self.hyper.max_decode_len + 1):
                text = tokens[b]
                tag = decoded_tag[b]
                seq = decode_list[b]

                # print(text, tag, seq)
                # print(len(text))
                # print(len(tag))
                # exit()

                if t % 3 == 0:  # rel
                    rel = seq[t].item()
                    rel = self.hyper.id2rel[rel]
                    triplet["predicate"] = rel
                    if rel == NO_RELATION:
                        break
                elif t % 3 == 1:  # ent
                    head_pos = seq[t]
                    # try:
                    head = find_entity(head_pos, tag, text)
                    # except:
                    #     print(head_pos)
                    #     print(tag)
                    #     print(text)
                    #     print(mask[b].tolist())
                    #     print(mask[b].sum())
                    #     exit()
                    head = self.hyper.join(head)
                    triplet["subject"] = head
                elif t % 3 == 2:  # ent
                    tail_pos = seq[t]
                    tail = find_entity(tail_pos, tag, text)
                    tail = self.hyper.join(tail)
                    triplet["object"] = tail
                    result[b].append(triplet)
                    triplet = {}
        return result

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, L_crf: {:.2f}, L_decode: {:.2f}, epoch: {}/{}:".format(
            output["loss"].item(),
            output["crf_loss"].item(),
            output["decoder_loss"].item(),
            epoch,
            epoch_num,
        )

    # def _decode_step(self, t: int, decoder_state, decoder_input, copy_o, fst_hidden):

    #     decoder_input = decoder_input.unsqueeze(dim=1)
    #     decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)

    #     if t % 2 == 0:
    #         # relation logits
    #         output_logits = self.rel_linear_b(decoder_state.squeeze())
    #     else:
    #         # cat(H_decoder, H_encoders)
    #         output_logits = torch.cat(
    #             (
    #                 decoder_state.permute(1, 0, 2).expand(
    #                     -1, self.hyper.max_text_len, -1
    #                 ),
    #                 copy_o,
    #             ),
    #             dim=2,
    #         )
    #         # cat(H_decoder, H_encoders, H_decoder_1)
    #         output_logits = torch.cat(
    #             (
    #                 output_logits,
    #                 fst_hidden.permute(1, 0, 2).expand(-1, self.hyper.max_text_len, -1),
    #             ),
    #             dim=2,
    #         )

    #         output_logits = self.entity_linear_2(
    #             self.activation(self.entity_linear_1((output_logits)))
    #         ).squeeze()

    #     return decoder_output, decoder_state, output_logits
    def calc_context(
        self, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:

        # decoder_state.size() == torch.Size([1, 100, 1000])
        # -> torch.Size([100, 1, 1000]) -> torch.Size([100, 80, 1000]) -cat-> torch.Size([100, 80, 2000])
        attn_weight = torch.cat(
            (
                decoder_state.permute(1, 0, 2).expand_as(encoder_outputs),
                encoder_outputs,
            ),
            dim=2,
        )
        attn_weight = F.softmax((self.attn(attn_weight)), dim=1)
        attn_applied = torch.bmm(attn_weight.permute(0, 2, 1), encoder_outputs).squeeze(
            1
        )

        return attn_applied

    def do_copy(
        self, output: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:

        out = torch.cat(
            (output.unsqueeze(1).expand_as(encoder_outputs), encoder_outputs), dim=2
        )
        out = F.selu(self.fuse(F.selu(out)))
        out = self.do_copy_linear(out).squeeze(2)
        # out = (self.do_copy_linear(out).squeeze(2))
        return out

    def _decode_step(
        self,
        rnn_cell: nn.modules,
        emb: torch.Tensor,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        # print(decoder_state.size())
        # print(encoder_outputs.size())
        decoder_state_h = decoder_state[0]

        context = self.calc_context(decoder_state_h, encoder_outputs)

        output = self.combine_inputs(torch.cat((emb, context), dim=1))

        output, decoder_state = rnn_cell(output.unsqueeze(1), decoder_state)

        output = output.squeeze()

        # eos_logits = F.selu(self.do_eos(output))
        # predict_logits = F.selu(self.do_predict(output))
        eos_logits = self.do_eos(output)
        predict_logits = self.do_predict(output)

        # predict_logits = F.log_softmax(
        #     torch.cat((predict_logits, eos_logits), dim=1), dim=1
        # )

        predict_logits = F.log_softmax(predict_logits, dim=1)

        copy_logits = self.do_copy(output, encoder_outputs)

        # assert copy_logits.size() == first_entity_mask.size()
        # original
        # copy_logits = copy_logits * first_entity_mask
        # copy_logits = copy_logits

        # copy_logits = torch.cat((copy_logits, eos_logits), dim=1)
        copy_logits = F.log_softmax(copy_logits, dim=1)

        # # bug fix
        # copy_logits = torch.cat((copy_logits, eos_logits), dim=1)
        # first_entity_mask = torch.cat((first_entity_mask, torch.ones_like(eos_logits)), dim=1)
        #
        # copy_logits = F.softmax(copy_logits, dim=1)
        # copy_logits = copy_logits * first_entity_mask
        # copy_logits = torch.clamp(copy_logits, 1e-10, 1.)
        # copy_logits = torch.log(copy_logits)

        return (predict_logits, copy_logits), decoder_state

    def masked_NLLloss(self, mask, output_logits, seq_gold):
        # print(output_logits.size(), seq_gold.size(), mask.size())
        loss = (self.loss(output_logits, seq_gold) * mask).mean()

        # loss = self.loss(output_logits, seq_gold).masked_select(mask)
        return loss

    def run_metrics(self, output):
        self.metrics(output["decode_result"], output["spo_gold"])
