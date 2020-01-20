import os
import json
import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

from cached_property import cached_property
from overrides import overrides

from lib.preprocessings.abc_data import Chinese


class Chinese_copymb_preprocessing(Chinese):
    def __init__(self, hyper):
        super(Chinese_copymb_preprocessing, self).__init__(hyper)

    @overrides
    def _read_line(self, line: str) -> Optional[str]:
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text = instance['text']

        bio = None

        if 'spo_list' in instance:
            spo_list = instance['spo_list']

            if not self._check_valid(text, spo_list):
                return None

            seq = self.spo_to_seq(text, spo_list)

            if not self._check_seq(seq):
                return None

            entities: List[str] = self.spo_to_entities(text, spo_list)
            relations: List[str] = self.spo_to_relations(text, spo_list)

            bio = self.spo_to_bio(text, entities)
            

        result = {
            'text': text,
            'spo_list': spo_list,
            'bio': bio,
            'seq': seq
        }
        return json.dumps(result, ensure_ascii=False)



    @overrides
    def gen_vocab(self, min_freq: int):
        super(Chinese_copymb_preprocessing, self).gen_vocab(
            min_freq, init_result={'<pad>': 0, '<eos>': 1})

    @overrides
    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        if spo_list == []:
            return False
        if len(text) > self.hyper.max_text_len:
            return False

        # if len(set([t['subject'] for t in spo_list])) > self.hyper.max_decode_len:
        #     return False
        if max(Counter([t['subject'] for t in spo_list]).values()) > self.hyper.max_decode_len:
            return False

        for t in spo_list:
            if t['object'] not in text or t['subject'] not in text:
                return False
        return True

    def _check_seq(self, seq):
        # TODO
        # The original goal of this function is to check the length of the decoder given [subject].
        # decoded sequece: p_1, o_1, p_2, o_2, p_3, o_3....
        # but it can also be p_1, p_2, p_3, then use p_1 to select [o_1, o_2], use p_2 to select [o_3]...

        if max(map(len, seq.values())) > self.hyper.max_decode_len * 2:
            return False
        else:
            return True

    def spo_to_seq(self, text: str, spo_list: List[Dict[str, str]], s_fst: bool = True) -> Dict[int, List[int]]:
        dic = {}
        for triplet in spo_list:

            object = triplet['object']
            subject = triplet['subject']

            object_pos = text.find(object) + len(object) - 1
            relation_pos = self.relation_vocab[triplet['predicate']]
            subject_pos = text.find(subject) + len(subject) - 1

            # dangerous!!!
            # ------------------------------------------------- #
            if not s_fst:
                # ops (default spo)
                object_pos, subject_pos = subject_pos, object_pos
            # ------------------------------------------------- #

            if subject_pos in dic:
                dic[subject_pos].extend([relation_pos, object_pos])
            else:
                dic[subject_pos] = [relation_pos, object_pos]
        # if max(map(len, dic.values())) > self.hyper.max_decode_len * 2:
        #     print(dic)
        return dic

    def spo_to_bio(self, text: str, entities: List[str]) -> List[str]:
        bio = ['O'] * len(text)
        for e in entities:
            begin = text.find(e)
            end = begin + len(e) - 1

            assert end <= len(text)

            bio[begin] = 'B'
            for i in range(begin + 1, end + 1):
                bio[i] = 'I'
        return bio
