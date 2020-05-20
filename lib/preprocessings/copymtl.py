import os
import json
import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

from cached_property import cached_property
from overrides import overrides

from lib.preprocessings.abc_preprocessor import ABC_data_preprocessing
from lib.config.const import find, NO_RELATION


class Copymtl_preprocessing(ABC_data_preprocessing):
    # def __init__(self, hyper):
    #     super(Chinese_copymb_preprocessing, self).__init__(hyper)

    @overrides
    def _read_line(self, line: str) -> Optional[str]:
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text = instance["text"]

        bio = None

        if "spo_list" in instance:
            spo_list = instance["spo_list"]

            if not self._check_valid(text, spo_list):
                return None

            seq = self.spo_to_seq(text, spo_list)

            if not self._check_seq(seq):
                return None

            entities: List[str] = self.spo_to_entities(text, spo_list)
            relations: List[str] = self.spo_to_relations(text, spo_list)

            bio = self.spo_to_bio(text, entities)

        result = {"text": text, "spo_list": spo_list, "bio": bio, "seq": seq}
        return json.dumps(result, ensure_ascii=False)

    # @overrides
    # def gen_vocab(self, min_freq: int):
    #     super(Chinese_copymb_preprocessing, self).gen_vocab(
    #         min_freq, init_result={"<pad>": 0, "<eos>": 1}
    #     )

    @overrides
    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        if spo_list == []:
            return False
        if len(self.hyper.tokenizer(text)) > self.hyper.max_text_len:
            return False

        if len(spo_list) > 5:
            return False
        # if (
        #     max(Counter([t["subject"] for t in spo_list]).values())
        #     > self.hyper.max_decode_len
        # ):
        #     return False

        for t in spo_list:
            if t["object"] not in text or t["subject"] not in text:
                return False
        return True

    def _check_seq(self, seq):

        if len(seq) > self.hyper.max_decode_len:
            return False
        else:
            return True

    def spo_to_seq(
        self, text: str, spo_list: List[Dict[str, str]]
    ) -> Dict[int, List[int]]:
        dic = {}
        tokens = self.hyper.tokenizer(text)
        result = []
        for triplet in spo_list:

            object = self.hyper.tokenizer(triplet["object"])
            subject = self.hyper.tokenizer(triplet["subject"])

            object_pos = find(tokens, object) + len(object) - 1
            subject_pos = find(tokens, subject) + len(subject) - 1
            relation_pos = self.relation_vocab[triplet["predicate"]]
            result.extend([relation_pos, subject_pos, object_pos])
        # result.append(self.relation_vocab[NO_RELATION])
        return result

    def spo_to_bio(self, text: str, entities: List[str]) -> List[str]:
        text = self.hyper.tokenizer(text)
        bio = ["O"] * len(text)
        for e in entities:
            e_list = self.hyper.tokenizer(e)

            begin = find(text, e_list)
            end = begin + len(e_list) - 1

            assert end <= len(text)

            bio[begin] = "B"
            for i in range(begin + 1, end + 1):
                bio[i] = "I"
        return bio
