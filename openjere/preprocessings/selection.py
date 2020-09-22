import os
import json
import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional
from overrides import overrides

from cached_property import cached_property

from openjere.preprocessings.abc_preprocessor import ABC_data_preprocessing
from openjere.config.const import find


class Selection_preprocessing(ABC_data_preprocessing):
    # def __init__(self, hyper):
    #     super(Chinese_selection_preprocessing, self).__init__(hyper)

    @overrides
    def _read_line(self, line: str) -> Optional[str]:
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text = instance["text"]

        bio = None
        selection = None

        if "spo_list" in instance:
            spo_list = instance["spo_list"]

            if not self._check_valid(text, spo_list):
                return None
            spo_list = [
                {
                    "predicate": spo["predicate"],
                    "object": spo["object"],
                    "subject": spo["subject"],
                }
                for spo in spo_list
            ]

            entities: List[str] = self.spo_to_entities(text, spo_list)
            relations: List[str] = self.spo_to_relations(text, spo_list)

            bio = self.spo_to_bio(text, entities)
            selection = self.spo_to_selection(text, spo_list)

        result = {
            "text": text,
            "spo_list": spo_list,
            "bio": bio,
            "selection": selection,
        }
        return json.dumps(result, ensure_ascii=False)

    @overrides
    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        if spo_list == []:
            return False
        if len(self.hyper.tokenizer(text)) > self.hyper.max_text_len:
            return False
        for t in spo_list:
            if t["object"] not in text or t["subject"] not in text:
                return False
        return True

    # @overrides
    # def gen_vocab(self, min_freq: int):
    #     super(Chinese_selection_preprocessing, self).gen_vocab(
    #         min_freq, init_result={"<pad>": 0}
    #     )

    def spo_to_selection(
        self, text: str, spo_list: List[Dict[str, str]]
    ) -> List[Dict[str, int]]:

        tokens = self.hyper.tokenizer(text)

        selection = []
        for triplet in spo_list:

            object = self.hyper.tokenizer(triplet["object"])
            subject = self.hyper.tokenizer(triplet["subject"])

            object_pos = find(tokens, object) + len(object) - 1
            subject_pos = find(tokens, subject) + len(subject) - 1
            # object_pos = text.find(object) + len(object) - 1
            relation_pos = self.relation_vocab[triplet["predicate"]]
            # subject_pos = text.find(subject) + len(subject) - 1

            selection.append(
                {
                    "subject": subject_pos,
                    "predicate": relation_pos,
                    "object": object_pos,
                }
            )

        return selection

    def spo_to_bio(self, text: str, entities: List[str]) -> List[str]:
        text = self.hyper.tokenizer(text)
        bio = ["O"] * len(text)
        for e in entities:
            begin = find(text, self.hyper.tokenizer(e))
            # begin = text.find(e)
            end = begin + len(self.hyper.tokenizer(e)) - 1

            assert end <= len(text)

            bio[begin] = "B"
            for i in range(begin + 1, end + 1):
                bio[i] = "I"
        return bio
