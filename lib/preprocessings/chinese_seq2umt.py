import os
import json
import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional
from overrides import overrides

from cached_property import cached_property

from lib.preprocessings.abc_preprocessor import Chinese


class Chinese_selection_preprocessing(Chinese):
    def __init__(self, hyper):
        super(Chinese_selection_preprocessing, self).__init__(hyper)

    @overrides
    def _read_line(self, line: str) -> Optional[str]:
        # for evaluation only
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text = instance["text"]

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

        result = {
            "text": text,
            "spo_list": spo_list,
        }
        return json.dumps(result, ensure_ascii=False)

    def _train_read_line(self, line: str) -> Optional[str]:
        # teacher forcing
        # batches are aligned by the triplets rather than sentences.
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text = instance["text"]

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

        result = {
            "text": text,
            "spo_list": spo_list,
        }
        return json.dumps(result, ensure_ascii=False)
    @overrides
    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        # if spo_list == []:
        #     return False
        # if len(text) > self.hyper.max_text_len:
        #     return False
        # for t in spo_list:
        #     if t["object"] not in text or t["subject"] not in text:
        #         return False
        return True

    @overrides
    def gen_vocab(self, min_freq: int):
        super(Chinese_selection_preprocessing, self).gen_vocab(
            min_freq, init_result={"<pad>": 0}
        )

    @overrides
    def gen_all_data(self):
        print("Override method: Different formats of train/dev!")
        pass

    def gen_train_data(self):
        pass