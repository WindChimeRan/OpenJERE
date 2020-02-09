import os
import json
import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional
from overrides import overrides

from cached_property import cached_property

from lib.preprocessings.abc_preprocessor import Chinese


class Chinese_seq2umt_preprocessing(Chinese):
    def __init__(self, hyper):
        super(Chinese_seq2umt_preprocessing, self).__init__(hyper)

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

    def _train_read_line(self, line: str) -> List[str]:
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
        # result = [
        #     {"text": text, "spo_list": [spo], "seq": self.spo_to_seq(text, spo_list)}
        #     for spo in spo_list
        # ]
        result = self.spo_to_seq(text, spo_list)
        result = [json.dumps(r, ensure_ascii=False) for r in result]
        return result
    
    def spo_to_tree(self, spo_list: List[Dict[str, str]]):
        pass

    def spo_to_seq(self, text: str, spo_list: List[Dict[str, str]]):
        # seq: rel, head, tail
        predicate = spo["predicate"]
        object = spo["object"]
        subject = spo["subject"]

        predicate_id = self.relation_vocab[predicate]
        tokens = self.hyper.tokenizer(text)

        s1, s2, o1, o2 = [[0] * len(text) for _ in range(4)]

        subjectid = tokens.find(subject)
        objectid = tokens.find(object)

        assert subjectid !=1 and objectid != 1

        if subjectid != -1 and objectid != -1:
            key = (subjectid, subjectid + len(sp["subject"]))
            if key not in items:
                items[key] = []
            items[key].append(
                (
                    objectid,
                    objectid + len(sp["object"]),
                    self.relation_vocab[sp["predicate"]],
                )
            )
        pass

    @overrides
    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        if spo_list == []:
            return False

        for t in spo_list:
            if t["object"] not in text or t["subject"] not in text:
                return False
        return True

    @overrides
    def gen_vocab(self, min_freq: int):
        super(Chinese_seq2umt_preprocessing, self).gen_vocab(
            min_freq, init_result={"<pad>": 0}
        )

    @overrides
    def gen_all_data(self):
        print("Override method: Different formats of train/dev!")
        self.gen_train_data()
        # TODO

    def gen_train_data(self):
        train = self.hyper.train
        source = os.path.join(self.raw_data_root, train)
        target = os.path.join(self.data_root, train)
        with open(source, "r") as s, open(target, "w") as t:
            for line in s:
                newlines = self._read_line(line)
                if newlines is not None:
                    for newline in newlines:
                        t.write(newline)
                        t.write("\n")
