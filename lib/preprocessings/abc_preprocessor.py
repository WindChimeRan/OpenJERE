import os
import json
import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional
from abc import ABC, abstractmethod

from cached_property import cached_property


class ABC_data_preprocessing(ABC):
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper.raw_data_root
        self.data_root = hyper.data_root

        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

        self.relation_vocab_path = os.path.join(self.data_root, hyper.relation_vocab)

    @cached_property
    def relation_vocab(self):
        if os.path.exists(self.relation_vocab_path):
            pass
        else:
            self.gen_relation_vocab()
        return json.load(open(self.relation_vocab_path, "r", encoding="utf-8"))

    # model
    @abstractmethod
    def _read_line(self, line: str) -> Optional[str]:
        raise NotImplementedError("abc method!")

    # model
    @abstractmethod
    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        pass

    # model
    def gen_all_data(self):
        for path in self.hyper.raw_data_list:
            self._gen_one_data(path)

    # model
    def _gen_one_data(self, dataset):
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)
        with open(source, "r", encoding="utf-8") as s, open(
            target, "w", encoding="utf-8"
        ) as t:
            for line in s:
                newline = self._read_line(line)
                if newline is not None:
                    t.write(newline)
                    t.write("\n")

    # model
    def gen_bio_vocab(self):
        result = {"<pad>": 3, "B": 0, "I": 1, "O": 2}
        json.dump(result, open(os.path.join(self.data_root, "bio_vocab.json"), "w"))

    # # data
    # @abstractmethod
    # def gen_relation_vocab(self):
    #     pass

    # # data
    # @abstractmethod
    # def yield_text(self, source: str) -> List[str]:
    #     pass

    # data
    # @abstractmethod
    def gen_vocab(
        self,
        min_freq: int,
        init_result: Dict[str, int] = {"<pad>": 0, "<eos>": 1, "<|>": 2},
    ):
        # might contain sos, eos, pad ....
        source = os.path.join(self.raw_data_root, self.hyper.train)
        target = os.path.join(self.data_root, "word_vocab.json")

        cnt = Counter()

        for text in self.yield_key(source, "text"):
            cnt.update(self.hyper.tokenizer(text))

        result = init_result
        i = len(init_result)
        assert max(init_result.values()) == i - 1
        for k, v in cnt.items():
            if v > min_freq:
                result[k] = i
                i += 1
        result["<oov>"] = i
        json.dump(result, open(target, "w", encoding="utf-8"), ensure_ascii=False)

    def spo_to_entities(self, text: str, spo_list: List[Dict[str, str]]) -> List[str]:
        entities = set(t["object"] for t in spo_list) | set(
            t["subject"] for t in spo_list
        )
        return list(entities)

    def spo_to_relations(self, text: str, spo_list: List[Dict[str, str]]) -> List[str]:
        return [t["predicate"] for t in spo_list]

    def gen_relation_vocab(self):
        relation_vocab = {}
        rel_set = set()
        source = os.path.join(self.raw_data_root, self.hyper.train)

        for spo_list in self.yield_key(source, "spo_list"):
            rel_set.update(self.spo_to_relations(None, spo_list))

        relation_vocab = {k: v for v, k in enumerate(rel_set)}
        relation_vocab["N"] = len(relation_vocab)
        json.dump(
            relation_vocab,
            open(self.relation_vocab_path, "w", encoding="utf-8"),
            ensure_ascii=False,
        )

    def yield_key(self, source: str, key: str) -> List[str]:
        # key = "text"
        with open(source, "r", encoding="utf-8") as s:
            for line in s:
                line = line.strip("\n")
                if not line:
                    return None
                instance = json.loads(line)
                value = instance[key]
                yield value
