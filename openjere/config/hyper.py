import json
import os
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class Hyper(object):
    def __init__(self, path: str):
        self.dataset: str
        self.model: str

        self.data_root: str
        self.raw_data_root: str
        self.train: str
        self.dev: str
        self.test: str
        self.subsets: List[str]
        self.raw_data_list: List[str]

        self.relation_vocab: str
        self.print_epoch: int
        self.evaluation_epoch: int
        self.max_text_len: int

        self.order: List[str]
        self.max_decode_len: Optional[int]
        self.max_encode_len: Optional[int]

        self.cell_name: str
        self.emb_size: int
        self.char_emb_size: int
        self.rel_emb_size: int
        self.bio_emb_size: int
        self.hidden_size: int
        self.dropout: float = 0.5
        self.threshold: float = 0.5
        self.activation: str
        self.optimizer: str
        self.epoch_num: int
        self.batch_size_train: int
        self.batch_size_eval: int
        self.seperator: str
        self.gpu: int

        self.__dict__ = json.load(open(path, "r"))

    def vocab_init(self):
        self.word2id = json.load(
            open(os.path.join(self.data_root, "word_vocab.json"), "r", encoding="utf-8")
        )
        self.rel2id = json.load(
            open(
                os.path.join(self.data_root, "relation_vocab.json"),
                "r",
                encoding="utf-8",
            )
        )
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, "bio_vocab.json"), "r", encoding="utf-8")
        )
        self.id2word = {k: v for v, k in self.word2id.items()}
        self.id2rel = {k: v for v, k in self.rel2id.items()}
        self.id2bio = {k: v for v, k in self.bio_vocab.items()}

    def join(self, toks: List[str]) -> str:
        if self.seperator == "":
            return "".join(toks)
        elif self.seperator == " ":
            return " ".join(toks)
        else:
            raise NotImplementedError("other tokenizer?")

    def tokenizer(self, text: str) -> List[str]:
        if self.seperator == "":
            return list(text)
        elif self.seperator == " ":
            return text.split(" ")
        else:
            raise NotImplementedError("other tokenizer?")
