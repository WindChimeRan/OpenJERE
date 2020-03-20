import os
import json
import numpy as np
import random

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional

from cached_property import cached_property
from overrides import overrides

from lib.preprocessings.abc_preprocessor import ABC_data_preprocessing
from lib.config import SEP_SEMICOLON, SEP_VERTICAL_BAR, EOS


class WDec_preprocessing(ABC_data_preprocessing):
    @overrides
    def _read_line(self, line: str) -> Optional[str]:
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text = instance["text"]

        if "spo_list" in instance:
            spo_list = instance["spo_list"]

            if not self._check_valid(text, spo_list):
                return None

            seq = self.spo_to_seq(text, spo_list)

            # if not self._check_seq(seq):
            #     return None

        result = {"text": text, "spo_list": spo_list, "seq": seq}
        return json.dumps(result, ensure_ascii=False)

    @overrides
    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        if spo_list == []:
            return False
        # if len(text) > self.hyper.max_text_len:
        #     return False

        # if len(set([t['subject'] for t in spo_list])) > self.hyper.max_decode_len:
        #     return False

        for t in spo_list:
            if t["object"] not in text or t["subject"] not in text:
                return False
        return True

    def _check_seq(self, seq):
        pass

    def spo_to_seq(
        self, text: str, spo_list: List[Dict[str, str]], s_fst: bool = True
    ) -> List[str]:
        dic = {}
        random.shuffle(spo_list)
        spo_seq = []
        for triplet in spo_list:

            object = triplet["object"]
            subject = triplet["subject"]
            predicate = triplet["predicate"]

            tuple = (" " + SEP_SEMICOLON + " ").join((subject, object, predicate))

            spo_seq.append(tuple)

        return (" " + SEP_VERTICAL_BAR + " ").join(spo_seq)
