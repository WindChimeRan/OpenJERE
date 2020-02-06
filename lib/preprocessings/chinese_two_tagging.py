#! -*- coding:utf-8 -*-
# export dev.json, train.json, id2char, char2id

import os
import json
import numpy as np

from collections import Counter
from typing import Dict, List, Tuple, Set, Optional
from overrides import overrides

from cached_property import cached_property

from lib.preprocessings.abc_preprocessor import Chinese


class Chinese_twotagging_preprocessing(Chinese):
    def __init__(self, hyper):
        super(Chinese_twotagging_preprocessing, self).__init__(hyper)

    @overrides
    def _read_line(self, line: str) -> Optional[str]:
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text = instance['text']

        if 'spo_list' in instance:
            spo_list = instance['spo_list']

            if not self._check_valid(text, spo_list):
                return None
            spo_list = [{
                'predicate': spo['predicate'],
                'object': spo['object'],
                'subject': spo['subject']
            } for spo in spo_list]

            entities: List[str] = self.spo_to_entities(text, spo_list)
            relations: List[str] = self.spo_to_relations(text, spo_list)


        result = {
            'text': text,
            'spo_list': spo_list,
        }
        return json.dumps(result, ensure_ascii=False)

    # TODO: Not sure
    @overrides
    def _check_valid(self, text: str, spo_list: List[Dict[str, str]]) -> bool:
        # if spo_list == []:
        #     return False
        # if len(text) > self.hyper.max_text_len:
        #     return False
        # for t in spo_list:
        #     if t['object'] not in text or t['subject'] not in text:
        #         return False
        return True

    @overrides
    def gen_vocab(self, min_freq: int):
        super(Chinese_twotagging_preprocessing, self).gen_vocab(min_freq, init_result={'<pad>': 0})





# import json
# from tqdm import tqdm
# import codecs


# all_50_schemas = set()

# with open('all_50_schemas') as f:
#     for l in tqdm(f):
#         a = json.loads(l)
#         all_50_schemas.add(a['predicate'])

# id2predicate = {i+1:j for i,j in enumerate(all_50_schemas)} # 0表示终止类别
# predicate2id = {j:i for i,j in id2predicate.items()}

# with codecs.open('all_50_schemas_me.json', 'w', encoding='utf-8') as f:
#     json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)


# chars = {}
# min_count = 2

# # train
# train_data = []

# with open('train_data.json') as f:
#     for l in tqdm(f):
#         a = json.loads(l)
#         train_data.append(
#             {
#                 'text': a['text'],
#                 'spo_list': [(i['subject'], i['predicate'], i['object']) for i in a['spo_list']]
#             }
#         )
#         for c in a['text']:
#             chars[c] = chars.get(c, 0) + 1

# with codecs.open('train_data_me.json', 'w', encoding='utf-8') as f:
#     json.dump(train_data, f, indent=4, ensure_ascii=False)

# # dev
# dev_data = []


# with open('dev_data.json') as f:
#     for l in tqdm(f):
#         a = json.loads(l)
#         dev_data.append(
#             {
#                 'text': a['text'],
#                 'spo_list': [(i['subject'], i['predicate'], i['object']) for i in a['spo_list']]
#             }
#         )
#         for c in a['text']:
#             chars[c] = chars.get(c, 0) + 1


# with codecs.open('dev_data_me.json', 'w', encoding='utf-8') as f:
#     json.dump(dev_data, f, indent=4, ensure_ascii=False)

# # dict
# with codecs.open('all_chars_me.json', 'w', encoding='utf-8') as f:
#     chars = {i:j for i,j in chars.items() if j >= min_count}
#     id2char = {i+2:j for i,j in enumerate(chars)} # padding: 0, unk: 1
#     char2id = {j:i for i,j in id2char.items()}
#     json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)