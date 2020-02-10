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

        result = self.spo_to_seq(text, spo_list)
        # print(result)
        result = [json.dumps(r, ensure_ascii=False) for r in result]
        # print(result)
        # if len(spo_list) > 1:
        #     exit()
        return result

    def spo_to_tree(self, spo_list: List[Dict[str, str]]) -> List[Tuple[str]]:
        """return the ground truth of the tree: rel, subj, obj, used for teacher forcing.

        r: given text, one of the relations

        s: given r_1, one of the subjects

        rel: multi-label classification of relation

        subj: multi-label classification of subject

        obj: multi-label classification of object
        
        Arguments:
            spo_list {List[Dict[str, str]]} -- [description]
        
        Returns:
            List[Tuple[str]] -- [(r, s, rel, subj, obj)]
        """
        # rel, subj, obj
        #
        result = []
        rel = [t["predicate"] for t in spo_list]
        for r in rel:
            subj = [t["subject"] for t in spo_list if t["predicate"] == r]
            for s in subj:
                obj = [
                    t["object"]
                    for t in spo_list
                    if t["predicate"] == r and t["subject"] == s
                ]
                result.append((r, s, rel, subj, obj))
        return result

    def spo_to_seq(self, text: str, spo_list: List[Dict[str, str]]):
        # seq: rel, head, tail
        # predicate = spo["predicate"]
        # object = spo["object"]
        # subject = spo["subject"]

        # predicate_id = self.relation_vocab[predicate]
        def find(tokens: List[str], entity: List[str]):
            # Python program for KMP Algorithm
            # https://www.geeksforgeeks.org/python-program-for-kmp-algorithm-for-pattern-searching-2/
            def KMPSearch(pat, txt):
                M = len(pat)
                N = len(txt)

                result = []

                # create lps[] that will hold the longest prefix suffix
                # values for pattern
                lps = [0] * M
                j = 0  # index for pat[]

                # Preprocess the pattern (calculate lps[] array)
                computeLPSArray(pat, M, lps)

                i = 0  # index for txt[]
                while i < N:
                    if pat[j] == txt[i]:
                        i += 1
                        j += 1

                    if j == M:
                        result.append(i - j)
                        # print("Found pattern at index " + str(i-j))
                        j = lps[j - 1]

                    # mismatch after j matches
                    elif i < N and pat[j] != txt[i]:
                        # Do not match lps[0..lps[j-1]] characters,
                        # they will match anyway
                        if j != 0:
                            j = lps[j - 1]
                        else:
                            i += 1
                return result

            def computeLPSArray(pat, M, lps):
                len = 0  # length of the previous longest prefix suffix

                lps[0]  # lps[0] is always 0
                i = 1

                # the loop calculates lps[i] for i = 1 to M-1
                while i < M:
                    if pat[i] == pat[len]:
                        len += 1
                        lps[i] = len
                        i += 1
                    else:
                        # This is tricky. Consider the example.
                        # AAACAAAA and i = 7. The idea is similar
                        # to search step.
                        if len != 0:
                            len = lps[len - 1]

                            # Also, note that we do not increment i here
                        else:
                            lps[i] = 0
                            i += 1

            cand_id = KMPSearch(entity, tokens)
            assert len(cand_id) > 0
            id = cand_id[0]

            return id

        tree = self.spo_to_tree(spo_list)
        # print(tree)
        # print('-'*50)

        tokens = self.hyper.tokenizer(text)

        results = []
        for t in tree:
            r, s, rel, subj, obj = t

            s1, s2, o1, o2 = [[0] * len(tokens) for _ in range(4)]

            rel_idx = [0] * len(self.relation_vocab)
            for rel_name in rel:
                rel_idx[self.relation_vocab[rel_name]] = 1
            for subj_name in subj:
                id = find(tokens, subj_name)
                s1[id] = 1
                s2[id + len(self.hyper.tokenizer(subj_name)) - 1] = 1
            for obj_name in obj:
                id = find(tokens, obj_name)
                o1[id] = 1
                o2[id + len(self.hyper.tokenizer(obj_name)) - 1] = 1

            r = self.relation_vocab[r]
            k1 = find(tokens, s)
            k2 = k1 + len(self.hyper.tokenizer(s)) - 1

            result = {
                "text": text,
                "spo_list": spo_list,
                "r": r,
                "k1": k1,
                "k2": k2,
                "rel_gt": rel_idx,
                "s1_gt": s1,
                "s2_gt": s2,
                "o1_gt": o1,
                "o2_gt": o2,
            }
            results.append(result)
        return results

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
        print("Override gen_all_data: Different formats of train/dev!")
        
        for path in self.hyper.raw_data_list:
            if path == "train_data.json":
                self.gen_train_data(path)
            elif path == "new_train_data.json":
                self.gen_train_data(path)
            else:
                self._gen_one_data(path)

    def gen_train_data(self, path):
        source = os.path.join(self.raw_data_root, path)
        target = os.path.join(self.data_root, path)
        with open(source, "r") as s, open(target, "w") as t:
            for line in s:
                newlines = self._train_read_line(line)
                if newlines is not None:
                    for newline in newlines:
                        t.write(newline)
                        t.write("\n")
