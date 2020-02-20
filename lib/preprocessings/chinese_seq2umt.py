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

    def spo_to_tree(
        self, spo_list: List[Dict[str, str]], order=("predicate", "subject", "object")
    ) -> List[Tuple[str]]:
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
        t1_out = [t[order[0]] for t in spo_list]
        for t1_in in t1_out:
            t2_out = [t[order[1]] for t in spo_list if t[order[0]] == t1_in]
            for t2_in in t2_out:
                t3_out = [
                    t[order[2]]
                    for t in spo_list
                    if t[order[0]] == t1_in and t[order[1]] == t2_in
                ]
                result.append((t1_in, t2_in, t1_out, t2_out, t3_out))
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

        order = self.hyper.order

        tree = self.spo_to_tree(spo_list, order)
        tokens = self.hyper.tokenizer(text)

        def to_rel(outp):
            # pure

            rel_idx = [0] * len(self.relation_vocab)
            for rel_name in outp:
                rel_idx[self.relation_vocab[rel_name]] = 1
            return rel_idx

        def to_ent(outp):
            # side effect!
            ent1, ent2 = [[0] * len(tokens) for _ in range(2)]
            for name in outp:
                id = find(tokens, name)
                ent1[id] = 1
                ent2[id + len(self.hyper.tokenizer(name)) - 1] = 1
            return ent1, ent2

        def to_in_key(inp, name):
            # side effect!
            if name == "predicate":
                rel_in = self.relation_vocab[inp]
                out = rel_in
            else:
                k1 = find(tokens, inp)
                k2 = k1 + len(self.hyper.tokenizer(inp)) - 1
                out = k1, k2
            return out

        op_dic = {"predicate": to_rel, "subject": to_ent, "object": to_ent}

        results = []
        for t in tree:
            t1_in, t2_in, t1_out, t2_out, t3_out = t

            for name, ori in zip(order, (t1_out, t2_out, t3_out)):
                new = op_dic[name](ori)
                if name == "predicate":
                    rel_idx = new
                elif name == "subject":
                    s1, s2 = new
                elif name == "object":
                    o1, o2 = new
                else:
                    raise ValueError("should be in predicate, subject, object")

            rel_in = to_in_key(t1_in, order[0])
            k1, k2 = to_in_key(t2_in, order[1])

            result = {
                "text": text,
                "spo_list": spo_list,
                "r": rel_in,
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
