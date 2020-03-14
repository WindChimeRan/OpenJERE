import os
import random
import json

from typing import Dict, List, Tuple, Set, Optional

CHINESE = "raw_data/chinese"
WEBNLG = "raw_data/EWebNLG"


def chinese_read_line(line: str) -> Tuple[str, int]:
    line = line.strip("\n")
    if not line:
        return None, 0
    else:
        instance = json.loads(line)
        line = line + "\n"
        triplet = instance["spo_list"]
        return line, len(triplet)


def webnlg_read_line(line: str) -> Tuple[str, int]:
    line = line.strip("\n")
    if not line:
        return None, 0
    else:
        instance = json.loads(line)
        text = instance["target"]
        ner2ent = instance["ner2ent"]
        # TODO: replace?

        triples = instance["triples"]
        triples = [
            {"subject": t[0], "predicate": t[1], "object": t[2]} for t in triples
        ]
        line = json.dumps({"spo_list": triples, "text": text})
        line = line + "\n"
        return line, len(triples)


def data_split(dataset: str, data_root: str):
    if dataset == "chinese":
        _split_cond(cond="chinese_train_eval", data_root=data_root)
        _split_cond(cond="chinese_test", data_root=data_root)
        _split_cond(cond="test1", data_root=data_root)
        _split_cond(cond="test2", data_root=data_root)
        _split_cond(cond="test3", data_root=data_root)
        _split_cond(cond="test4", data_root=data_root)
        _split_cond(cond="test5", data_root=data_root)
    elif dataset == "webnlg":
        _split_cond(cond="webnlg_train", data_root=data_root)
        _split_cond(cond="webnlg_eval", data_root=data_root)
        _split_cond(cond="webnlg_test", data_root=data_root)
        _split_cond(cond="test1", data_root=data_root)
        _split_cond(cond="test2", data_root=data_root)
        _split_cond(cond="test3", data_root=data_root)
        _split_cond(cond="test4", data_root=data_root)
        _split_cond(cond="test5", data_root=data_root)
    else:
        raise NotImplementedError("conll")


def _split_cond(cond: str, data_root: str):

    if cond == "chinese_test":
        source = os.path.join(data_root, "dev_data.json")
        target = os.path.join(data_root, "new_test_data.json")
        all_triplet_num = 0
        all_sent_num = 0
        with open(source, "r", encoding="utf-8") as s, open(
            target, "w", encoding="utf-8"
        ) as t:
            for line in s:
                line, triplet_num = chinese_read_line(line)
                t.write(line)
                all_triplet_num += triplet_num
                all_sent_num += 1
        print("chinese test sent %d, triplet %d" % (all_sent_num, all_triplet_num))

    elif cond == "chinese_train_eval":
        source = os.path.join(data_root, "train_data.json")
        train = os.path.join(data_root, "new_train_data.json")
        validate = os.path.join(data_root, "new_validate_data.json")

        train_sent = 0
        train_triplet = 0
        validate_sent = 0
        validate_triplet = 0

        with open(source, "r", encoding="utf-8") as s, open(
            train, "w", encoding="utf-8"
        ) as t, open(validate, "w", encoding="utf-8") as v:
            for line in s:
                line, triplet_num = chinese_read_line(line)
                if random.random() < 0.9:
                    t.write(line)
                    train_sent += 1
                    train_triplet += triplet_num
                else:
                    v.write(line)
                    validate_sent += 1
                    validate_triplet += triplet_num
        print("chinese train sent %d, triplet %d" % (train_sent, train_triplet))
        print(
            "chinese validate sent %d, triplet %d" % (validate_sent, validate_triplet)
        )

    elif cond == "webnlg_train":
        source = os.path.join(data_root, "train_data.json")
        target = os.path.join(data_root, "new_train_data.json")
        all_triplet_num = 0
        all_sent_num = 0
        with open(source, "r", encoding="utf-8") as s, open(
            target, "w", encoding="utf-8"
        ) as t:
            for line in s:
                line, triplet_num = webnlg_read_line(line)
                t.write(line)
                all_triplet_num += triplet_num
                all_sent_num += 1
        print("webnlg train sent %d, triplet %d" % (all_sent_num, all_triplet_num))

    elif cond == "webnlg_eval":
        source = os.path.join(data_root, "dev_data.json")
        target = os.path.join(data_root, "new_validate_data.json")
        all_triplet_num = 0
        all_sent_num = 0
        with open(source, "r", encoding="utf-8") as s, open(
            target, "w", encoding="utf-8"
        ) as t:
            for line in s:
                line, triplet_num = webnlg_read_line(line)
                t.write(line)
                all_triplet_num += triplet_num
                all_sent_num += 1
        print("webnlg dev sent %d, triplet %d" % (all_sent_num, all_triplet_num))

    elif cond == "webnlg_test":
        source = os.path.join(data_root, "test_data.json")
        target = os.path.join(data_root, "new_test_data.json")
        all_triplet_num = 0
        all_sent_num = 0
        with open(source, "r", encoding="utf-8") as s, open(
            target, "w", encoding="utf-8"
        ) as t:
            for line in s:
                line, triplet_num = webnlg_read_line(line)
                t.write(line)
                all_triplet_num += triplet_num
                all_sent_num += 1
        print("webnlg test sent %d, triplet %d" % (all_sent_num, all_triplet_num))

    elif data_root == CHINESE and cond in ["test1", "test2", "test3", "test4", "test5"]:
        source = os.path.join(data_root, "dev_data.json")
        target = os.path.join(data_root, "new_" + cond + ".json")

        all_triplet_num = 0
        all_sent_num = 0

        with open(source, "r", encoding="utf-8") as s, open(
            target, "w", encoding="utf-8"
        ) as t:
            for line in s:
                line, triplet_num = chinese_read_line(line)
                if cond == "test5" and triplet_num >= 5:
                    t.write(line)
                    all_triplet_num += triplet_num
                    all_sent_num += 1
                elif triplet_num == int(cond[-1]):
                    t.write(line)
                    all_triplet_num += triplet_num
                    all_sent_num += 1
                else:
                    pass
        print(
            data_root
            + " "
            + cond
            + " sent %d, triplet %d" % (all_sent_num, all_triplet_num)
        )

    elif data_root == WEBNLG and cond in ["test1", "test2", "test3", "test4", "test5"]:
        source = os.path.join(data_root, "test_data.json")
        target = os.path.join(data_root, "new_" + cond + ".json")

        all_triplet_num = 0
        all_sent_num = 0

        with open(source, "r", encoding="utf-8") as s, open(
            target, "w", encoding="utf-8"
        ) as t:
            for line in s:
                line, triplet_num = webnlg_read_line(line)
                if cond == "test5" and triplet_num >= 5:
                    t.write(line)
                    all_triplet_num += triplet_num
                    all_sent_num += 1
                elif triplet_num == int(cond[-1]):
                    t.write(line)
                    all_triplet_num += triplet_num
                    all_sent_num += 1
                else:
                    pass
        print(
            data_root
            + " "
            + cond
            + " sent %d, triplet %d" % (all_sent_num, all_triplet_num)
        )


if __name__ == "__main__":

    data_split("chinese", CHINESE)
    # data_split("webnlg", WEBNLG)
