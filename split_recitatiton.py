from collections import Counter
import json
import os

data_root_list = [
    "data/chinese/seq2umt_ops",
    "data/chinese/wdec",
    "data/nyt/wdec",
    "data/nyt/seq2umt_ops",
]

wdec_nyt_root = "data/nyt/wdec"
wdec_chinese_root = "data/chinese/wdec"

# triplet = s p o


def cnt_train(data_root):
    triplet_cnt = Counter()
    source = os.path.join(data_root, "new_train_data.json")

    with open(source, "r") as s:
        for line in s:
            line = json.loads(line)
            spo_list = line["spo_list"]
            spo = [(t["subject"], t["predicate"], t["object"]) for t in spo_list]
            triplet_cnt.update(spo)

    return triplet_cnt


def filter_test(data_root, dataset, cnt):
    source = os.path.join(data_root, dataset)
    target = os.path.join(data_root, "filter" + dataset)
    write_linenum = 0
    with open(source, "r", encoding="utf-8") as s, open(
        target, "w", encoding="utf-8"
    ) as t:
        for all_linenum, line in enumerate(s):
            jline = json.loads(line)
            spo_list = jline["spo_list"]
            if check_valid(spo_list, cnt):
                t.write(line)
                write_linenum += 1
    print(data_root + " valid sent / all sent = %d/%d" % (write_linenum, all_linenum))
    return write_linenum / all_linenum


def check_valid(spo_list, cnt):
    MAX = 3
    spo = [(t["subject"], t["predicate"], t["object"]) for t in spo_list]
    spo_freq = [cnt[tri] < MAX for tri in spo]

    return all(spo_freq)


def process_all(data_root):
    cnt = cnt_train(data_root)
    filter_test(data_root, "new_test_data.json", cnt)


if __name__ == "__main__":

    for root in data_root_list:
        process_all(root)
