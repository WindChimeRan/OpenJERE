from collections import Counter
import json
import os

data_root_list = [
    "data/chinese/seq2umt_ops",
    "data/chinese/wdec",
    "data/nyt/wdec",
    "data/nyt/seq2umt_ops",
    "data/nyt/seq2umt_pos",
]

wdec_nyt_root = "data/nyt/wdec"
wdec_chinese_root = "data/chinese/wdec"

# triplet = s p o


def cnt_train_key(
    data_root, key_fn=lambda l: [(t["subject"], t["predicate"], t["object"]) for t in l]
):
    triplet_cnt = Counter()
    source = os.path.join(data_root, "new_train_data.json")

    with open(source, "r") as s:
        for line in s:
            line = json.loads(line)
            spo_list = line["spo_list"]
            # spo = [(t["subject"], t["predicate"], t["object"]) for t in spo_list]
            spo = key_fn(spo_list)
            triplet_cnt.update(spo)

    return triplet_cnt


def filter_test(data_root, dataset, cnt, thr: int):
    # filter the sub test set whose freq less than thr
    source = os.path.join(data_root, dataset)
    target = os.path.join(data_root, "filter_" + str(thr) + "_" + dataset)
    write_linenum = 0
    with open(source, "r", encoding="utf-8") as s, open(
        target, "w", encoding="utf-8"
    ) as t:
        for all_linenum, line in enumerate(s):
            jline = json.loads(line)
            spo_list = jline["spo_list"]
            if check_thr(spo_list, cnt, thr):
                t.write(line)
                write_linenum += 1
    print(data_root + " valid sent / all sent = %d/%d" % (write_linenum, all_linenum))
    return write_linenum / all_linenum


def filter_partial_triplet(data_root, dataset, cnt, fn):
    # filter the sub test set whose freq less than thr
    source = os.path.join(data_root, dataset)
    all_triplet_num = 0
    all_valid_num = 0
    with open(source, "r", encoding="utf-8") as s:
        for all_linenum, line in enumerate(s):
            jline = json.loads(line)
            spo_list = jline["spo_list"]
            spo_list = fn(spo_list)
            all_triplet_num += len(spo_list)
            valid_list = [cnt[spo] > 0 for spo in spo_list]
            all_valid_num += sum(valid_list)

    print(
        data_root
        + " valid sent / all sent = %d/%d = %.3f"
        % (all_valid_num, all_triplet_num, all_valid_num / all_triplet_num)
    )


def check_num(spo_list, fn):
    # MAX = 3
    spo = [(t["subject"], t["predicate"], t["object"]) for t in spo_list]
    return fn(len(spo))


def check_thr(spo_list, cnt, MAX):
    # MAX = 3
    spo = [(t["subject"], t["predicate"], t["object"]) for t in spo_list]
    spo_freq = [cnt[tri] < MAX for tri in spo]

    return all(spo_freq)


if __name__ == "__main__":
    nyt_list = [
        "data/nyt/wdec",
        "data/nyt/seq2umt_pos",
        "data/nyt/multi_head_selection",
    ]
    test = nyt_list[0]
    sp_fn = lambda l: [(t["subject"], t["predicate"]) for t in l]
    spo_fn = lambda l: [(t["subject"], t["predicate"], t["object"]) for t in l]

    cnt = cnt_train_key(test, key_fn=sp_fn)
    filter_partial_triplet(test, "new_test_data.json", cnt, fn=sp_fn)
