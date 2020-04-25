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


def filter_test_num(data_root, dataset, cnt, thr: int, num: int, fn):
    # filter the sub test set whose freq less than thr
    source = os.path.join(data_root, dataset)
    target = os.path.join(data_root, "new" + str(thr) + "_" + str(num) + "_" + dataset)
    write_linenum = 0
    with open(source, "r", encoding="utf-8") as s, open(
        target, "w", encoding="utf-8"
    ) as t:
        for all_linenum, line in enumerate(s):
            jline = json.loads(line)
            spo_list = jline["spo_list"]
            if check_thr(spo_list, cnt, thr) and check_num(spo_list, fn):
                t.write(line)
                write_linenum += 1
    print(data_root + " valid sent / all sent = %d/%d" % (write_linenum, all_linenum))
    return write_linenum / all_linenum


def filter_partial_triplet(data_root, dataset, cnt, thr: int, num: int, fn):
    # filter the sub test set whose freq less than thr
    source = os.path.join(data_root, dataset)
    target = os.path.join(data_root, "new" + str(thr) + "_" + str(num) + "_" + dataset)
    write_linenum = 0
    with open(source, "r", encoding="utf-8") as s, open(
        target, "w", encoding="utf-8"
    ) as t:
        for all_linenum, line in enumerate(s):
            jline = json.loads(line)
            spo_list = jline["spo_list"]
            if check_thr(spo_list, cnt, thr) and check_num(spo_list, fn):
                t.write(line)
                write_linenum += 1
    print(data_root + " valid sent / all sent = %d/%d" % (write_linenum, all_linenum))
    return write_linenum / all_linenum


def check_num(spo_list, fn):
    # MAX = 3
    spo = [(t["subject"], t["predicate"], t["object"]) for t in spo_list]
    return fn(len(spo))


def check_thr(spo_list, cnt, MAX):
    # MAX = 3
    spo = [(t["subject"], t["predicate"], t["object"]) for t in spo_list]
    spo_freq = [cnt[tri] < MAX for tri in spo]

    return all(spo_freq)


def process_all(data_root):
    cnt = cnt_train(data_root)
    filter_test(data_root, "new_test_data.json", cnt, 10)
    filter_test(data_root, "new_test_data.json", cnt, 9)
    filter_test(data_root, "new_test_data.json", cnt, 8)
    filter_test(data_root, "new_test_data.json", cnt, 7)
    filter_test(data_root, "new_test_data.json", cnt, 6)
    filter_test(data_root, "new_test_data.json", cnt, 5)
    filter_test(data_root, "new_test_data.json", cnt, 4)
    filter_test(data_root, "new_test_data.json", cnt, 3)
    filter_test(data_root, "new_test_data.json", cnt, 2)
    filter_test(data_root, "new_test_data.json", cnt, 1)


def nyt_pro(data_root, cnt):
    # cnt = cnt_train(data_root)
    filter_test_num(
        data_root, "new_test_data.json", cnt, 10, 1, fn=lambda length: length == 1
    )
    filter_test_num(
        data_root, "new_test_data.json", cnt, 10, 2, fn=lambda length: length == 2
    )
    filter_test_num(
        data_root, "new_test_data.json", cnt, 10, 3, fn=lambda length: length >= 3
    )


if __name__ == "__main__":
    nyt_list = [
        "data/nyt/wdec",
        "data/nyt/seq2umt_pos",
        "data/nyt/multi_head_selection",
    ]
    cnt = cnt_train(nyt_list[1])
    for root in nyt_list:
        nyt_pro(root, cnt)

    # for root in data_root_list:
    #     process_all(root)
