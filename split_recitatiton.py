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
    # nyt: 1, 2, 3, when freq <= 10
    filter_test_num(
        data_root, "new_test_data.json", cnt, 10, 1, fn=lambda length: length == 1
    )
    filter_test_num(
        data_root, "new_test_data.json", cnt, 10, 2, fn=lambda length: length == 2
    )
    filter_test_num(
        data_root, "new_test_data.json", cnt, 10, 3, fn=lambda length: length >= 3
    )


def nyt_ab_test(data_root):
    s_set, _ = split_test(data_root)
    split_train(root, s_set)


def dic2tuple(dic):
    return (dic["subject"], dic["predicate"], dic["object"])


def split_test(data_root):
    source = os.path.join(data_root, "new_test_data.json")
    target_o = os.path.join(data_root, "test_o.json")
    target_no = os.path.join(data_root, "test_no.json")

    s_set = set()  # seen
    ns_set = set()  # unseen
    seen_cnt = 0
    unseen_cnt = 0
    with open(source, "r", encoding="utf-8") as s, open(
        target_o, "w", encoding="utf-8"
    ) as tseen, open(target_no, "w", encoding="utf-8") as tnseen:

        for all_linenum, line in enumerate(s):
            jline = json.loads(line)
            spo_list = list(map(dic2tuple, jline["spo_list"]))
            if any([spo in s_set for spo in spo_list]):  # any spo are in the seen set
                tseen.write(line)  # write seen line
                s_set.update(spo_list)  # update seen set
                seen_cnt += 1
            else:  # all spo are not in unseen set
                tnseen.write(line)  # write unseen test with line
                s_set.update(spo_list)  # update unseen set with spos
                unseen_cnt += 1
    print(seen_cnt, unseen_cnt)
    return s_set, ns_set


def split_train(data_root, s_set):
    source = os.path.join(data_root, "new_train_data.json")
    target_o = os.path.join(data_root, "train_o.json")
    cnt = 0
    with open(source, "r", encoding="utf-8") as s, open(
        target_o, "w", encoding="utf-8"
    ) as t:
        for all_linenum, line in enumerate(s):
            jline = json.loads(line)
            spo_list = list(map(dic2tuple, jline["spo_list"]))
            if all([spo in s_set for spo in spo_list]):  # any sop are in the seen set
                cnt += 1
                t.write(line)
            else:
                pass
        print("train cnt", cnt)


if __name__ == "__main__":
    nyt_list = [
        "data/nyt/wdec",
        "data/nyt/seq2umt_pos",
        # "data/nyt/multi_head_selection",
    ]
    cnt = cnt_train(nyt_list[1])
    for root in nyt_list:
        # nyt_pro(root, cnt)
        nyt_ab_test(root)
    # for root in data_root_list:
    #     process_all(root)
