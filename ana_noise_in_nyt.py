import json
import os
import random
from typing import Dict, List, Tuple, Set, Optional


def yield_line(source: str):
    with open(source, "r", encoding="utf-8") as s:
        for line in s:
            line = line.strip("\n")
            if not line:
                continue
            else:
                yield line


def key_list(lines, fn):
    all_triplets = []
    for line in lines:
        line = json.loads(line)
        values = fn(line)
        all_triplets.extend(values)
    return all_triplets


def triplet_list(lines):
    return key_list(
        lines,
        fn=lambda t: [
            (spo["subject"], spo["predicate"], spo["object"]) for spo in t["spo_list"]
        ],
    )


def text_list(lines):
    return key_list(lines, fn=lambda t: [t["text"]])


def percentage_test_of_train(source, target, fn):

    s_list = fn(source)
    t_list = fn(target)
    one = [t in s_list for t in t_list]
    # print(len(one))
    percentage = sum(one) / len(one)

    return percentage


def triplet_percentage_test_of_train(source, target):
    return percentage_test_of_train(source, target, fn=triplet_list)


def text_percentage_test_of_train(source, target):
    return percentage_test_of_train(source, target, fn=text_list)


def read_data(train, dev, test, shuffle=False):
    train, dev, test = (
        list(yield_line(train)),
        list(yield_line(dev)),
        list(yield_line(test)),
    )

    ltrain, ldev, ltest = map(len, (train, dev, test))
    print("length of the datasets train %d \t dev %d\t test %d" % (ltrain, ldev, ltest))

    all_data = train + dev + test
    if shuffle:
        random.shuffle(all_data)

    train = all_data[:ltrain]
    dev = all_data[ltrain:-ltest]
    test = all_data[-ltest:]

    assert len(train) == ltrain
    assert len(dev) == ldev
    assert len(test) == ltest

    return train, dev, test


def log_order(data_root):
    ptest = data_root + "new_test_data.json"

    ptrain = data_root + "new_train_data.json"

    pdev = data_root + "new_validate_data.json"

    train, dev, test = read_data(ptrain, pdev, ptest, shuffle=False)

    print(
        data_root
        + " "
        + "dev triplet overlap train = %.3f"
        % triplet_percentage_test_of_train(train, dev)
    )

    print(
        data_root
        + " "
        + "test triplet overlap train = %.3f"
        % triplet_percentage_test_of_train(train, test)
    )

    print(
        data_root
        + " "
        + "op test triplet overlap train = %.3f"
        % percentage_test_of_train(
            train,
            test,
            fn=lambda lines: key_list(
                lines,
                fn=lambda t: [
                    (spo["predicate"], spo["object"]) for spo in t["spo_list"]
                ],
            ),
        )
    )

    print(
        data_root
        + " "
        + "so test triplet overlap train = %.3f"
        % percentage_test_of_train(
            train,
            test,
            fn=lambda lines: key_list(
                lines,
                fn=lambda t: [(spo["subject"], spo["object"]) for spo in t["spo_list"]],
            ),
        )
    )
    print(
        data_root
        + " "
        + "sp test triplet overlap train = %.3f"
        % percentage_test_of_train(
            train,
            test,
            fn=lambda lines: key_list(
                lines,
                fn=lambda t: [
                    (spo["subject"], spo["predicate"]) for spo in t["spo_list"]
                ],
            ),
        )
    )


def log_data(data_root):
    ptest = data_root + "new_test_data.json"

    ptrain = data_root + "new_train_data.json"

    pdev = data_root + "new_validate_data.json"

    train, dev, test = read_data(ptrain, pdev, ptest, shuffle=False)

    print(
        data_root
        + " "
        + "dev triplet overlap train = %.2f"
        % triplet_percentage_test_of_train(train, dev)
    )

    print(
        data_root
        + " "
        + "test triplet overlap train = %.2f"
        % triplet_percentage_test_of_train(train, test)
    )

    print(
        data_root
        + " "
        + "dev text overlap train = %.2f" % text_percentage_test_of_train(train, dev)
    )

    print(
        data_root
        + " "
        + "test text overlap train = %.2f" % text_percentage_test_of_train(train, test)
    )

    print("turn on SHUFFULE")
    train, dev, test = read_data(ptrain, pdev, ptest, shuffle=True)

    print(
        data_root
        + " "
        + "dev triplet overlap train = %.2f"
        % triplet_percentage_test_of_train(train, dev)
    )

    print(
        data_root
        + " "
        + "test triplet overlap train = %.2f"
        % triplet_percentage_test_of_train(train, test)
    )

    print(
        data_root
        + " "
        + "dev text overlap train = %.2f" % text_percentage_test_of_train(train, dev)
    )

    print(
        data_root
        + " "
        + "test text overlap train = %.2f" % text_percentage_test_of_train(train, test)
    )


if __name__ == "__main__":
    nyt_data_root = "raw_data/nyt/"

    chinese_data_root = "raw_data/chinese/"

    # log_data(nyt_data_root)
    # log_data(nyt_data_root)
    log_order(chinese_data_root)

    """
    length of the datasets train 56196       dev 5000        test 5000
    raw_data/nyt/ dev triplet overlap train = 0.77
    raw_data/nyt/ test triplet overlap train = 0.77
    raw_data/nyt/ dev text overlap train = 0.04
    raw_data/nyt/ test text overlap train = 0.05
    turn on SHUFFULE
    length of the datasets train 56196       dev 5000        test 5000
    raw_data/nyt/ dev triplet overlap train = 0.79
    raw_data/nyt/ test triplet overlap train = 0.79
    raw_data/nyt/ dev text overlap train = 0.05
    raw_data/nyt/ test text overlap train = 0.04

    length of the datasets train 155794      dev 17315       test 21639
    raw_data/chinese/ dev triplet overlap train = 0.25
    raw_data/chinese/ test triplet overlap train = 0.24
    raw_data/chinese/ dev text overlap train = 0.00
    raw_data/chinese/ test text overlap train = 0.00
    turn on SHUFFULE
    length of the datasets train 155794      dev 17315       test 21639
    raw_data/chinese/ dev triplet overlap train = 0.25
    raw_data/chinese/ test triplet overlap train = 0.24
    raw_data/chinese/ dev text overlap train = 0.00
    raw_data/chinese/ test text overlap train = 0.00
    """
