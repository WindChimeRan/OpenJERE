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
    return key_list(lines, fn=lambda t: t["spo_list"])


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
    train, dev, test = list(yield_line(train)), list(yield_line(dev)), list(yield_line(test))

    ltrain, ldev, ltest = map(len, (train, dev, test))
    print("length of the datasets train %d \t dev %d\t test %d" % (ltrain, ldev, ltest))

    all_data = train + dev + test
    if shuffle:
        random.shuffle(all_data)

    train = all_data[:ltrain]
    dev = all_data[ltrain:-ltest]
    test = all_data[-ltest:]

    assert len(train) == ltrain
    assert len(dev)   == ldev
    assert len(test)  == ltest

    return train, dev, test

def check_instance_overlap():
    pass

if __name__ == "__main__":
    nyt_data_root = "raw_data/nyt/"

    data_root = nyt_data_root

    ptest = data_root + "new_test_data.json"

    ptrain = data_root + "new_train_data.json"

    pdev = data_root + "new_validate_data.json"

    train, dev, test = read_data(ptrain, pdev, ptest, shuffle=False)

    print(data_root + ' '+ "dev triplet overlap train = %.2f" % triplet_percentage_test_of_train(train, dev))

    print(data_root + ' '+ "test triplet overlap train = %.2f" % triplet_percentage_test_of_train(train, test))

    print(data_root + ' '+ "dev text overlap train = %.2f" % text_percentage_test_of_train(train, dev))

    print(data_root + ' '+ "test text overlap train = %.2f" % text_percentage_test_of_train(train, test))

    print("turn on SHUFFULE")
    train, dev, test = read_data(ptrain, pdev, ptest, shuffle=True)

    print(data_root + ' '+ "dev triplet overlap train = %.2f" % triplet_percentage_test_of_train(train, dev))

    print(data_root + ' '+ "test triplet overlap train = %.2f" % triplet_percentage_test_of_train(train, test))

    print(data_root + ' '+ "dev text overlap train = %.2f" % text_percentage_test_of_train(train, dev))

    print(data_root + ' '+ "test text overlap train = %.2f" % text_percentage_test_of_train(train, test))