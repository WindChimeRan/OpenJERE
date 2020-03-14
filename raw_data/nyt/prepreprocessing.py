import json


def yield_line(source: str):
    with open(source, "r", encoding="utf-8") as s:
        for line in s:
            line = line.strip("\n")
            if not line:
                continue
            else:
                yield line


def process_line(line):

    ins = json.loads(line)
    text = ins["sentText"]
    spo_list = ins["relationMentions"]
    spo_list = [
        {"subject": t["em1Text"], "predicate": t["em2Text"], "object": t["label"]}
        for t in spo_list
    ]
    new_dic = {"text": text, "spo_list": spo_list}
    new_line = json.dumps(new_dic) + "\n"

    return new_line


def reformat(source, target):
    with open(target, "w", encoding="utf-8") as t:
        for line in yield_line(source):
            new_line = process_line(line)
            t.write(new_line)


if __name__ == "__main__":
    otest = "./raw_test.json"
    ttest = "./new_test_data.json"

    otrain = "./raw_train.json"
    ttrain = "./new_train_data.json"

    odev = "./raw_valid.json"
    tdev = "./new_validate_data.json"

    reformat(otest, ttest)
    reformat(otrain, ttrain)
    reformat(tdev, tdev)
