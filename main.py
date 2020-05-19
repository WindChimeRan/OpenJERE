import os
import json
import time
import argparse
import random
import logging

import torch
import numpy as np

from typing import Dict, List, Tuple, Set, Optional

# from torchsummary import summary
# from prefetch_generator import BackgroundGenerator
BackgroundGenerator = lambda x: x
# print = logging.info


from tqdm import tqdm

from collections import Counter

from torch.optim import Adam, SGD

from lib.preprocessings import (
    Selection_preprocessing,
    Copymb_preprocessing,
    Twotagging_preprocessing,
    Seq2umt_preprocessing,
    WDec_preprocessing,
    Copymtl_preprocessing,
)

from lib.dataloaders import (
    Selection_Dataset,
    Selection_loader,
    Copymb_Dataset,
    Copymb_loader,
    Twotagging_Dataset,
    Twotagging_loader,
    Seq2umt_Dataset,
    Seq2umt_loader,
    WDec_Dataset,
    WDec_loader,
    Copymtl_Dataset,
    Copymtl_loader,
)
from lib.metrics import F1_triplet
from lib.models import (
    MultiHeadSelection,
    CopyMB,
    Twotagging,
    Seq2umt,
    Threetagging,
    WDec,
    CopyMTL,
)
from lib.config import Hyper

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_name",
    "-e",
    type=str,
    default="chinese_seq2umt",
    help="experiments/exp_name.json",
)
parser.add_argument(
    "--mode",
    "-m",
    type=str,
    default="train",
    help="preprocessing|train|evaluation|subevaluation|data_summary|model_summary",
)
args = parser.parse_args()


class Runner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.model_dir = "saved_models"

        self.hyper = Hyper(os.path.join("experiments", self.exp_name + ".json"))

        self.gpu = self.hyper.gpu
        self.preprocessor = self._preprocessor(self.hyper.model)
        # self.metrics = F1_triplet()
        self.optimizer = None
        self.model = None

        self.Dataset, self.Loader = self._init_loader(self.hyper.model)

        logging.basicConfig(
            filename=os.path.join("experiments", self.exp_name + ".log"),
            filemode="w",
            format="%(asctime)s - %(message)s",
            level=logging.INFO,
        )

    def _init_loader(self, name: str):

        dataset_dic = {
            "selection": Selection_Dataset,
            "copymb": Copymb_Dataset,
            "twotagging": Twotagging_Dataset,
            "seq2umt": Seq2umt_Dataset,
            "wdec": WDec_Dataset,
            "copymtl": Copymtl_Dataset,
        }

        loader_dic = {
            "selection": Selection_loader,
            "copymb": Copymb_loader,
            "twotagging": Twotagging_loader,
            "seq2umt": Seq2umt_loader,
            "wdec": WDec_loader,
            "copymtl": Copymtl_loader,
        }

        Dataset = dataset_dic[name]
        Loader = loader_dic[name]

        if name not in dataset_dic or name not in loader_dic:
            raise ValueError("wrong name!")
        else:
            return Dataset, Loader

    def _optimizer(self, name, model):
        m = {"adam": Adam(model.parameters()), "sgd": SGD(model.parameters(), lr=0.5)}
        return m[name]

    def _preprocessor(self, name: str):
        p = {
            "selection": Selection_preprocessing(self.hyper),
            "copymb": Copymb_preprocessing(self.hyper),
            "twotagging": Twotagging_preprocessing(self.hyper),
            "seq2umt": Seq2umt_preprocessing(self.hyper),
            "threetagging": Seq2umt_preprocessing(self.hyper),
            "wdec": WDec_preprocessing(self.hyper),
            "copymtl": Copymtl_preprocessing(self.hyper),
        }
        return p[name]

    def _init_model(self):
        logging.info(self.hyper.model)
        name = self.hyper.model
        p = {
            "selection": MultiHeadSelection,
            "copymb": CopyMB,
            "twotagging": Twotagging,
            "seq2umt": Seq2umt,
            "threetagging": Seq2umt,
            "wdec": WDec,
            "copymtl": CopyMTL,
        }
        self.model = p[name](self.hyper).cuda(self.gpu)

    def preprocessing(self):
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=2)
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self, mode: str):
        if mode == "preprocessing":
            self.preprocessing()
        elif mode == "train":
            self.hyper.vocab_init()
            self._init_model()
            self.optimizer = self._optimizer(self.hyper.optimizer, self.model)
            self.train()
        elif mode == "evaluation":
            self.hyper.vocab_init()
            self._init_model()
            # self.load_model(str(self.hyper.evaluation_epoch))
            self.load_model("best")
            # self.load_model("2")
            test_set = self.Dataset(self.hyper, self.hyper.test)
            loader = self.Loader(
                test_set,
                batch_size=self.hyper.batch_size_eval,
                pin_memory=True,
                num_workers=8,
            )
            f1, log = self.evaluation(loader)
            print(log)
            print("f1 = ", f1)

        elif mode == "data_summary":
            self.hyper.vocab_init()
            # for path in self.hyper.raw_data_list:
            #     self.summary_data(path)
            self.summary_data(self.hyper.test)
        elif mode == "model_summary":
            self.hyper.vocab_init()
            self._init_model()
            self.load_model("best")
            parameter_num = np.sum([p.numel() for p in self.model.parameters()]).item()
            print(self.model)
            print(parameter_num)

        elif mode == "subevaluation":
            self.hyper.vocab_init()
            self._init_model()
            self.load_model("best")
            for data in self.hyper.subsets:
                test_set = self.Dataset(self.hyper, data)
                loader = self.Loader(
                    test_set,
                    batch_size=self.hyper.batch_size_eval,
                    pin_memory=True,
                    num_workers=8,
                )
                f1, log = self.evaluation(loader)
                print(log)
                print("f1 = ", f1)
            # raise NotImplementedError("subevaluation")

        elif mode == "debug":
            self.hyper.vocab_init()
            # self._init_model()
            train_set = self.Dataset(self.hyper, self.hyper.dev)
            loader = self.Loader(
                train_set,
                batch_size=self.hyper.batch_size_train,
                pin_memory=True,
                num_workers=0,
            )
            for epoch in range(self.hyper.epoch_num):
                pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

                for batch_idx, sample in pbar:

                    print(sample.__dict__)
                    exit()

        else:
            raise ValueError("invalid mode")

    def load_model(self, name: str):
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_dir, self.exp_name + "_" + name))
        )

    def save_model(self, name: str):
        # def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, self.exp_name + "_" + name),
        )

    def summary_data(self, dataset):
        def count_numsubset_overlap(triplets: List[Dict[str, str]], n: int = 2) -> None:
            def _eq_tripletsNum_n(triplets, n):
                return len(triplets) == n

            def _is_olp(ts):
                ent_list = [t["subject"] for t in ts] + [t["object"] for t in ts]
                ent_set = set(ent_list)
                return len(ent_list) != len(ent_set)

            cnt_all = 0
            cnt_olp = 0
            for ts in triplets:
                if _eq_tripletsNum_n(ts, n):
                    cnt_all += 1
                    if _is_olp(ts):
                        cnt_olp += 1
            log = (
                "num = %d, all sentence = %d, overlapped sentence = %d, rate = %.2f"
                % (n, cnt_all, cnt_olp, cnt_olp / cnt_all * 100)
            )
            print(log)

        # TODO
        data = self.Dataset(self.hyper, dataset)
        loader = self.Loader(data, batch_size=400, pin_memory=True, num_workers=4)

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        len_sent_list = []
        triplet_num = []
        triplets = []

        ## count_numsubset_overlap
        # for batch_ndx, sample in pbar:
        #     len_sent_list.extend(sample.length)
        #     triplet_num.extend(list(map(len, sample.spo_gold)))
        #     triplets.extend(sample.spo_gold)
        # count_numsubset_overlap(triplets, 2)
        # exit()
        print(dataset)
        print("sentence num %d" % len(len_sent_list))
        print("all triplet num %d" % sum(triplet_num))
        print("avg sentence length %f" % (sum(len_sent_list) / len(len_sent_list)))
        print("avg triplet num %f" % (sum(triplet_num) / len(triplet_num)))
        print(Counter(triplet_num))
        print("\n")

    def evaluation(self, loader):
        self.model.metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)

                # # DEBUG: error analysis

                # # WDec
                # for text, g, p, seq, in zip(
                #     output["text"],
                #     output["spo_gold"],
                #     output["decode_result"],
                #     output["seq"],
                # ):
                #     print(text)
                #     print(g)
                #     print(p)
                #     print(seq)
                #     print("-" * 50)
                # exit()

                # Seq2UMT
                # for text, g, p in zip(
                #     output["text"], output["spo_gold"], output["decode_result"],
                # ):
                #     print(text)
                #     print(g)
                #     print(p)
                # exit()

                self.model.run_metrics(output)

        result = self.model.get_metric()
        # print(
        #     ", ".join(
        #         [
        #             "%s: %.4f" % (name, value)
        #             for name, value in result.items()
        #             if not name.startswith("_")
        #         ]
        #     )
        #     + " ||"
        # )
        score = result["fscore"]
        log = (
            ", ".join(
                [
                    "%s: %.4f" % (name, value)
                    for name, value in result.items()
                    if not name.startswith("_")
                ]
            )
            + " ||"
        )
        return result["fscore"], log

    def train(self):

        train_set = self.Dataset(self.hyper, self.hyper.train)
        train_loader = self.Loader(
            train_set,
            batch_size=self.hyper.batch_size_train,
            pin_memory=True,
            num_workers=8,
        )
        dev_set = self.Dataset(self.hyper, self.hyper.dev)
        dev_loader = self.Loader(
            dev_set,
            batch_size=self.hyper.batch_size_eval,
            pin_memory=True,
            num_workers=4,
        )
        test_set = self.Dataset(self.hyper, self.hyper.test)
        test_loader = self.Loader(
            test_set,
            batch_size=self.hyper.batch_size_eval,
            pin_memory=True,
            num_workers=4,
        )
        score = 0
        best_epoch = 0
        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(
                enumerate(BackgroundGenerator(train_loader)), total=len(train_loader)
            )

            for batch_idx, sample in pbar:
                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)

                loss = output["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

                self.optimizer.step()

                pbar.set_description(output["description"](epoch, self.hyper.epoch_num))

            self.save_model(str(epoch))
            if epoch % self.hyper.print_epoch == 0 and epoch >= 2:
                new_score, log = self.evaluation(dev_loader)
                logging.info(log)
                if new_score >= score:
                    score = new_score
                    best_epoch = epoch
                    self.save_model("best")
        logging.info("best epoch: %d \t F1 = %.2f" % (best_epoch, score))
        self.load_model("best")
        new_score, log = self.evaluation(test_loader)
        logging.info(log)


if __name__ == "__main__":
    config = Runner(exp_name=args.exp_name)
    config.run(mode=args.mode)
