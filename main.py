import os
import json
import time
import argparse
import random

import torch

from typing import Dict, List, Tuple, Set, Optional

# from prefetch_generator import BackgroundGenerator
BackgroundGenerator = lambda x: x
from tqdm import tqdm

from collections import Counter

from torch.optim import Adam, SGD

from lib.preprocessings import (
    Chinese_selection_preprocessing,
    Chinese_copymb_preprocessing,
    Chinese_twotagging_preprocessing,
    Chinese_seq2umt_preprocessing,
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
)
from lib.metrics import F1_triplet
from lib.models import MultiHeadSelection, CopyMB, Twotagging
from lib.config import Hyper

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_name",
    "-e",
    type=str,
    default="chinese_seq2umt_re",
    help="experiments/exp_name.json",
)
parser.add_argument(
    "--mode",
    "-m",
    type=str,
    default="train",
    help="preprocessing|train|evaluation|subevaluation|data_summary",
)
args = parser.parse_args()


class Runner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.model_dir = "saved_models"

        self.hyper = Hyper(os.path.join("experiments", self.exp_name + ".json"))

        self.gpu = self.hyper.gpu
        self.preprocessor = self._preprocessor(self.hyper.model)
        self.metrics = F1_triplet()
        self.optimizer = None
        self.model = None

        self.Dataset, self.Loader = self._init_loader(self.hyper.model)

    def _init_loader(self, name: str):

        if name == "selection":

            Dataset = Selection_Dataset
            Loader = Selection_loader
        elif name == "copymb":

            Dataset = Copymb_Dataset
            Loader = Copymb_loader
        elif name == "twotagging":

            Dataset = Twotagging_Dataset
            Loader = Twotagging_loader
        elif name == "seq2umt":

            Dataset = Seq2umt_Dataset
            Loader = Seq2umt_loader

        else:
            raise ValueError("wrong name!")
        return Dataset, Loader

    def _optimizer(self, name, model):
        m = {"adam": Adam(model.parameters()), "sgd": SGD(model.parameters(), lr=0.5)}
        return m[name]

    def _preprocessor(self, name: str):
        p = {
            "selection": Chinese_selection_preprocessing(self.hyper),
            "copymb": Chinese_copymb_preprocessing(self.hyper),
            "twotagging": Chinese_twotagging_preprocessing(self.hyper),
            "seq2umt": Chinese_seq2umt_preprocessing(self.hyper),
        }
        return p[name]

    def _init_model(self):
        print(self.hyper.model)
        if self.hyper.model == "selection":
            self.model = MultiHeadSelection(self.hyper).cuda(self.gpu)
        elif self.hyper.model == "copymb":
            self.model = CopyMB(self.hyper).cuda(self.gpu)
        elif self.hyper.model == "twotagging":
            self.model = Twotagging(self.hyper).cuda(self.gpu)
        elif self.hyper.model == "seq2umt":
            self.model = NotImplementedError("seq2umt not implemented!")
        else:
            raise NotImplementedError("Future works!")

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
            self.load_model(epoch=self.hyper.evaluation_epoch)
            dev_set = self.DevDataset(self.hyper, self.hyper.dev)
            loader = self.DevLoader(
                dev_set,
                batch_size=self.hyper.batch_size_eval,
                pin_memory=True,
                num_workers=8,
            )
            self.evaluation(loader)

        elif mode == "data_summary":
            self.hyper.vocab_init()
            for path in self.hyper.raw_data_list:
                self.summary_data(path)
        elif mode == "subevaluation":
            raise NotImplementedError("subevaluation")

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

                    print(sample.text)
                    print(sample.R_gt)
                    print(sample.R_in)
                    exit()

        else:
            raise ValueError("invalid mode")

    def load_model(self, epoch: int):
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_dir, self.exp_name + "_" + str(epoch)))
        )

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, self.exp_name + "_" + str(epoch)),
        )

    def summary_data(self, dataset):
        # TODO
        data = self.Dataset(self.hyper, dataset)
        loader = self.Loader(data, batch_size=400, pin_memory=True, num_workers=4)

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        len_sent_list = []
        triplet_num = []

        for batch_ndx, sample in pbar:
            len_sent_list.extend(sample.length)
            triplet_num.extend(list(map(len, sample.spo_gold)))
        print(dataset)
        print("sentence num %d" % len(len_sent_list))
        print("avg sentence length %f" % (sum(len_sent_list) / len(len_sent_list)))
        print("avg triplet num %f" % (sum(triplet_num) / len(triplet_num)))
        print(Counter(triplet_num))
        print("\n")

    def evaluation(self, loader):
        # dev_set = self.DevDataset(self.hyper, self.hyper.dev)
        # loader = self.DevLoader(
        #     dev_set,
        #     batch_size=self.hyper.batch_size_eval,
        #     pin_memory=True,
        #     num_workers=4,
        # )
        self.model.metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)

                # # DEBUG: error analysis

                # for g, p in zip(output['spo_gold'], output['decode_result']):
                #     print(g)
                #     print(p)
                #     print('-'*50)
                # exit()

                self.model.run_metrics(output)

            result = self.model.get_metric()
            print(
                ", ".join(
                    [
                        "%s: %.4f" % (name, value)
                        for name, value in result.items()
                        if not name.startswith("_")
                    ]
                )
                + " ||"
            )

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
                self.optimizer.step()

                pbar.set_description(output["description"](epoch, self.hyper.epoch_num))

            self.save_model(epoch)
            if epoch % self.hyper.print_epoch == 0 and epoch > 2:
                self.evaluation(dev_loader)


if __name__ == "__main__":
    config = Runner(exp_name=args.exp_name)
    config.run(mode=args.mode)
