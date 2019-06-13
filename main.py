import os
import json
import time
import argparse

import torch

from typing import Dict, List, Tuple, Set, Optional

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

from torch.optim import Adam, SGD

from lib.preprocessings import Chinese_selection_preprocessing
from lib.dataloaders import Selection_Dataset, Selection_loader
from lib.metrics import F1_triplet
from lib.models import MultiHeadSelection
from lib.config import Hyper

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',
                    '-e',
                    type=str,
                    default='chinese_selection_re',
                    help='experiments/exp_name.json')
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='preprocessing',
                    help='preprocessing|train|evaluation|data_summary')
args = parser.parse_args()


class Runner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.model_dir = 'saved_models'

        self.hyper = Hyper(os.path.join('experiments',
                                        self.exp_name + '.json'))

        self.gpu = self.hyper.gpu
        self.preprocessor = Chinese_selection_preprocessing(self.hyper)
        self.metrics = F1_triplet()
        self.optimizer = None
        self.model = None

    def _optimizer(self, name, model):
        m = {
            'adam': Adam(model.parameters()),
            'sgd': SGD(model.parameters(), lr=0.5)
        }
        return m[name]

    def _init_model(self):
        self.model = MultiHeadSelection(self.hyper).cuda(self.gpu)

    def preprocessing(self):
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self, mode: str):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(self.hyper.optimizer, self.model)
            self.train()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model(epoch=self.hyper.evaluation_epoch)
            self.evaluation()
        elif mode == 'data_summary':
            self.summary_data(self.hyper.train)
            self.summary_data(self.hyper.dev)
        else:
            raise ValueError('invalid mode')

    def load_model(self, epoch: int):
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir,
                             self.exp_name + '_' + str(epoch))))

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, self.exp_name + '_' + str(epoch)))
    
    def summary_data(self, dataset):
        data = Selection_Dataset(self.hyper, dataset)
        loader = Selection_loader(data, batch_size=400, pin_memory=True, num_workers=8)

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        len_sent_list = []
        triplet_num = []

        for batch_ndx, sample in pbar:
            len_sent_list.extend(sample.length)
            triplet_num.extend(list(map(len, sample.spo_gold)))

        print('sentence num %d' % len(len_sent_list))
        print('avg sentence length %f' % (sum(len_sent_list)/len(len_sent_list)))
        print('avg triplet num %f' % (sum(triplet_num)/len(triplet_num)))

    def evaluation(self):
        dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
        loader = Selection_loader(dev_set, batch_size=400, pin_memory=True, num_workers=8)
        self.metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)
                self.metrics(output['selection_triplets'], output['spo_gold'])

            result = self.metrics.get_metric()
            print(', '.join([
                "%s: %.4f" % (name, value)
                for name, value in result.items() if not name.startswith("_")
            ]) + " ||")

    def train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = Selection_loader(train_set, batch_size=100, pin_memory=True, num_workers=4)

        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            for batch_idx, sample in pbar:

                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))

            self.save_model(epoch)

            if epoch % self.hyper.print_epoch == 0 and epoch > 3:
                self.evaluation()


if __name__ == "__main__":
    config = Runner(exp_name=args.exp_name)
    config.run(mode=args.mode)
