import pytorch_lightning as pl
from features import FeatureMapper
from phrase_tree import PhraseTree
from utils import load_yaml
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class SpanParserData(Dataset):
    def __init__(self, data, train=True):
        self.data = data
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.train:
            return {
                "tree": self.data[index]["tree"],
                "w": torch.LongTensor(self.data[index]["w"]),
                "t": torch.LongTensor(self.data[index]["t"]),
                "struct_data": self.data[index]["struct_data"],
                "label_data": self.data[index]["label_data"],
            }
        else:
            # dev_tree is a list of PhraseTree object
            return self.data[index]


class SpanParserDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()

        self.fm = FeatureMapper.load_json(hparams.data.vocab)
        self.train_data_file = hparams.data.train
        self.dev_data_file = hparams.data.dev
        self.batch_size = hparams.trainer.batch_size

    def setup(self, stage=None):
        self._training_data = self.fm.gold_data_from_file(self.train_data_file)
        self.training_data = SpanParserData(self._training_data)

        self._dev_trees = PhraseTree.load_treefile(self.dev_data_file)
        self.dev_trees = SpanParserData(self._dev_trees, train=False)

    def train_dataloader(self):
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def dev_dataloader(self):
        return DataLoader(
            self.dev_trees,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
        )
