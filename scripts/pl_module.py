import torch
import pytorch_lightning as pl
from features import FeatureMapper
from model import Network
from torch.optim import Adadelta
from parser import Parser


class SpanParser(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        pl.seed_everything(42)
        self.save_hyperparameters()

        self.fm = FeatureMapper.load_json(hparams.data.vocab)
        self.word_count = self.fm.total_words()
        self.tag_count = self.fm.total_tags()

        self.network = Network(
            feature_mapper=self.fm,
            word_count=self.word_count,
            tag_count=self.tag_count,
            word_dims=hparams.model.word_dims,
            tag_dims=hparams.model.tag_dims,
            lstm_units=hparams.model.lstm_units,
            hidden_units=hparams.model.hidden_units,
            struct_out=2,
            label_out=self.fm.total_label_actions(),
            droprate=hparams.model.droprate,
        )

    def configure_optimizers(self):
        return Adadelta(self.network.parameters(), eps=1e-7, rho=0.99)

    def forward(self, batch):
        batch_error = self.network(batch)
        return batch_error

    def training_step(self, batch):
        batch_error = self(batch)
        self.log("train/loss", batch_error)
        return batch_error

    def validation_step(self, batch):
        dev_acc = Parser.evaluate_corpus(batch, self.fm, self.network)
        self.log("val/acc", dev_acc)
        return dev_acc
