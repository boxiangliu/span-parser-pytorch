from utils import load_yaml
from data import SpanParserDataModule
from pl_module import SpanParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


hparams = load_yaml("scripts/model.yaml")
span_parser_data_module = SpanParserDataModule(hparams)
network = SpanParser(hparams)
wandb_logger = WandbLogger(
    project=hparams.logger.project,
    name=hparams.logger.name,
    notes=hparams.logger.notes,
    save_dir=hparams.logger.save_dir,
)
trainer = pl.Trainer(
    max_epochs=hparams.trainer.epochs,
    accelerator="gpu",
    logger=wandb_logger,
    gradient_clip_val=hparams.trainer.grad_clip,
    gpus=hparams.trainer.gpus,
    auto_select_gpus=True,
    log_every_n_steps=hparams.trainer.log_interval,
    val_check_interval=hparams.trainer.val_interval,
)
trainer.fit(network, span_parser_data_module)
