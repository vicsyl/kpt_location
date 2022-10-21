import sys

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from config import *
from patch_dataset import PatchesDataModule, get_wand_name

from prepare_data import log_metada

from resnet_cnn import *


def train(config_path, set_config_dir_scheme=False, hash=None, checkpoint_path=None):

    conf = get_config(config_path)
    if set_config_dir_scheme:
        set_config_dir_scale_scheme(conf['dataset'])
    train_conf = conf['train']

    loggers = []
    wandb_logger = None
    enable_wandb = train_conf.get('enable_wandlog', False)
    if enable_wandb:
        wandb_project = train_conf["wandb_project"]
        wandb_name = get_wand_name(conf, wandb_run_name_keys=train_conf['wandb_run_name_keys'], hash=hash)
        wandb_tags = train_conf['tags']
        wandb.init(project=wandb_project,
                   name=wandb_name,
                   tags=wandb_tags)
        wandb_logger = WandbLogger(name=wandb_name, project=wandb_project)
        wandb_logger.experiment.config.update(OmegaConf.to_container(conf))
        loggers = [wandb_logger]
        # NOTE this doesn't show anywhere
        wandb.config = OmegaConf.to_container(train_conf)

    dm = PatchesDataModule(conf, wandb_logger)
    model = get_module(conf, checkpoint_path)

    if enable_wandb:
        wandb.watch(model)

    # TODO just redo this so that it's more logical (problem is 'log_metada' is in 'prepare_data.py'
    baseline_loss = log_metada(dict(dm.dataset.metadata_list), conf['dataset'], conf_to_log=conf)
    model.set_baseline_loss(baseline_loss)

    trainer = Trainer(max_epochs=conf['train']['max_epochs'],
                      accelerator=train_conf['accelerator'],
                      log_every_n_steps=50,
                      logger=loggers # try tensorboard as well
                      )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Train a deep model')
    parser.add_argument('--config', help='config path', required=False)
    parser.add_argument('--hash', help='git hash', required=False)
    parser.add_argument('--checkpoint_path', help='checkpoint path', required=False)
    args = parser.parse_args()

    config_path = 'config/config.yaml' if not args.config else args.config
    print(f"config_path={config_path}")

    train(config_path=config_path,
          set_config_dir_scheme=False,
          hash=args.hash,
          checkpoint_path=args.checkpoint_path)
