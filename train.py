from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from config import *
from patch_dataset import PatchesDataModule, get_wand_name

from prepare_data import log_metada

from resnet_cnn import *


def train(path='config/config.yaml', wandb_project="kpt_location_training_private", set_config_dir_scheme=False):

    conf = get_config(path)
    if set_config_dir_scheme:
        set_config_dir_scale_scheme(conf['dataset'])
    train_conf = conf['train']

    dm = PatchesDataModule(conf)

    model = get_model(conf)

    loggers = []
    enable_wandb = train_conf.get('enable_wandlog', False)
    if enable_wandb:
        wandb_name = get_wand_name(conf, wandb_run_name_keys=train_conf['wandb_run_name_keys'])
        wandb_tags = train_conf['tags']
        wandb.init(project=wandb_project,
                   name=wandb_name,
                   tags=wandb_tags)
        wandb_logger = WandbLogger(name=wandb_name, project=wandb_project)
        wandb_logger.experiment.config.update(OmegaConf.to_container(conf))
        loggers = [wandb_logger]
        # NOTE this doesn't show anywhere
        wandb.config = OmegaConf.to_container(train_conf)
        wandb.watch(model)

    # TODO just redo this so that it's more logical (problem is 'log_metada' is in 'prepare_data.py'
    baseline_loss = log_metada(dict(dm.dataset.metadata_list), conf['dataset'], enable_wandb, file=None, conf_to_log=conf) / 2
    model.set_baseline_loss(baseline_loss)

    trainer = Trainer(max_epochs=conf['train']['max_epochs'],
                      accelerator=train_conf['accelerator'],
                      log_every_n_steps=50,
                      logger=loggers # try tensorboard as well
                      )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train(set_config_dir_scheme=False, wandb_project="kpt_location_training_dev")
