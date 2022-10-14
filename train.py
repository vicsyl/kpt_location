from pytorch_lightning import Trainer

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

    module_name = train_conf['module'].lower()
    if module_name == "resnet_based":
        model = ResnetBasedModule(train_conf)
    elif module_name == "zero_inference":
        model = ZeroModule(train_conf, dm)
    else:
        raise ValueError

    enable_wandb = train_conf.get('enable_wandlog', False)
    if enable_wandb:
        wandb.init(project=wandb_project,
                   name=get_wand_name(conf, wandb_run_name_keys=train_conf['wandb_run_name_keys']),
                   tags=train_conf['tags'])
        # NOTE this doesn't show anywhere
        wandb.config = train_conf
        wandb.watch(model)

    log_metada(dict(dm.dataset.metadata_list), conf['dataset'], enable_wandb, file=None, conf_to_log=conf)

    trainer = Trainer(max_epochs=conf['train']['max_epochs'],
                      accelerator=train_conf['accelerator'],
                      )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train(set_config_dir_scheme=False, wandb_project="kpt_location_training_dev")
