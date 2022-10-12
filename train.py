from pytorch_lightning import Trainer

from config import *
from patch_dataset import PatchesDataModule, get_wand_name

from resnet_cnn import *


def train(path='config/config.yaml', wandb_project="kpt_location_training_private", set_config_dir_scheme=False, zero_module=False):

    conf = get_config(path)
    if set_config_dir_scheme:
        set_config_dir_scale_scheme(conf['dataset'])
    train_conf = conf['train']

    dm = PatchesDataModule(conf)
    model = ResnetBasedModule(train_conf) if not zero_module else ZeroModule(train_conf, dm)

    if train_conf.get('enable_wandlog', False):
        wandb.init(project=wandb_project, name=get_wand_name(conf['dataset'], entry_list=None))
        # NOTE this doesn't show anywhere
        wandb.config = train_conf
        wandb.watch(model)

    trainer = Trainer(max_epochs=conf['train']['max_epochs'],
                      accelerator=train_conf['accelerator'],
                      )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train(set_config_dir_scheme=False, zero_module=True)
