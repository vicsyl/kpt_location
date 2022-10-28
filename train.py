from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from config import *
from patch_dataset import PatchesDataModule, get_wand_name
from prepare_data import log_config_and_datasets
from resnet_cnn import *


def train(config_path, args, set_config_dir_scheme=False, hash=None, checkpoint_path=None):

    conf = get_config(config_path)
    if set_config_dir_scheme:
        set_config_dir_scale_scheme(conf['dataset'])
    train_conf = conf['train']

    loggers = []
    wandb_logger = None
    enable_wandb = train_conf.get('enable_wandlog', False)
    if enable_wandb:
        wandb_name = get_wand_name(conf, wandb_run_name_keys=train_conf['wandb_run_name_keys'], hash=hash)
        if args.name:
            wandb_name = f"{args.name}_{wandb_name}"
        wandb_project = train_conf["wandb_project"]
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
    baseline_loss = log_config_and_datasets(dm.get_all_metadata_list_map(), conf['dataset'], conf_to_log=conf)
    devices = train_conf['devices'] if type(train_conf['devices']) == int else list(train_conf['devices'])
    if args.devices:
        devices = [int(d) for d in args.devices.split(" ")]

    print(f"baseline_loss: {baseline_loss}")
    print(f"devices: {devices}")

    model.set_baseline_loss(baseline_loss)

    trainer = Trainer(max_epochs=conf['train']['max_epochs'],
                      accelerator=train_conf['accelerator'],
                      devices=devices,
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
    parser.add_argument('--devices', help='gpus', required=False)
    parser.add_argument('--name', help='name', required=False)
    args = parser.parse_args()

    if args.name:
        print(f"args.name: {args.name}")
    else:
        print("no args.name!")
    if args.devices:
        devices = [int(d) for d in args.devices.split(" ")]
    else:
        print("no args.devices!")

    config_path = 'config/config.yaml' if not args.config else args.config
    print(f"config_path={config_path}")

    if True:
        raise Exception()

    train(config_path=config_path,
          args=args,
          set_config_dir_scheme=False,
          hash=args.hash,
          checkpoint_path=args.checkpoint_path)
