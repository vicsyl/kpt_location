import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import LightningModule, Trainer

import wandb
from patch_dataset import PatchesDataModule
from config import *


class PatchesModule(LightningModule):
    def __init__(self, train_conf):
        super().__init__()

        resnet50 = models.resnet50(pretrained=True)
        in_features = resnet50.fc.in_features
        layers = list(resnet50.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.freeze_feature_extractor = train_conf['freeze_feature_extractor']
        self.classifier = nn.Linear(in_features, 2)
        self.loss_function = nn.MSELoss()
        self.learning_rate = train_conf['learning_rate']
        self.enable_wandlog = train_conf.get('enable_wandlog', False)

    def wandlog(self, obj):
        if self.enable_wandlog:
            wandb.log(obj)

    def forward(self, x):
        if self.freeze_feature_extractor:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor(x)
        else:
            features = self.feature_extractor(x)
        representations = features.flatten(1)
        x = self.classifier(representations)
        return x

    def log_stats(self, ys, ys_hat, prefix):
        if self.enable_wandlog:
            batch_size = ys.shape[0]
            distance_loss = ((ys ** 2).sum(dim=1) ** 0.5 - (ys_hat ** 2).sum(dim=1) ** 0.5).abs().sum() / batch_size
            wandb.log({"{}_distance_loss".format(prefix): distance_loss})
            angle_loss = (torch.atan2(ys[:, 0], ys[:, 1]) - torch.atan2(ys_hat[:, 0], ys_hat[:, 1])).abs() / batch_size
            wandb.log({"{}_angle_loss".format(prefix): angle_loss})

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        ys_hat = self(xs)
        loss = self.loss_function(ys_hat, ys)
        self.wandlog({"training_loss": loss})
        self.log_stats(ys, ys_hat, "training")
        return dict(
            loss=loss,
            log=dict(
                train_loss=loss
            )
        )

    def validation_step(self, batch, batch_idx):
        xs, ys = batch[0], batch[1]
        ys_hat = self(xs)
        loss = self.loss_function(ys_hat, ys)
        self.log_stats(ys, ys_hat, "validation")
        self.wandlog({"validation_loss": loss})
        return dict(
            validation_loss=loss,
            log=dict(
                val_loss=loss
            )
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train(path='config/config.yaml', wandb_project="kpt_location_training_private"):

    conf = get_config(path)
    train_conf = conf['train']

    model = PatchesModule(train_conf)
    if train_conf.get('enable_wandlog', False):
        wandb.init(project=wandb_project)
        # NOTE this doesn't show anywhere
        wandb.config = train_conf
        wandb.watch(model)

    dm = PatchesDataModule(conf)

    trainer = Trainer(max_epochs=conf['train']['max_epochs'],
                      accelerator=train_conf['accelerator'],
                      )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()
