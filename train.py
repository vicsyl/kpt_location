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

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        ys_hat = self(xs)
        loss = self.loss_function(ys_hat, ys)
        self.wandlog({"loss": loss, "training_loss": loss})
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
        self.wandlog({"loss": loss, "validation_loss": loss})
        return dict(
            validation_loss=loss,
            log=dict(
                val_loss=loss
            )
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train(path='config/config.yaml'):

    conf = get_config(path)
    train_conf = conf['train']

    # TODO gpu and externalize

    model = PatchesModule(train_conf)
    if train_conf.get('enable_wandlog', False):
        wandb.init(project="kpt_location")
        # NOTE this doesn't show anywhere
        wandb.config = train_conf
        wandb.watch(model)

    dm = PatchesDataModule(conf)
    trainer = Trainer(max_epochs=conf['train']['max_epochs'])
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()
