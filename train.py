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
        self.classifier = nn.Linear(in_features, 2)
        self.loss_function = nn.MSELoss()
        self.learning_rate = train_conf['learning_rate']
        self.log = ''

    def forward(self, x):
        # self.feature_extractor.eval()
        # with torch.no_grad():
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        ys_hat = self(xs)
        loss = self.loss_function(ys_hat, ys)
        wandb.log({"loss": loss})
        self.log += 'T'
        #print("fskutecnosti train loss: {}".format(loss))
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
        wandb.log({"loss": loss})
        self.log += 'V'
        #print("fskutecnosti val loss: {}".format(loss))
        return dict(
            validation_loss=loss,
            log=dict(
                val_loss=loss
            )
        )

    # TODO where are the parameters defined?
    # TODO why not classifier!!!
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train(path='config/config.yaml'):

    wandb.init(project="kpt_location")
    conf = get_config(path)
    train_conf = conf['train']
    wandb.config = train_conf

    # wandb.config = {
    #     "learning_rate": 0.001,
    #     "epochs": 100,
    #     "batch_size": 128
    # }

    # Optional
    model = PatchesModule(train_conf)
    # what does this actually do?
    wandb.watch(model)

    dm = PatchesDataModule(conf)
    trainer = Trainer(max_epochs=conf['train']['max_epochs'])
    trainer.fit(model, datamodule=dm)
    eval_ret = trainer.validate(model, datamodule=dm)
    print("t/v: {}".format(model.log))


if __name__ == "__main__":
    train()
