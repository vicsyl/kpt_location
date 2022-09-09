import torch.nn as nn
import torch
import torchvision.models as models
from pytorch_lightning import LightningModule, Trainer
from patch_dataset import PatchesDataModule


class PatchesModule(LightningModule):
    def __init__(self):
        super().__init__()

        resnet50 = models.resnet50(pretrained=True)
        in_features = resnet50.fc.in_features
        layers = list(resnet50.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, 2)
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)

        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        xs, ys = batch[0], batch[1]
        ys_hat = self(xs)
        loss = self.loss_function(ys_hat, ys)
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
        return dict(
            validation_loss=loss,
            log=dict(
                val_loss=loss
            )
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def main():

    model = PatchesModule()
    dm = PatchesDataModule("./dataset")
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)


if __name__ == "__main__":
    main()
