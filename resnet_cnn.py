import torch
import torch.nn as nn
import wandb

import torchvision.models as models
from pytorch_lightning import LightningModule


# TODO rename the file
def get_model(conf):

    train_conf = conf['train']
    module_name = train_conf['module'].lower()
    if module_name == "resnet_based":
        model = ResnetBasedModule(train_conf)
    elif module_name == "zero_inference":
        model = ZeroModule(train_conf)
    elif module_name == "mlp":
        model = MlpModule(conf)
    else:
        raise ValueError(f"unknown '{module_name}' module name")
    return model


class BasicModule(LightningModule):

    def __init__(self, train_conf):
        super().__init__()

        self.tr_conf = train_conf
        #self.feature_extractor = nn.Sequential(*layers)
        #self.classifier = nn.Linear(in_features, 2)
        self.freeze_feature_extractor = train_conf['freeze_feature_extractor']
        self.loss_function = nn.MSELoss()
        self.learning_rate = train_conf['learning_rate']
        self.enable_wandlog = train_conf.get('enable_wandlog', False)
        self.log_every_n_entries = train_conf['log_every_n_entries']
        self.scale_error = train_conf['scale_error']
        assert self.log_every_n_entries is not None
        self.cumulative_losses_lists = {}
        self.baseline_loss = None

    def set_baseline_loss(self, loss):
        self.baseline_loss = loss

    def forward(self, x):
        # TODO flatten and put to the child class
        if self.freeze_feature_extractor:
            # QUESTION - really?
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
        self.add_loss_log("training_loss", loss)
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
        self.add_loss_log("validation_loss", loss)
        return dict(
            validation_loss=loss,
            log=dict(
                val_loss=loss
            )
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def add_loss_log(self, key, loss):
        if not self.cumulative_losses_lists.__contains__(key):
            self.cumulative_losses_lists[key] = []
        l = self.cumulative_losses_lists[key]
        l.append(loss / (self.scale_error**2 * self.baseline_loss))
        if len(l) * self.tr_conf['batch_size'] >= self.log_every_n_entries:
            t = torch.tensor(l)
            self.wandlog({key: t.sum() / t.shape[0]})
            self.cumulative_losses_lists[key] = []

    def wandlog(self, obj):
        if self.enable_wandlog:
            wandb.log(obj)

    def log_stats(self, ys, ys_hat, prefix):
        disabled = True
        if disabled:
            return
        if self.enable_wandlog:
            batch_size = ys.shape[0]
            distance_loss = ((ys ** 2).sum(dim=1) ** 0.5 - (ys_hat ** 2).sum(dim=1) ** 0.5).abs().sum() / batch_size
            wandb.log({"{}_distance_loss".format(prefix): distance_loss})
            angle_loss = (torch.atan2(ys[:, 0], ys[:, 1]) - torch.atan2(ys_hat[:, 0], ys_hat[:, 1])).abs() / batch_size
            wandb.log({"{}_angle_loss".format(prefix): angle_loss})


class ResnetBasedModule(BasicModule):

    def __init__(self, train_conf):
        super().__init__(train_conf)
        resnet50 = models.resnet50(pretrained=True)
        in_features = resnet50.fc.in_features
        layers = list(resnet50.children())[:-1]
        self.tr_conf = train_conf
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, 2)


class ZeroModule(BasicModule):

    def __init__(self, train_conf):
        super().__init__(train_conf)
        # optimizer has to have at least some parameters
        # TODO how about just compute the loss in prediction time
        self.foo_classifier = nn.Linear(2, 2)


class MlpModule(BasicModule):

    def __init__(self, conf):
        super().__init__(conf['train'])

        train_crop = conf['dataset']['filtering']['train_crop']
        assert not conf['train']['freeze_feature_extractor']
        # assuming patch and heatmap
        # TODO if for different choices (heatmap, both,,,,)
        channels = 3
        hm_factor = 2
        in_dim = channels * hm_factor * train_crop ** 2
        inter_dim = 100

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, inter_dim),
            nn.BatchNorm1d(inter_dim, affine=False),
            nn.ReLU(),
            nn.Linear(inter_dim, inter_dim),
            nn.BatchNorm1d(inter_dim, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(inter_dim, inter_dim),
            nn.BatchNorm1d(inter_dim, affine=False),
            nn.ReLU()
        )
        self.classifier = nn.Linear(inter_dim, 2)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return super().forward(x)