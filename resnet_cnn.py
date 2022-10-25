import omegaconf
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import LightningModule

import wandb


# TODO rename the file
def get_module(conf, checkpoint_path=None):
    module_key = conf['train']['module']
    return get_module_by_key(module_key, conf, checkpoint_path)


def get_module_by_key(module_key, conf, checkpoint_path=None):

    if type(module_key) == str:
        module_key = module_key.lower()
        if module_key == "resnet_based":
            if checkpoint_path is None:
                return ResnetBasedModule(conf)
            else:
                module = ResnetBasedModule.load_from_checkpoint(checkpoint_path)
                module.init(conf)
                return module
        elif module_key == "zero_inference":
            return ZeroModule(conf)
        elif module_key == "mlp":
            return MlpModule(conf)
        else:
            raise ValueError(f"unknown '{module_key}' module name")
    elif type(module_key) == omegaconf.dictconfig.DictConfig:
        assert len(module_key.keys()) == 1
        name = list(module_key.keys())[0].lower()
        if name == "two_heads":
            models = [get_module_by_key(n, conf) for n in module_key[name]]
            return TwoHeadsModule(models, conf)
        else:
            raise ValueError(f"Unknown key for module: {name}")
    else:
        raise ValueError(f"Unknown type: {type(module_key)}")


def get_loss_function(train_conf):
    loss_f_name = train_conf["loss"].upper()
    if loss_f_name == "L1":
        return nn.L1Loss()
    elif loss_f_name == "L2":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function'{loss_f_name}'")


class BasicModule(LightningModule):

    def __init__(self, conf):
        super().__init__()
        if conf is None:
            print("WARNING: train_conf is None, probably loading a checkpoint?")
        else:
            self.init(conf)

    def init(self, conf):
        assert conf is not None
        self.conf = conf
        train_conf = conf['train']
        self.tr_conf = train_conf
        self.freeze_feature_extractor = train_conf['freeze_feature_extractor']
        self.loss_function = get_loss_function(train_conf)
        self.learning_rate = train_conf['learning_rate']
        self.enable_wandlog = train_conf.get('enable_wandlog', False)
        self.log_every_n_entries = train_conf['log_every_n_entries']
        self.scale_error = train_conf['scale_error']
        assert self.log_every_n_entries is not None
        self.cumulative_losses_lists = {}
        self.baseline_loss = None

    def set_baseline_loss(self, loss):
        self.baseline_loss = loss

    def compute_representations(self, x):
        # TODO flatten and put to the child class
        if self.freeze_feature_extractor:
            # QUESTION - really?
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor(x)
        else:
            features = self.feature_extractor(x)
        representations = features.flatten(1)
        return representations

    def forward(self, x, clip=False):
        representations = self.compute_representations(x)
        x = self.classifier(representations)
        # TODO
        # if clip:
        #     self.tr_conf["filtering"]
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
        # TODO
        # validation_clip = self.tr_conf['validation_clip']
        # if validation_clip:
        #     norms = torch.linalg.norm(ys_hat, dim=1)
        #     factors = torch.clip(norms, 0, validation_clip * self.scale_error) / norms
        #     ys_hat_cl = ys_hat * factors
        #     loss_clipped = self.loss_function(ys_hat_cl, ys)
        #     #norm_factor = self.baseline_loss if self.baseline_loss else 1.0
        #     #normalized_loss = loss.detach().cpu() / (self.scale_error ** 2 * norm_factor)
        #     self.add_loss_log("clipped_validation_loss", loss_clipped)
        return dict(
            validation_loss=loss,
            log=dict(
                val_loss=loss
            )
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4, eps=1e-2)

    def add_loss_log(self, key, loss):
        if not self.cumulative_losses_lists.__contains__(key):
            self.cumulative_losses_lists[key] = []
        l = self.cumulative_losses_lists[key]
        norm_factor = self.baseline_loss if self.baseline_loss else 1.0
        normalized_loss = loss.detach().cpu() / (self.scale_error**2 * norm_factor)
        l.append(normalized_loss)
        if len(l) * self.tr_conf['batch_size'] >= self.log_every_n_entries:
            t = torch.tensor(l)
            self.wandlog({key: t.sum() / t.shape[0]})
            self.cumulative_losses_lists[key] = []

    def wandlog(self, obj):
        # https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#manual-logging
        if self.enable_wandlog:
            self.log_dict(obj, rank_zero_only=True)

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


class TwoHeadsModule(BasicModule):

    def __init__(self, heads: BasicModule, conf):
        super().__init__(conf)
        assert len(heads) == 2
        self.heads = heads
        inputs = sum([h.classifier.in_features for h in self.heads])
        self.classifier = nn.Linear(inputs, 2)
        # TODO this is needed to register the parameters
        self.feature_extractor0 = heads[0].feature_extractor
        self.feature_extractor1 = heads[1].feature_extractor

    def forward(self, x):

        # TODO OK this still doesn't fix the problem with different input sizes
        width = x.shape[3]
        height = x.shape[2]
        assert width == 2 * height
        x_image = x[:, :, :, height:]
        repr_image = self.heads[0].compute_representations(x_image)
        x_heat_map = x[:, :, :, :height]
        repr_heat_map = self.heads[1].compute_representations(x_heat_map)
        all_repres = torch.hstack((repr_image, repr_heat_map))
        x = self.classifier(all_repres)
        return x


class ResnetBasedModule(BasicModule):

    def __init__(self, conf=None):
        super().__init__(conf)

        # in base?
        self.save_hyperparameters()
        # checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        # print(checkpoint["hyper_parameters"])
        resnet50 = models.resnet50(pretrained=True)
        in_features = resnet50.fc.in_features
        layers = list(resnet50.children())[:-1]
        # probably unnecessary
        # self.tr_conf = conf["train"]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, 2)


# TODO apparently this is broken
class ZeroModule(BasicModule):

    def __init__(self, conf):
        super().__init__(conf)
        # optimizer has to have at least some parameters
        # TODO how about just compute the loss in prediction time
        self.foo_classifier = nn.Linear(2, 2)

    def forward(self, x):
        return torch.zeros((x.shape[0], 2), device=self.device)


class MlpModule(BasicModule):

    def __init__(self, conf):
        super().__init__(conf)

        train_crop = conf['dataset']['filtering']['train_crop']
        if not train_crop:
            train_crop = 33
        assert not conf['train']['freeze_feature_extractor']
        # assuming patch and heatmap
        # TODO implement for different choices (heatmap, both,,,,)
        channels = 3
        hm_factor = 2
        in_dim = channels * hm_factor * train_crop ** 2
        inter_dim = 100

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, inter_dim),
            nn.BatchNorm1d(inter_dim, affine=False),
            nn.ReLU(), # +residual
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