import math

import wandb
from torch import randperm
from torch import Generator
from torch import default_generator

from typing import (
    Any,
    List,
    Optional,
    Sequence,
    TypeVar,
)
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from torch.utils.data import random_split, DataLoader, Subset
from torch.utils.data.dataset import Dataset
from config import *


def batched_random_split(dataset: Dataset[T], lengths: Sequence[int], batch_size, generator: Optional[Generator] = default_generator) -> List[Subset[T]]:

    if sum(lengths) * batch_size != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    perm_indices = randperm(sum(lengths), generator=generator).tolist()
    lsum = 0
    ret = []
    for length in lengths:
        indices = []
        for l in range(lsum, lsum + length):
            indices.extend(list(range(perm_indices[l] * batch_size, (perm_indices[l] + 1) * batch_size)))
        ret.append(Subset(dataset, indices))
        lsum += length

    return ret


class PatchDataset(Dataset):

    def __init__(self, root_dir, batch_size) -> None:
        super().__init__()
        self.root_dir = root_dir
        metadata = {}
        with open("{}/a_values.txt".format(root_dir), "r") as f:
            for line in f.readlines():
                if line.__contains__("#"):
                    continue
                tokens = line.strip().split(",")
                file = tokens[0]
                dx = float(tokens[1])
                dy = float(tokens[2])
                size = int(tokens[3])
                metadata[file] = (dx, dy, size)
        self.metadata_list = list(metadata.items())

        if batch_size is not None:
            group_bys = {}
            for mt in self.metadata_list:
                size = mt[1][2]
                if not group_bys.__contains__(size):
                    group_bys[size] = list()
                group_bys[size].append((mt[0], mt[1]))
            self.metadata_list = []
            for size in group_bys:
                l = len(group_bys[size])
                l_final = l - l % batch_size
                print("all batches for size {}: {} -> {}".format(size, l, l_final))
                self.metadata_list.extend(group_bys[size][:l_final])

    def __getitem__(self, index) -> Any:
        metadata = self.metadata_list[index]
        path = "{}/{}".format(self.root_dir, metadata[0])
        patch_pil = Image.open(path)
        patch_t = torchvision.transforms.functional.to_tensor(np.array(patch_pil))
        return patch_t, torch.tensor(metadata[1][:2])

    def __len__(self) -> int:
        return len(self.metadata_list)


# TODO add transforms (e.g. normalization)
class PatchesDataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        train_conf = conf['train']
        self.batch_size = train_conf['batch_size']
        self.grouped_by_sizes = train_conf['grouped_by_sizes']
        bs = self.batch_size if self.grouped_by_sizes else None
        root_dir = get_full_ds_dir(conf)
        self.dataset = PatchDataset(root_dir, batch_size=bs)

    def setup(self, stage: str):
        size = len(self.dataset)

        if self.grouped_by_sizes:
            assert size % self.batch_size == 0
            part_size = (size // self.batch_size) // 4
            parts = [part_size, part_size, part_size, size // self.batch_size - 3 * part_size]
            self.train, self.validate, self.test, self.predict = batched_random_split(self.dataset, parts, self.batch_size)
        else:
            part_size = size // 4
            parts = [part_size, part_size, part_size, size - 3 * part_size]
            self.train, self.validate, self.test, self.predict = random_split(self.dataset, parts)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)


def iterate_dataset():

    # download, etc...
    dm = PatchesDataModule("./dataset", batch_size=32, grouped_by_sizes=True)
    dm.prepare_data()

    # splits/transforms
    dm.setup(stage="fit")

    # use data
    for batch in dm.train_dataloader():
        print(batch.__class__)

    for batch in dm.val_dataloader():
        print(batch.__class__)

    dm.teardown(stage="fit")

    # lazy load test data
    dm.setup(stage="test")
    for batch in dm.test_dataloader():
        print(batch.__class__)

    dm.teardown(stage="test")


def log_stats(ds_path):

    wandb.init(project="kpt_location_error_analysis")

    # = [x, x], where x = sqrt(2) / 2 * scale  (for scale ~0.3)
    adjustment = [0.15, 0.15]

    metadata_list = PatchDataset(ds_path, batch_size=None).metadata_list

    distances = []
    distances_adjusted = []
    errors = []
    errors_adjusted = []
    angles = []
    angles_adjusted = []
    for _, value in metadata_list:
        dxy = np.array(value[:2])
        errors.append(dxy)

        dxy_adjusted = dxy - adjustment
        errors_adjusted.append(dxy_adjusted)

        distances.append([math.sqrt(dxy[0] ** 2 + dxy[1] ** 2)])
        distances_adjusted.append([math.sqrt(dxy_adjusted[0] ** 2 + dxy_adjusted[1] ** 2)])
        angles.append([np.arctan2(dxy[0], dxy[1])])
        angles_adjusted.append([np.arctan2(dxy_adjusted[0], dxy_adjusted[1])])

    print("err mean: {}".format(np.array(errors).mean(axis=0)))
    print("err abs mean: {}".format(np.abs(np.array(errors)).mean(axis=0)))
    print("adjusted err mean: {}".format(np.array(errors_adjusted).mean(axis=0)))
    print("adjusted err abs mean: {}".format(np.abs(np.array(errors_adjusted)).mean(axis=0)))
    print("angles mean: {}".format(np.array(angles).mean(axis=0)))
    print("angles abs mean: {}".format(np.abs(np.array(angles)).mean(axis=0)))
    print("adjusted angles mean: {}".format(np.array(angles_adjusted).mean(axis=0)))
    print("adjusted angles abs mean: {}".format(np.abs(np.array(angles_adjusted)).mean(axis=0)))

    t_d = wandb.Table(data=distances, columns=["distance"])
    wandb.log({'distances': wandb.plot.histogram(t_d, "distance", title="scale error")})

    t_d = wandb.Table(data=distances_adjusted, columns=["distance"])
    wandb.log({'distances adjusted': wandb.plot.histogram(t_d, "distance", title="scale error adjusted")})

    t_d = wandb.Table(data=angles, columns=["angle"])
    wandb.log({'angles': wandb.plot.histogram(t_d, "angle", title="angle error")})

    t_d = wandb.Table(data=angles_adjusted, columns=["angle"])
    wandb.log({'angles adjusted': wandb.plot.histogram(t_d, "angle", title="angle error adjusted")})


if __name__ == "__main__":
    #iterate_dataset()
    log_stats("dataset/const_size_33")
    print("patch dataset")
