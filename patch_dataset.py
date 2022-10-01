import dataclasses
import math

import wandb
from torch import randperm
from torch import Generator
from torch import default_generator

from dataclasses import dataclass

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


# others: match_args=True, kw_only=False, slots=False
@dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=False, frozen=False)
class DataRecord:

    dy: float
    dx: float
    patch_size: int
    kpt_orig_scale: float
    kpt_resize_scale: float
    scale_ratio: float
    real_scale: float
    img_scale_y: float
    img_scale_x: float
    original_img_size_y: int
    original_img_size_x: int
    resized_img_size_y: int
    resized_img_size_x: int
    augmented: str  # original, rotated_$angles, reflected_$axis

    def is_augmented(self):
        return self.augmented != "original"

    def line_str(self):
        return ", ".join([str(i) for i in dataclasses.astuple(self)])

    @staticmethod
    def schema():
        # NOTE - __match_args__ cannot use it on Colab as it's only since python 3.10 (3.6 on Colab)
        return "dy, dx, patch_size, kpt_orig_scale, kpt_resize_scale, scale_ratio, real_scale, \
img_scale_y, img_scale_x, original_img_size_y, original_img_size_x, \
resized_img_size_y, resized_img_size_x, augmented"

    @staticmethod
    def read_from_tokens(tokens):
        tokens = [token.strip() for token in tokens]
        return DataRecord(
            dx=float(tokens[0]),
            dy=float(tokens[1]),
            patch_size=int(tokens[2]),
            kpt_orig_scale=float(tokens[3]),
            kpt_resize_scale=float(tokens[4]),
            scale_ratio=float(tokens[5]),
            real_scale=float(tokens[6]),
            img_scale_y=float(tokens[7]),
            img_scale_x=float(tokens[8]),
            original_img_size_y=int(tokens[9]),
            original_img_size_x=int(tokens[10]),
            resized_img_size_y=int(tokens[11]),
            resized_img_size_x=int(tokens[12]),
            augmented=tokens[13])

    @staticmethod
    def read_metadata_list_from_file(file_path):
        mt_d = {}
        with open(file_path, "r") as f:
            for line in f.readlines():
                if line.__contains__("#"):
                    continue
                tokens = line.strip().split(",")
                file = tokens[0].strip()
                dr = DataRecord.read_from_tokens(tokens[1:])
                mt_d[file] = dr
        return list(mt_d.items())


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

    def __init__(self, root_dir, train_config) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.metadata_list = DataRecord.read_metadata_list_from_file("{}/a_values.txt".format(root_dir))
        self.batch_size = train_config['batch_size']
        self.grouped_by_sizes = train_config['grouped_by_sizes']
        self.augment = (train_config['augment'].lower() == "lazy")

        # NOTE probably broken
        if self.grouped_by_sizes:
            group_bys = {}
            for mt in self.metadata_list:
                size = mt[1][2]
                if not group_bys.__contains__(size):
                    group_bys[size] = list()
                group_bys[size].append((mt[0], mt[1]))
            self.metadata_list = []
            for size in group_bys:
                l = len(group_bys[size])
                l_final = l - l % self.batch_size
                print("all batches for size {}: {} -> {}".format(size, l, l_final))
                self.metadata_list.extend(group_bys[size][:l_final])

    def __getitem__(self, index) -> Any:

        md_index = index
        if self.augment:
            md_index = md_index // 6
        metadata = self.metadata_list[md_index]
        dx, dy = metadata[1].dx, metadata[1].dy
        path = "{}/data/{}".format(self.root_dir, metadata[0])
        patch_pil = Image.open(path)
        patch_t = torchvision.transforms.functional.to_tensor(np.array(patch_pil))
        if self.augment and index % 6 != 0:
            augment_index = index % 6
            patches_aug, diffs_aug, _ = augment_patch(patch_t[0], (dy, dx))
            patch_t = patches_aug[augment_index][None]
            dy, dx = diffs_aug[augment_index]
        if patch_t.shape[0] == 1:
            patch_t = patch_t.expand(3, -1, -1)
        y = torch.tensor([dy, dx])
        return patch_t, y

    def __len__(self) -> int:
        return len(self.metadata_list)


def augment_patch(patch, diffs):
    patch_r_y = torch.flip(patch, dims=[0])
    diffs_r_y = -diffs[0], diffs[1]
    patch_r_x = torch.flip(patch, dims=[1])
    diffs_r_x = diffs[0], -diffs[1]

    patches = [patch]
    diff_l = [diffs]
    augment_keys = ["original"]
    for i in range(3):
        patch = torch.rot90(patch, 1, [0, 1])
        patches.append(patch)
        diffs = -diffs[1], diffs[0]
        diff_l.append(diffs)
        augment_keys.append("rotated_{}".format((i + 1) * 90))
    patches = patches + [patch_r_y, patch_r_x]
    diff_l = diff_l + [diffs_r_y, diffs_r_x]
    augment_keys = augment_keys + ["reflected_y", "reflected_x"]
    return patches, diff_l, augment_keys


# TODO add transforms (e.g. normalization)
class PatchesDataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()

        # NOTE to be changed
        # == 2: leave test and predict out for now
        # == 4: with test and predict ds

        train_conf = conf['train']
        self.splits = train_conf['dataset_splits']
        assert self.splits in [2, 4]
        self.batch_size = train_conf['batch_size']
        self.grouped_by_sizes = train_conf['grouped_by_sizes']
        root_dir = get_full_ds_dir(conf['dataset'])
        self.dataset = PatchDataset(root_dir, train_conf)

    def prepend_parts(self, parts, part_size):
        if self.splits == 2:
            parts = [0, 0] + parts
        elif self.splits == 4:
            parts = [part_size, part_size] + parts
        else:
            raise "unexpected value for self.splits: {}".format(self.splits)
        return parts

    def setup(self, stage: str):
        size = len(self.dataset)

        if self.grouped_by_sizes:
            assert size % self.batch_size == 0
            part_size = (size // self.batch_size) // self.splits
            parts = [part_size, size // self.batch_size - (self.splits - 1) * part_size]
            parts = self.prepend_parts(parts, part_size)
            self.test, self.predict, self.train, self.validate = batched_random_split(self.dataset, parts, self.batch_size)
        else:
            part_size = size // self.splits
            parts = [part_size, size - (self.splits - 1) * part_size]
            parts = self.prepend_parts(parts, part_size)
            self.test, self.predict, self.train, self.validate = random_split(self.dataset, parts)

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
    config = {
        "batch_size": 32,
        "grouped_by_sizes": True,
    }
    # FIXME won't work anyway
    dm = PatchesDataModule("./dataset", config)
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


def get_error_stats(entry_list, adjustment=[0.0, 0.0]):

    distances = []
    errors = []
    angles = []
    for _, data_record in entry_list:
        if data_record.is_augmented():
            continue
        dxy = np.array([data_record.dy, data_record.dx])
        dxy_adjusted = dxy + adjustment
        errors.append(dxy_adjusted)
        distances.append([math.sqrt(dxy_adjusted[0] ** 2 + dxy_adjusted[1] ** 2)])
        angles.append([np.arctan2(dxy_adjusted[0], dxy_adjusted[1])])

    distances = np.array(distances)
    errors = np.array(errors)
    angles = np.array(angles)

    return distances, errors, angles


def mean_abs_mean(stat):
    mean = stat.mean(axis=0)
    abs_mean = np.abs(stat).mean(axis=0)
    return mean, abs_mean


def log_stats(ds_path, wand_project):

    def print_stat(stat, name):
        mean, abs_mean = mean_abs_mean(stat)
        print("{} mean: {}".format(name, mean))
        print("{} abs mean: {}".format(name, abs_mean))

    # FIXME
    metadata_list = PatchDataset(ds_path, batch_size=None).metadata_list

    distances, errors, angles = get_error_stats(metadata_list, [0.0, 0.0])
    print_stat(distances, "distance")
    print_stat(errors, "error")
    print_stat(angles, "angle")

    adjustment = [0.15, 0.15]
    distances_adjusted, errors_adjusted, angles_adjusted = get_error_stats(metadata_list, adjustment)
    print_stat(distances_adjusted, "adjusted distance")
    print_stat(errors_adjusted, "adjusted error")
    print_stat(angles_adjusted, "adjusted angle")

    group = False
    if group:

        def analyse_unique(stat, name):
            groups = np.unique(stat, return_counts=True, axis=0)
            indices = np.flip(np.argsort(groups[1]))
            values = groups[0][indices]
            counts = groups[1][indices]
            print("{} values:".format(name))
            for value in values:
                print(value)
            print("{} counts:".format(name))
            for count in counts:
                print(count)

        analyse_unique(errors, "errors")
        analyse_unique(angles, "angles")

    if wand_project:

        wandb.init(project=wand_project)

        t_d = wandb.Table(data=distances, columns=["distance"])
        wandb.log({'distances': wandb.plot.histogram(t_d, "distance", title="distance of error")})

        t_d = wandb.Table(data=distances_adjusted, columns=["distance"])
        wandb.log({'distances adjusted': wandb.plot.histogram(t_d, "distance", title="distance of error adjusted")})

        t_d = wandb.Table(data=angles, columns=["angle"])
        wandb.log({'angles': wandb.plot.histogram(t_d, "angle", title="angle error")})

        t_d = wandb.Table(data=angles_adjusted, columns=["angle"])
        wandb.log({'angles adjusted': wandb.plot.histogram(t_d, "angle", title="angle error adjusted")})


def t_data_record():

    dr = DataRecord(
        dy=1.0,
        dx=2.0,
        patch_size=33,
        kpt_orig_scale=100.0,
        kpt_resize_scale=10.0,
        scale_ratio=1.1,
        real_scale=0.3,
        img_scale_y=0.3,
        img_scale_x=0.354548454,
        original_img_size_y=1080,
        original_img_size_x=1960,
        resized_img_size_y=108,
        resized_img_size_x=196,
        augmented="original"
    )
    tp1 = dr.schema()
    print("schema 1: {}".format(tp1))
    tp2 = DataRecord.schema()
    print("schema 2: {}".format(tp2))
    tp3 = dataclasses.astuple(dr)
    print("astuple: {}".format(tp3))
    print("img_scale_x {}".format(dr.img_scale_x))
    print("ls: '{}'".format(dr.line_str()))
    print("schema: {}".format(DataRecord.schema()))


if __name__ == "__main__":
    #iterate_dataset()
    #wand_project = "kpt_location_error_analysis_private"
    log_stats("dataset/const_size_adj_33", wand_project=None)
    #t_data_record()
