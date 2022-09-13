# No 'default_generator' in torch/__init__.pyi
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

        err = 0.0
        for mt in self.metadata_list:
            err_2d = torch.tensor(mt[1][:2])
            err += (err_2d @ err_2d.T).item()
        err = err / len(self.metadata_list)
        print("'SIFT'/default error: {}".format(err))

    def __getitem__(self, index) -> Any:
        metadata = self.metadata_list[index]
        path = "{}/{}".format(self.root_dir, metadata[0])
        patch_pil = Image.open(path)
        patch_t = torchvision.transforms.functional.to_tensor(np.array(patch_pil))
        return patch_t, torch.tensor(metadata[1][:2])

    def __len__(self) -> int:
        return len(self.metadata_list)


class PatchesDataModule(pl.LightningDataModule):

    def __init__(self, root_dir, batch_size, batch_aware):
        super().__init__()
        self.batch_size = batch_size
        # TODO add transforms (e.g. normalization)
        self.dataset = PatchDataset(root_dir, batch_size)
        self.batch_aware = batch_aware

    def setup(self, stage: str):
        size = len(self.dataset)

        # TODO batch aware
        if self.batch_aware:
            assert size % self.batch_size == 0
        part_size = (size // self.batch_size) // 4
        parts = [part_size, part_size, part_size, size // self.batch_size - 3 * part_size]
        self.train, self.validate, self.test, self.predict = batched_random_split(self.dataset, parts, self.batch_size)

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
    dm = PatchesDataModule("./dataset", batch_size=32, batch_aware=True)
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


if __name__ == "__main__":
    iterate_dataset()
