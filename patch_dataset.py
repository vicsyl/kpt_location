from typing import Any
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import random_split, DataLoader
from torch.utils.data.dataset import Dataset


class PatchDataset(Dataset):

    def __init__(self, root_dir) -> None:
        super().__init__()
        self.root_dir = root_dir
        metadata = {}
        with open("{}/values.txt".format(root_dir), "r") as f:
            for line in f.readlines():
                tokens = line.strip().split(",")
                file = tokens[0]
                dx = float(tokens[1])
                dy = float(tokens[2])
                metadata[file] = (dx, dy)
        self.metadata_list = list(metadata.items())

    def __getitem__(self, index) -> Any:
        metadata = self.metadata_list[index]
        path = "{}/{}".format(self.root_dir, metadata[0])
        patch_pil = Image.open(path)
        patch_t = torch.from_numpy(np.array(patch_pil))
        return patch_t, metadata[1]

    def __len__(self) -> int:
        return len(self.metadata_list)


class PatchesDataModule(pl.LightningDataModule):

    def __init__(self, root_dir):
        super().__init__()
        # TODO add transforms
        self.dataset = PatchDataset(root_dir)

    def setup(self, stage: str):

        size = len(self.dataset)
        part_size = size // 4
        parts = [part_size, part_size, part_size, size - 3 * part_size]
        self.train, self.validate, self.test, self.predict = random_split(self.dataset, parts)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=32)


def iterate_dataset():

    # download, etc...
    dm = PatchesDataModule("./dataset")
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
