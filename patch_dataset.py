import argparse
import dataclasses
import math
import os
from dataclasses import dataclass
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    TypeVar,
)

import matplotlib.pyplot as plt
import torch
import torchvision.transforms
import wandb
from torch import Generator
from torch import default_generator
from torch import randperm

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

import pytorch_lightning as pl
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import Dataset
from config import *
from wand_utils import log_table


def get_wand_name(config, entry_list=None, extra_key=None, wandb_run_name_keys=None, hash=None):

    if not wandb_run_name_keys:
        wandb_run_name_keys = config['wandb_run_name_keys']
    name = "" if not hash else hash + ":"
    if extra_key:
        name += extra_key
    for wandb_tags_key in wandb_run_name_keys:
        if wandb_tags_key == "magic_items":
            name = name + ":i=" + str(len(entry_list))
        elif wandb_tags_key == "magic_out_dir":
            dir = get_full_ds_dir(config['dataset'])
            name += f"d={dir.split('/')[-1]}"
        elif wandb_tags_key.startswith("no_key"):
            wandb_tags_key = wandb_tags_key[7:]
            value = config.get(wandb_tags_key, None)
            # TODO check for the type -> str of dict
            if value:
                name = name + ":" + str(value)
        else:
            keys = wandb_tags_key.split(".")
            map_value = config
            for key in keys:
                if map_value:
                    map_value = map_value.get(key, None)
            if map_value is not None and map_value is not False:
                name = name + ":{}={}".format(keys[-1][0], str(map_value))

    return name


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
    def read_metadata_list_from_file(file_path, subdir):
        mt_d = {}
        with open(file_path, "r") as f:
            for line in f.readlines():
                if line.__contains__("#"):
                    continue
                tokens = line.strip().split(",")
                file_path = f"{subdir}/data/{tokens[0].strip()}"
                dr = DataRecord.read_from_tokens(tokens[1:])
                mt_d[file_path] = dr
        return list(mt_d.items())


def naive_split(dataset, parts):
    sum = 0
    ret = []
    for part in parts:
        ret.append(Subset(dataset, range(sum, sum + part)))
    return ret


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

    @staticmethod
    def filter_metadata_list(md_list, filtering_conf):

        sort_error = None if not filtering_conf["sort_error"] else filtering_conf["sort_error"].lower()
        assert sort_error in ["min", "max", None]
        max_error_distance = filtering_conf['max_error_distance']

        mask = np.ones(len(md_list), dtype=bool)
        if sort_error or max_error_distance:
            err_dists = np.zeros(len(md_list))
            for i, e in enumerate(md_list):
                _, dr = e
                err_dists[i] = math.sqrt(dr.dx ** 2 + dr.dy ** 2)
            if max_error_distance:
                mask = err_dists <= max_error_distance
            indices = np.arange(len(md_list))
            if sort_error == "min":
                indices = np.argsort(err_dists)
            elif sort_error == "max":
                indices = np.argsort(err_dists)
                indices = np.flip(indices)
            mask = mask[indices]

            new_l = []
            for i in range(len(md_list)):
                if mask[i]:
                    new_l.append(md_list[indices[i]])
            md_list = new_l
        return md_list

    default_train_crop = 33

    @staticmethod
    def get_split_datasets(root_dir, conf, wand_logger):

        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(f"{root_dir}/{d}")])
        f_conf = conf['dataset']['filtering']
        train_entries = f_conf['train_entries']
        train_scenes = f_conf['train_scenes']
        val_entries = f_conf['val_entries']
        val_scenes = f_conf['val_scenes']
        test_entries = f_conf['test_entries']
        test_scenes = f_conf['test_scenes']

        fr = 0
        train_dataset = PatchDataset(root_dir, conf, wand_logger, subdirs[fr: fr + train_scenes], train_entries)
        fr += train_scenes
        val_dataset = PatchDataset(root_dir, conf, wand_logger, subdirs[fr: fr + val_scenes], val_entries)
        fr += val_scenes
        test_dataset = PatchDataset(root_dir, conf, wand_logger, subdirs[fr: fr + test_scenes], test_entries)
        return train_dataset, val_dataset, test_dataset

    def __init__(self, root_dir, conf, wandb_logger=None, dirs=None, items=None) -> None:
        super().__init__()
        train_config = conf['train']
        self.root_dir = root_dir

        if dirs is None:
            dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(f"{root_dir}/{d}")])
            items = conf['dataset']['filtering']['entries']
        self.metadata_list, self.metadata_map = self.pick_items(conf, root_dir, dirs, items)
        self.batch_size = train_config['batch_size']
        self.grouped_by_sizes = train_config['grouped_by_sizes']
        self.train_crop = conf['dataset']['filtering']['train_crop']
        if not self.train_crop:
            self.train_crop = PatchDataset.default_train_crop
        self.scale_error = train_config['scale_error']
        self.augment_on_the_fly = conf['dataset']['augment'] and (conf['dataset']['augment'].lower() == "lazy")
        assert not self.augment_on_the_fly, "not supported"
        if is_hm_relevant(conf):
            self.heatmap_or_img = conf['dataset']['filtering']['heatmap_or_img']
        else:
            self.heatmap_or_img = None
        self.handle_grouping()
        self.conf = conf
        self.input_counter = 1
        #self.log_wand_imgs = self.conf['dataset']["enable_wandlog"] and self.conf['dataset']["wandb_log_imgs"]
        self.wandb_logger = wandb_logger
        self.patches_to_log = [None]
        self.resample_method = self.get_upscale_method()

    def pick_items(self, conf, root_dir, dirs, items):
        all_items_l = []
        all_items_map = {}
        for i, subdir in enumerate(dirs):
            metadata_list = DataRecord.read_metadata_list_from_file(f"{root_dir}/{subdir}/a_values.txt", subdir)
            metadata_list = PatchDataset.filter_metadata_list(metadata_list, conf['dataset']['filtering'])
            integer_part = items // len(dirs)
            remainder = items % len(dirs)
            items_to_get = integer_part if i + 1 > remainder else integer_part + 1
            if len(metadata_list) < items_to_get:
                raise Exception(f"data too small, trying to get {items_to_get} from data that is {len(metadata_list)} long.")
            metadata_list = metadata_list[:items_to_get]
            all_items_l.extend(metadata_list)
            all_items_map[subdir] = metadata_list
        return all_items_l, all_items_map

    def __getitem__(self, index) -> Any:

        merge_early = True

        md_index = index
        if self.augment_on_the_fly:
            md_index = md_index // 6
        metadata = self.metadata_list[md_index]
        dx, dy = metadata[1].dx, metadata[1].dy
        path = "{}/{}".format(self.root_dir, metadata[0])
        patch_pil = Image.open(path)
        patch_t = torchvision.transforms.functional.to_tensor(np.array(patch_pil))
        #self.show_patch(patch_t, "on input")

        def merge(data):
                assert data.shape[1] * 2 == data.shape[2]
                l_patch = torch.empty((3, data.shape[1], data.shape[1]))
                l_patch[0] = data[0, :, :data.shape[1]]
                l_patch[1] = data[0, :, data.shape[1]:]
                l_patch[2] = 0
                return l_patch

        if merge_early:
            # TODO check for heatmap-both
            patch_t = merge(patch_t)
            self.show_patch(patch_t, "merged early")

        # # TODO these fallback options to be tested
        # if self.heatmap_or_img == "img":
        #     raise NotImplemented
        #     patch_t = patch_t[:, :, :patch_t.shape[2] // 2]
        # elif self.heatmap_or_img == "heatmap":
        #     raise NotImplemented
        #     patch_t = patch_t[:, :, patch_t.shape[2] // 2:]
        both_heatmap_or_img = self.heatmap_or_img == "both"

        def clip_part(data, size):
            assert data.shape[1] == data.shape[2]
            assert data.shape[1] % 2 == 1
            _from = data.shape[1] // 2 - size // 2
            to = _from + size
            data = data[:, _from:to, _from:to]
            return data

        def crop(patch_l, size, split):
            if split:
                assert patch_l.shape[2] % 2 == 0
                img = clip_part(patch_l[:, :, :patch_l.shape[1] // 2], size)
                hm = clip_part(patch_l[:, :, patch_l.shape[1] // 2:], size)
                patch_l = torch.hstack((img, hm))
            else:
                patch_l = clip_part(patch_l, size)
            return patch_l

        if self.train_crop != PatchDataset.default_train_crop:
            assert self.train_crop % 2 == 1
            if merge_early:
                patch_t = crop(patch_t, self.train_crop, split=False)
            else:
                patch_t = crop(patch_t, self.train_crop, split=both_heatmap_or_img)

        upscale_fact = self.conf['dataset']['filtering']['train_patch_upscale_factor']
        if upscale_fact != 1.0:
            assert upscale_fact > 1.0

            original_size = patch_t.shape[1]
            # NOTE a bugfix!!!
            crop_size = patch_t.shape[1] // upscale_fact
            if crop_size % 2 == 0:
                crop_size += 1

            def upscale(data):
                data = crop(data, crop_size, split=False)
                img_p = torchvision.transforms.ToPILImage()(data)
                img_p = img_p.resize((original_size, original_size), resample=self.resample_method)
                img_p = torchvision.transforms.PILToTensor()(img_p).float() / 255.
                return img_p

            # TODO can make it configurable, but there will be more once there are different input heads
            if merge_early:
                patch_t = upscale(patch_t)
            else:
                split_index = patch_t.shape[2] // 2
                patch_t[:, :, :split_index] = upscale(patch_t[:, :, :split_index])
                patch_t[:, :, split_index:] = upscale(patch_t[:, :, split_index:])

        if not merge_early:
            patch_t = merge(patch_t)

        # if patch_t.shape[0] == 1:
        #     patch_t = patch_t.expand(3, -1, -1)
        # else:
        #     pass
        assert patch_t.shape[0] == 3

        y = torch.tensor([dy, dx]) * self.scale_error
        self.show_patch(patch_t, "after filtering", y, last=True)
        return patch_t, y

    def show_patch(self, patch_t, suffix="", y_diff=None, last=False):
        max_items = 10
        # shows = 3
        # max_counter = max_items * shows
        max_counter = max_items
        if self.input_counter < max_counter:
            patch_np = patch_t[0].numpy()
            if self.conf["dataset"]["show_inputs"]:
                plt.figure(figsize=(9, 9))
                plt.imshow(patch_np)
                sec_suffix = f"; err: {y_diff}, scale_error: {self.scale_error}" if y_diff is not None else ""
                plt.title(f"input patch no. {self.input_counter} {suffix} {sec_suffix}")
                plt.show()
                plt.close()
            if self.wandb_logger:
                img = self.patches_to_log[-1]
                to_add = patch_np
                # failed attempt at a more reasonable color palette
                # to_add = np.repeat(patch_np[:, :, None], 3, axis=2)
                # to_add = (to_add * 255.0).astype(np.uint8)
                # to_add[:, :, :2] = cv.cvtColor(to_add, cv.COLORMAP_SUMMER)
                # to_add[:, :, 2] = 0
                if img is not None:
                    img = np.vstack((img, to_add))
                else:
                    img = to_add
                self.patches_to_log[-1] = img
                if last:
                    self.patches_to_log.append(None)
        elif self.wandb_logger and self.input_counter == max_counter and last:
            #images_w = [wandb.Image(img, caption="Top: DS input, Middle: Augmented, Bottom: Final input") for img in self.patches_to_log[:-1]]
            self.wandb_logger.log_image("examples", self.patches_to_log[:-1], caption=["Top: DS input, Middle: Augmented, Bottom: Final input"] * (len(self.patches_to_log) - 1))
        if last:
            self.input_counter += 1

    def __len__(self) -> int:
        if self.augment_on_the_fly:
            # TODO centralize ?
            augment_patch_length = 6
            return len(self.metadata_list) * augment_patch_length
        else:
            return len(self.metadata_list)

    def handle_grouping(self):
        # NOTE probably broken, so let's fail early
        assert not self.grouped_by_sizes
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

    def get_upscale_method(self):
        method = self.conf['dataset']['filtering']['train_patch_upscale_method'].lower()
        if method == "bicubic":
            return Image.BICUBIC
        elif method == "lanczos":
            return Image.LANCZOS
        else:
            raise f"Unknown method '{method}'"


def augment_patch(patch, diffs, split):

    diffs_r_y = -diffs[0], diffs[1]
    diffs_r_x = diffs[0], -diffs[1]
    if split:
        split_i = patch.shape[1] // 2
        patch_r_y = torch.zeros_like(patch)
        patch_r_y[:, :split_i] = torch.flip(patch[:, :split_i], dims=[0])
        patch_r_y[:, split_i:] = torch.flip(patch[:, split_i:], dims=[0])
        patch_r_x = torch.zeros_like(patch)
        patch_r_x[:, :split_i] = torch.flip(patch[:, :split_i], dims=[1])
        patch_r_x[:, split_i:] = torch.flip(patch[:, split_i:], dims=[1])
    else:
        patch_r_y = torch.flip(patch, dims=[0])
        patch_r_x = torch.flip(patch, dims=[1])

    patches = [patch]
    diff_l = [diffs]
    augment_keys = ["original"]
    for i in range(3):
        if split:
            #show_torch(patch, "before splitting")
            split_i = patch.shape[1] // 2
            part1, part2 = patch[:, :split_i], patch[:, split_i:]
            part1 = torch.rot90(part1, 1, [0, 1])
            part2 = torch.rot90(part2, 1, [0, 1])
            patch = torch.hstack((part1, part2))
            #show_torch(patch, "after splitting, rotation and hstack")
            pass
        else:
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

    def __init__(self, conf, wandb_logger=None):
        super().__init__()

        train_conf = conf['train']
        self.conf = conf
        self.batch_size = train_conf['batch_size']
        self.grouped_by_sizes = train_conf['grouped_by_sizes']
        self.root_dir = get_full_ds_dir(conf['dataset'])
        self.wandb_logger = wandb_logger

    def prepend_parts(self, parts, part_size):
        if self.splits == 2:
            parts = [0, 0] + parts
        elif self.splits == 4:
            parts = [part_size, part_size] + parts
        else:
            raise "unexpected value for self.splits: {}".format(self.splits)
        return parts

    def get_all_metadata_list_map(self):
        train, validation, test = self.get_dss()
        return {
            "train": dict(train.metadata_list),
            "validation": dict(validation.metadata_list),
            "test": dict(test.metadata_list)
        }

    def get_dss(self):
        # needs to be called (and dss created) twice, such is life
        all = PatchDataset.get_split_datasets(self.root_dir, self.conf, self.wandb_logger)
        train = all[0]
        PatchesDataModule.normalize(train, all)
        return all

    @staticmethod
    def normalize(normalizing_dataset, datasets):

        errors = []
        for _, data_record in normalizing_dataset.metadata_list:
            dxy = np.array([data_record.dy, data_record.dx])
            errors.append(dxy)
        errors = np.array(errors)
        mean = errors.mean(axis=0)
        std_dev = np.sqrt(errors.var(axis=0))

        for ds in datasets:
            for _, dr in ds.metadata_list:
                dr.dy = (dr.dy - mean[0]) / std_dev[0]
                dr.dy = dr.dy.item()
                dr.dx = (dr.dx - mean[1]) / std_dev[1]
                dr.dx = dr.dx.item()

    def setup(self, stage: str):
        self.train, self.validation, self.test = self.get_dss()
        # FIXME probably the data need to be created lazily here, but this merhod is
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


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


def get_error_stats(entry_list):

    distances = []
    errors = []
    angles = []
    for _, data_record in entry_list:
        dxy = np.array([data_record.dy, data_record.dx])
        errors.append(dxy)
        distances.append([math.sqrt(dxy[0] ** 2 + dxy[1] ** 2)])
        angles.append([np.arctan2(dxy[0], dxy[1])])

    distances = np.array(distances)
    errors = np.array(errors)
    angles = np.array(angles)

    return distances, errors, angles


def mean_abs_mean(stat):
    mean = stat.mean(axis=0)
    abs_mean = np.abs(stat).mean(axis=0)
    return mean, abs_mean


def log_stats(scale, tags, histogram_wandb_project):

    config = get_config()
    dataset_conf = config['dataset']
    dataset_conf["tags"] = tags
    dataset_conf['down_scale'] = scale
    set_config_dir_scale_scheme(dataset_conf)
    ds_path = get_full_ds_dir(dataset_conf)

    metadata_list = PatchDataset(ds_path, config).metadata_list

    def print_stat(stat, name):
        mean, abs_mean = mean_abs_mean(stat)
        print("{} mean: {}".format(name, mean))
        print("{} abs mean: {}".format(name, abs_mean))

    distances, errors, angles = get_error_stats(metadata_list)
    print_stat(distances, "distance")
    print_stat(errors, "error")
    print_stat(angles, "angle")

    #adjustment = [0.15, 0.15]
    # distances_adjusted, errors_adjusted, angles_adjusted = get_error_stats(metadata_list, adjustment)
    # print_stat(distances_adjusted, "adjusted distance")
    # print_stat(errors_adjusted, "adjusted error")
    # print_stat(angles_adjusted, "adjusted angle")

    # TODO maybe can come in handy
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

    if histogram_wandb_project:

        wandb.init(project=histogram_wandb_project,
                   name=get_wand_name(config['dataset'], entry_list=None),
                   tags=dataset_conf["tags"])

        t_d = wandb.Table(data=distances, columns=["distance"])
        wandb.log({'distances': wandb.plot.histogram(t_d, "distance", title="distance of error")})

        t_d = wandb.Table(data=np.sqrt(distances), columns=["sqt(distances)"])
        wandb.log({'sqt(distances)': wandb.plot.histogram(t_d, "sqt(distances)", title="sqt(distances) of error")})

        t_d = wandb.Table(data=angles, columns=["angle"])
        wandb.log({'angles': wandb.plot.histogram(t_d, "angle", title="angle error")})

        # t_d = wandb.Table(data=distances_adjusted, columns=["distance"])
        # wandb.log({'distances adjusted': wandb.plot.histogram(t_d, "distance", title="distance of error adjusted")})
        #
        # t_d = wandb.Table(data=angles_adjusted, columns=["angle"])
        # wandb.log({'angles adjusted': wandb.plot.histogram(t_d, "angle", title="angle error adjusted")})


def log_min_distance(wandb_project, scale, tags):

    dataset_conf = get_config()['dataset']
    dataset_conf["tags"] = tags

    dataset_conf['down_scale'] = scale
    set_config_dir_scale_scheme(dataset_conf)
    out_dir = get_full_ds_dir(dataset_conf)

    loaded_data = np.load("{}/a_other.npz".format(out_dir))
    assert loaded_data.__contains__('minimal_dists'), "well, ..."

    wandb.init(project=wandb_project,
               name=get_wand_name(dataset_conf, entry_list=None, extra_key="minimal_distance"),
               tags=dataset_conf["tags"])

    distances = loaded_data["minimal_dists"]
    print("number of distance items: {}x{}".format(*distances.shape[:2]))

    def log_k_distance(k):
        prefix = "" if k == 0 else "{}th ".format(k + 1)
        log_table(distances[k][:, None], column="{}distance".format(prefix), table_ref="{}distances".format(prefix), title="{}distance of error".format(prefix))
        log_table(np.sqrt(distances[k][:, None]), column="sqt({}distances)".format(prefix), title="sqt({}distances) of error".format(prefix))

    max_k = 3
    for i in range(max_k):
        log_k_distance(i)

    second_to_first_ratio = distances[1][:, None] / distances[0][:, None]
    max_ratio = 5.0
    second_to_first_ratio = second_to_first_ratio[second_to_first_ratio <= max_ratio][:, None]
    log_table(second_to_first_ratio, column="2nd to 1st ratio")


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


def visualize():

    config = get_config()
    ds_path = get_full_ds_dir(config["dataset"])
    metadata_list = PatchDataset(ds_path, config).metadata_list

    # params:
    count = 5
    crop_size = 5
    split = is_hm_relevant(config)
    basic_scale = 5

    # sensible constants
    # point_size = 3
    cross_idx = np.array([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2], [2, -2], [1, -1], [-2, 2], [-1, 1]])

    # params asserts
    assert crop_size % 2 == 1

    def prepare_split_part(img_part, dr_l):
        # NOTE - scale according to crop_size
        assert img_part.shape[0] == img_part.shape[1]
        assert img_part.shape[0] % 2 == 1
        scale = basic_scale * round(img_part.shape[0] / crop_size)
        _from = img_part.shape[0] // 2 - crop_size // 2
        to = _from + crop_size
        img_patch = img_part[_from:to, _from:to]
        img_patch = cv.resize(img_patch, dsize=(img_patch.shape[1] * scale, img_patch.shape[0] * scale), interpolation=cv.INTER_NEAREST)
        img_patch = np.repeat(img_patch[:, :, None], 3, axis=2)
        center = (img_patch.shape[0] // 2)
        center = np.array([center, center])
        img_patch[(cross_idx + center)[:, 0], (cross_idx + center)[:, 1]] = [255, 0, 0]
        gt_y = center[0] + round(-dr_l.dy * scale)
        gt_x = center[1] + round(-dr_l.dx * scale)
        point = np.array([gt_y, gt_x])
        img_patch[(cross_idx + point)[:, 0], (cross_idx + point)[:, 1]] = [0, 0, 255]
        img_part = cv.resize(img_part, dsize=(img_part.shape[1] * basic_scale, img_part.shape[0] * basic_scale), interpolation=cv.INTER_NEAREST)
        return img_patch, img_part

    for i in range(count):
        file_path, dr = metadata_list[i]
        patch_np = np.array(Image.open(f"{ds_path}/data/{file_path}"))
        if split:
            width = patch_np.shape[1]
            assert width % 2 == 0
            new_width = width // 2
            img_patch, img = prepare_split_part(patch_np[:, :new_width], dr)
            heat_map_patch, heat_map = prepare_split_part(patch_np[:, new_width:], dr)
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
            fig.suptitle(f"scale = {basic_scale}, crop_size = {crop_size}, adjustment=({-dr.dx:.03f}, {-dr.dy:.03f})")
            axs[0, 0].set_title("img patch")
            axs[0, 0].imshow(img)
            axs[1, 0].set_title("img patch zoomed")
            axs[1, 0].imshow(img_patch)
            axs[0, 1].set_title("heat map")
            axs[0, 1].imshow(heat_map)
            axs[1, 1].set_title("heat map zoomed")
            axs[1, 1].imshow(heat_map_patch)
            plt.show()
            plt.close()
        else:
            raise NotImplemented


if __name__ == "__main__":

    # TODO visualize during training (best, worst, original error, adjustment)
    # visualize()

    #iterate_dataset()

    # custom scale
    scale = 0.1
    log_stats_method = "log_stats"
    log_min_distance_method = "log_min_distance"
    method = log_stats_method
    tags=["dev"]

    parser = argparse.ArgumentParser(description='Analyze data for a given scale')
    parser.add_argument('--method', dest='method', help='method', choices=[log_stats_method, log_min_distance_method], required=False)
    parser.add_argument('--scale', dest='scale', help='scale', required=False)
    args = parser.parse_args()
    if args.scale:
        scale = float(args.scale)
    if args.method:
        method = args.method

    if method == log_stats_method:
        log_stats(scale=scale, tags=tags, histogram_wandb_project="kpt_location_error_analysis_private")
    elif method == log_min_distance_method:
        log_min_distance(wandb_project="kpt_location_error_analysis_private", scale=scale, tags=tags)

    #conf = get_config()
    #log_stats("dataset/superpoint_30_files_int_33", wandb_project, conf)
    #t_data_record()
