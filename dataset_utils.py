import os
import re
import shutil
from config import get_full_ds_dir


def get_dirs_and_keys(dirs_to_process, base_dir_unzips, ds_config):

    pattern = re.compile("^scene_cam.*final_preview$")
    unzips = list_scenes_to_process(base_dir_unzips, ds_config)[:dirs_to_process]
    zip_dirs_map = {}

    for unzip in unzips:
        in_dirs = []
        keys = []
        path = f"{base_dir_unzips}/{unzip}"
        print("processing: {}".format(path))
        path = "{}/images".format(path)
        if not os.path.isdir(path):
            continue
        for prev_dir_name in list(os.listdir(path)):
            print(prev_dir_name)
            if pattern.match(prev_dir_name) is not None:
                full_path = "{}/{}".format(path, prev_dir_name)
                key = full_path[len(base_dir_unzips) + 1:].replace("/", "_")
                in_dirs.append(full_path)
                keys.append(key)
        zip_dirs_map[unzip] = (in_dirs, keys)
        print(f"in_dirs for {unzip}: {in_dirs}")
        print(f"keys for {unzip}: {keys}")

    return zip_dirs_map


def list_scenes_to_process(base_dir_unzips, ds_config):

    base_out_dir = get_full_ds_dir(ds_config)
    pattern = re.compile("^ai_.*")
    paths = []
    for cur_dir_name in list(os.listdir(base_dir_unzips)):
        if pattern.match(cur_dir_name) is not None:
            check_existing = f"{base_out_dir}/{cur_dir_name}"
            if os.path.isdir(check_existing):
                print(f"{check_existing} skipped, exists")
            else:
                paths.append(cur_dir_name)
    return paths


def clean_scene(scene_path, ends_with=None):

    def remove_all(path):
        print("removing {}".format(path))
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    pattern = re.compile("^scene_cam.*final_preview$")
    print("Cleaning scene in {}".format(scene_path))

    for cur_dir_name in list(os.listdir(scene_path)):
        if cur_dir_name != "images":
            full_path = "{}/{}".format(scene_path, cur_dir_name)
            remove_all(full_path)

    path = "{}/images".format(scene_path)
    for cur_dir_name in list(os.listdir(path)):
        full_path = "{}/{}".format(path, cur_dir_name)
        if pattern.match(cur_dir_name) is None:
            full_path = "{}/{}".format(path, cur_dir_name)
            remove_all(full_path)
        else:
            if ends_with:
                for cur_image_name in list(os.listdir(full_path)):
                    if not cur_image_name.endswith(ends_with):
                        remove_all("{}/{}".format(full_path, cur_image_name))


if __name__ == "__main__":
    # clean_scene("scenes/ai_001_001_foo")
    # for s in list_scenes("scenes"):
    #     print(s)
    # dirs, keys = get_dirs_and_keys(1000, "./scenes")
    clean_scene("scenes/ai_001_001", '.tonemap.jpg')
