import os
import re
import shutil


def get_dirs_and_keys(dirs_to_process, base_dir):

    pattern = re.compile("^scene_cam.*final_preview$")
    dirs = list_scenes(base_dir)[:dirs_to_process]

    in_dirs = []
    keys = []

    for dir in dirs:
        print("processing: {}".format(dir))
        path = "{}/images".format(dir)
        if not os.path.isdir(path):
            continue
        for prev_dir_name in list(os.listdir(path)):
            print(prev_dir_name)
            if pattern.match(prev_dir_name) is not None:
                full_path = "{}/{}".format(path, prev_dir_name)
                key = full_path[len(base_dir) + 1:].replace("/", "_")
                in_dirs.append(full_path)
                keys.append(key)
                print(full_path)
                print(key)

    print("in_dirs: {}".format(in_dirs))
    print("keys: {}".format(keys))
    return in_dirs, keys


def list_scenes(base_dir):
    print("Listing scenes in {}".format(base_dir))
    pattern = re.compile("^ai_0.._0..$")
    paths = []
    for cur_dir_name in list(os.listdir(base_dir)):
        if pattern.match(cur_dir_name) is not None:
            full_path = "{}/{}".format(base_dir, cur_dir_name)
            paths.append(full_path)

    return paths


def clean_scene(scene_path):

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
        if pattern.match(cur_dir_name) is None:
            full_path = "{}/{}".format(path, cur_dir_name)
            remove_all(full_path)


if __name__ == "__main__":
    clean_scene("scenes/ai_001_001_foo")
    # for s in list_scenes("scenes"):
    #     print(s)
    # dirs, keys = get_dirs_and_keys(1000, "./scenes")
