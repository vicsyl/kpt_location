import os
import re
import shutil


def clean_scene(scene_path):

    pattern = re.compile("^scene_cam.*final_preview$")
    print("Cleaning scene in {}".format(scene_path))

    for cur_dir_name in list(os.listdir(scene_path)):
        if cur_dir_name != "images":
            full_path = "{}/{}".format(scene_path, cur_dir_name)
            print("removing {}".format(full_path))
            shutil.rmtree(full_path)

    path = "{}/images".format(scene_path)
    for cur_dir_name in list(os.listdir(path)):
        if pattern.match(cur_dir_name) is None:
            full_path = "{}/{}".format(path, cur_dir_name)
            print("removing {}".format(full_path))
            shutil.rmtree(full_path)


if __name__ == "__main__":
    clean_scene("scenes/ai_001_001_foo")
