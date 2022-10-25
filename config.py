import omegaconf
from omegaconf import OmegaConf

import cv2 as cv
from kornia_sift import NumpyKorniaSiftDetector
from sift_detectors import AdjustedSiftDetector
from scale_pyramid import MyScalePyramid
import numpy as np


def get_config(path='config/config.yaml'):
    config = OmegaConf.load(path)
    # TODO others
    validate_wrt_detector(config)
    return config


def validate_wrt_detector(config):
    dataset_config = config['dataset']
    detector_name = dataset_config['detector'].lower()
    if detector_name == 'superpoint':
        msg = "invalid value for superpoint detector"
        assert dataset_config['scale_ratio_th'] is None, msg
        min_scale_th = dataset_config['min_scale_th']
        assert min_scale_th == 0.0, msg # TODO allow None
        # NOTE: scale_ratio_th is None
        assert not dataset_config['dynamic_resizing'], msg


def is_hm_relevant(conf):
    key = conf['dataset']['detector']
    if type(key) == str:
        return key.lower().__contains__("superpoint")
    else:
        False


def get_full_ds_dir(config):
    path_size = config.get('const_patch_size')
    out_dir = config['out_dir']
    if path_size is None:
        return out_dir
    else:
        return out_dir + str(path_size)


def get_detector(config):
    # TODO parameters for all
    # NOTE I stuck with the cv API as e.g. scale can be used

    key = config['detector']
    return get_detector_by_key(key)


def set_config_dir_scale_scheme(dataset_config):
    scale = dataset_config['down_scale']
    dn = dataset_config['detector']
    err = dataset_config['err_th']
    max_files = dataset_config['max_files']
    dataset_config['out_dir'] = "dataset/{}_new_err_{}_files_{}_scale_{}_size_".format(dn, err, max_files, scale).replace(".", "_")


def get_detector_by_key(dict_key):
    if type(dict_key) == str:
        dict_key = dict_key.lower()
        if dict_key == 'sift':
            return cv.SIFT_create()
        # TODO add "improved kornia SIFT"
        # custom_scale_pyramid = MyScalePyramid(3, 1.6, 32, double_image=True)
        # detector = NumpyKorniaSiftDetector(scale_pyramid=custom_scale_pyramid)
        elif dict_key == 'adjusted_sift':
            return AdjustedSiftDetector()
        elif dict_key == 'sift_kornia':
            return NumpyKorniaSiftDetector()
        elif dict_key == 'adjusted_sift_kornia':
            custom_scale_pyramid = MyScalePyramid(3, 1.6, 32, double_image=True)
            return NumpyKorniaSiftDetector(scale_pyramid=custom_scale_pyramid)
        elif dict_key == 'superpoint':
            from superpoint_local import SuperPointDetector
            return SuperPointDetector()
        elif dict_key == 'adjusted_superpoint':
            from superpoint_local import SuperPointDetector
            # 3 SE translations
            translations = np.array([[4, 4], [4, 0], [0, 4]])
            # 9 translations
            # translations = np.array([[4, 4], [4, 0], [0, 4], [2, 2], [2, 0], [0, 2], [1, 1], [1, 0], [0, 1]])
            # 8 centered [4/0] translations
            # translations = np.array([[4, 4], [4, 0], [0, 4], [-4, -4], [-4, 0], [0, -4], [-4, 4], [4, -4]])
            translations = []
            # rotations = range(1, 4)
            rotations = []
            # const_adjustment = None
            const_adjustment = [0.45, 0.30]
            return SuperPointDetector(const_adjustment=const_adjustment, translations=translations, rotations=rotations)
        else:
            raise "unrecognized detector: {}".format(dict_key)
    else:
        raise ValueError(f"Unknown type: {type(dict_key)}")
