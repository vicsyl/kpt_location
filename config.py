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

    name = config['detector'].lower()
    return get_detector_by_name(name)


def set_config_dir_scale_scheme(dataset_config):
    scale = dataset_config['down_scale']
    dn = dataset_config['detector']
    err = dataset_config['err_th']
    max_files = dataset_config['max_files']
    dataset_config['out_dir'] = "dataset/{}_new_err_{}_files_{}_scale_{}_size_".format(dn, err, max_files, scale).replace(".", "_")


def get_detector_by_name(name):
    if name == 'sift':
        return cv.SIFT_create()
    # TODO add "improved kornia SIFT"
    # custom_scale_pyramid = MyScalePyramid(3, 1.6, 32, double_image=True)
    # detector = NumpyKorniaSiftDetector(scale_pyramid=custom_scale_pyramid)
    elif name == 'adjusted_sift':
        return AdjustedSiftDetector()
    elif name == 'sift_kornia':
        return NumpyKorniaSiftDetector()
    elif name == 'adjusted_sift_kornia':
        custom_scale_pyramid = MyScalePyramid(3, 1.6, 32, double_image=True)
        return NumpyKorniaSiftDetector(scale_pyramid=custom_scale_pyramid)
    elif name == 'superpoint':
        from superpoint_local import SuperPointDetector
        return SuperPointDetector()
    elif name == 'adjusted_superpoint':
        from superpoint_local import SuperPointDetector
        translations = np.array([[4, 4], [4, 0], [0, 4], [2, 2], [2, 0], [0, 2], [1, 1], [1, 0], [0, 1]])
        return SuperPointDetector(translations=translations)
    else:
        raise "unrecognized detector: {}".format(name)
