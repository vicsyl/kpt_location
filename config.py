from omegaconf import OmegaConf
import cv2 as cv

import sys
sys.path.append("./superpoint_forked")
from superpoint import SuperPointDescriptor
import torch


def get_config(path='config/config.yaml'):
    config = OmegaConf.load(path)
    # TODO others
    validate_wrt_detector(config)
    return config


def validate_wrt_detector(config):
    detector_name = config['dataset']['detector'].lower()
    if detector_name == 'superpoint':
        msg = "invalid value for superpoint detector"
        assert config['scale_ratio_th'] is None, msg
        min_scale_th = config['min_scale_th']
        assert min_scale_th == 0.0, msg # TODO None?


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
    if name == 'sift':
        return cv.SIFT_create()
    elif name == 'superpoint':
        return SuperPointDetector()
    else:
        raise "unrecognized detector: {}".format(name)

class SuperPointDetector:

    def __init__(self, path=None, device: torch.device = torch.device('cpu')):
        if not path:
            path = "./superpoint_forked/superpoint_v1.pth"
        self.super_point = SuperPointDescriptor(path, device)

    def detect(self, img_np, mask=None):
        pts, _ = self.super_point.detectAndComputeGrey(img_np, mask)
        return pts
