from omegaconf import OmegaConf
import torch

import cv2 as cv
from kornia_sift import NumpyKorniaSiftDescriptor
from sift_detectors import AdjustedSiftDescriptor
from scale_pyramid import MyScalePyramid
import numpy as np
from superpoint_local import SuperPointDetector


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
    # # TODO parameters for all
    # # NOTE I stuck with the cv API as e.g. scale can be used
    #
    # nearest_fix_sp_d_f = MyScalePyramid(3, 1.6, 32, double_image=True, interpolation_mode='nearest',
    #                                     gauss_separable=True, every_2nd=False)
    # nearest_fix_sp_d_t = MyScalePyramid(3, 1.6, 32, double_image=True, interpolation_mode='nearest',
    #                                     gauss_separable=True, every_2nd=True)
    #
    # l = []
    #
    # for flip in [1, -1]:
    #     for adj in [0, 0.25, 0.5]:
    #         adj_r = flip * adj
    #         l.append(NumpyKorniaSiftDescriptor(num_features=8000,
    #                                   conv_quad_interp_adjustment=0,
    #                                   scale_pyramid=nearest_fix_sp_d_f,
    #                                   scatter_fix=False, swap_xy_fix=False, adjustment=[adj_r]))
    #
    #         for q_adj in [0, -0.375, -0.5, -0.75, -1.0]:
    #             if adj == 0 and q_adj == 0 and flip == -1:
    #                 continue
    #             q_adj_r = flip * q_adj
    #             l.append(NumpyKorniaSiftDescriptor(num_features=8000,
    #                                       conv_quad_interp_adjustment=q_adj_r,
    #                                       scale_pyramid=nearest_fix_sp_d_t,
    #                                       scatter_fix=True, swap_xy_fix=True, adjustment=[adj_r]))
    #
    # counter = config['counter']
    # print(f"COUNTER: {counter}")
    # return l[counter]
    key = config['detector']
    return get_detector_by_key(key)


def set_config_dir_scale_scheme(dataset_config):
    scale = dataset_config['down_scale']
    dn = dataset_config['detector']
    err = dataset_config['err_th']
    max_files = dataset_config['max_files']
    dataset_config['out_dir'] = "dataset/{}_new_err_{}_files_{}_scale_{}_size_".format(dn, err, max_files, scale).replace(".", "_")


# TODO needs a redo
def get_detector_by_key(dict_key):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if type(dict_key) == str:
        dict_key = dict_key.lower()
        if dict_key == 'sift':
            return AdjustedSiftDescriptor(adjustment=[0.0, 0.0])
        elif dict_key == 'adjusted_sift':
            return AdjustedSiftDescriptor(adjustment=[-0.25, -0.25])
        # elif dict_key == 'adjusted_sift_negative':
        #     return AdjustedSiftDescriptor(adjustment=[-0.25, -0.25])
        # elif dict_key == 'adjusted_sift_linear':
        #     return AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.11, -0.11])
        elif dict_key == 'sift_kornia':

            original_sp = MyScalePyramid(3, 1.6, 32,
                                         double_image=True,
                                         interpolation_mode='nearest',
                                         gauss_separable=True,
                                         every_2nd=False)
            num_features = 8000
            kornia_original = NumpyKorniaSiftDescriptor(name=f"Kornia baseline {num_features}",
                                                         num_features=num_features,
                                                         scale_pyramid=original_sp,
                                                         scatter_fix=False, swap_xy_fix=False)

            return kornia_original
        elif dict_key == 'sift_kornia_adjusted':

            original_sp = MyScalePyramid(3, 1.6, 32,
                                         double_image=True,
                                         interpolation_mode='nearest',
                                         gauss_separable=True,
                                         every_2nd=False)
            num_features = 8000
            kornia_adjusted = NumpyKorniaSiftDescriptor(name=f"Kornia baseline adjusted {num_features}",
                                                         adjustment=[-0.25, -0.25],
                                                         num_features=num_features,
                                                         scale_pyramid=original_sp,
                                                         scatter_fix=False, swap_xy_fix=False)

            return kornia_adjusted
        elif dict_key == 'sift_kornia_fixed':

            nearest_fix_sp = MyScalePyramid(3, 1.6, 32,
                                            double_image=True,
                                            interpolation_mode='nearest',
                                            gauss_separable=True,
                                            every_2nd=True,
                                            better_up=True)
            num_features = 8000
            kornia_correct = NumpyKorniaSiftDescriptor(name=f"Kornia fixed {num_features}",
                                                       num_features=num_features,
                                                       scale_pyramid=nearest_fix_sp,
                                                       scatter_fix=True, swap_xy_fix=True)

            return kornia_correct
        elif dict_key == 'adjusted_sift_kornia':
            #custom_scale_pyramid = MyScalePyramid(3, 1.6, 32, double_image=True)
            return NumpyKorniaSiftDescriptor(interpolation_mode='bilinear')
        elif dict_key == 'superpoint':
            return SuperPointDetector(device=device)
        elif dict_key == 'adjusted_superpoint':
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
            return SuperPointDetector(device=device, const_adjustment=const_adjustment, translations=translations, rotations=rotations)
        else:
            raise ValueError(f"unrecognized detector: {dict_key}")
    else:
        raise ValueError(f"Unknown type: {type(dict_key)}")
