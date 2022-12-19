import math
import argparse

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from kornia_sift import NumpyKorniaSiftDescriptor
from scale_pyramid import MyScalePyramid
from sift_detectors import AdjustedSiftDescriptor
from lowe_sift_file_descriptor import LoweSiftDescriptor
from superpoint_local import SuperPointDetector
from utils import get_tentatives

from hloc_sift import HlocSiftDescriptor
from utils import csv2latex


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def decompose_homographies(Hs):
    """
    :param Hs:(B, 3, 3)
    :param device:
    :return: pure_homographies(B, 3, 3), affine(B, 3, 3)
    """

    B, three1, three2 = Hs.shape
    assert three1 == 3
    assert three2 == 3

    device = get_device()
    def batched_eye_deviced(B, D):
        eye = torch.eye(D, device=device)[None].repeat(B, 1, 1)
        return eye

    KR = Hs[:, :2, :2]
    KRt = -Hs[:, :2, 2:3]
    # t = torch.inverse(KR) @ KRt # for the sake of completeness - this is unused
    a_t = Hs[:, 2:3, :2] @ torch.inverse(KR)
    b = a_t @ KRt + Hs[:, 2:3, 2:3]

    pure_homographies1 = torch.cat((batched_eye_deviced(B, 2), torch.zeros(B, 2, 1, device=device)), dim=2)
    pure_homographies2 = torch.cat((a_t, b), dim=2)
    pure_homographies = torch.cat((pure_homographies1, pure_homographies2), dim=1)

    affines1 = torch.cat((KR, -KRt), dim=2)
    affines2 = torch.cat((torch.zeros(B, 1, 2, device=device), torch.ones(B, 1, 1, device=device)), dim=2)
    affines = torch.cat((affines1, affines2), dim=1)

    assert torch.all(affines[:, 2, :2] == 0)
    test_compose_back = pure_homographies @ affines
    assert torch.allclose(test_compose_back, Hs, rtol=1e-01, atol=1e-01)
    return pure_homographies, affines


def get_visible_part_mean_absolute_reprojection_error(img1, img2, H_gt, H, metric="L2"):
    '''We reproject the image 1 mask to image2 and back to get the visible part mask.
    Then we average the reprojection absolute error over that area'''
    h, w = img1.shape[:2]
    mask1 = np.ones((h, w))
    mask1in2 = cv.warpPerspective(mask1, H_gt, img2.shape[:2][::-1])
    mask1inback = cv.warpPerspective(mask1in2, np.linalg.inv(H_gt), img1.shape[:2][::-1]) > 0
    xi = np.arange(w)
    yi = np.arange(h)
    xg, yg = np.meshgrid(xi, yi)
    coords = np.concatenate([xg.reshape(*xg.shape, 1), yg.reshape(*yg.shape, 1)], axis=-1)
    shape_orig = coords.shape
    xy_rep_gt = cv.perspectiveTransform(coords.reshape(-1, 1, 2).astype(np.float32), H_gt.astype(np.float32)).squeeze(1)
    xy_rep_estimated = cv.perspectiveTransform(coords.reshape(-1, 1, 2).astype(np.float32),
                                               H.astype(np.float32)).squeeze(1)
    metric = metric.upper()
    if metric == "L1":
        error = np.abs(xy_rep_gt-xy_rep_estimated).sum(axis=1).reshape(xg.shape) * mask1inback
        mean_error = error.sum() / mask1inback.sum()
    elif metric == "L2":
        error = np.sqrt(((xy_rep_gt - xy_rep_estimated) ** 2).sum(axis=1)).reshape(xg.shape) * mask1inback
        mean_error = error.sum() / mask1inback.sum()
    elif metric == "VEC":
        error = (xy_rep_estimated - xy_rep_gt).reshape(xg.shape + (2,))
        mask = np.tile(mask1inback[:, :, None], (1, 1, 2))
        error = error * mask
        mean_error = error.sum(axis=0).sum(axis=0) / mask1inback.sum()

    return mean_error


def decolorize(img):
    return cv.cvtColor(cv.cvtColor(img, cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)


def draw_matches(kps1, kps2, tentative_matches, H_est, H_gt, inlier_mask, img1, img2):
    h = img1.shape[0]
    w = img1.shape[1]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    def possibly_decolorize(img_local):
        if len(img_local.shape) <= 2:
            return img2
        return decolorize(img_local)

    img1_dec = possibly_decolorize(img1)
    img2_dec = possibly_decolorize(img2)

    if H_est is not None:
        dst = cv.perspectiveTransform(pts, H_est)
        img2_dec = cv.polylines(img2_dec, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
        dst = cv.perspectiveTransform(pts, H_gt)
        img2_dec = cv.polylines(img2_dec, [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
    # else:
    #     img2_tr = img2_dec

    matches_mask = inlier_mask.ravel().tolist()

    # Blue is estimated homography
    draw_params = dict(matchColor=(255, 255, 0),  # draw matches in yellow color
                       singlePointColor=None,
                       matchesMask=matches_mask,  # draw only inliers
                       flags=20)
    img_out = cv.drawMatches(img1_dec, kps1, img2_dec, kps2, tentative_matches, None, **draw_params)
    return img_out


def read_imgs(file_paths, show=False, crop=None):
    imgs = []
    for i, file in enumerate(file_paths):
        img = Image.open(file)
        img = np.array(img)

        def modulo32(n, modulo):
            n_n = n - ((n - modulo) % 32)
            assert n_n % 32 == modulo
            return n_n
        if crop:
            h, w = img.shape[:2]
            w = modulo32(w, crop)
            h = modulo32(h, crop)
            img = img[:h, :w]

        imgs.append(img)
        if show:
            plt.figure()
            plt.imshow(img)
            plt.title(i + 1)
            plt.show()
            plt.close()
    return imgs


def print_Hs_decomposition(Hs):

    print("scale\trotation")
    for H_gt in Hs:

        pure_homography, affine = decompose_homographies(torch.from_numpy(H_gt[None]).to(get_device()))

        affine = affine[0].cpu().numpy()
        # print(f"affine: {affine}")

        det = np.linalg.det(affine)
        scale = math.sqrt(det)
        affine = affine / scale

        cos_avg = (affine[0, 0] + affine[1, 1]) / 2.0
        sin_avg = (affine[0, 1] - affine[1, 0]) / 2.0
        alpha = math.atan2(sin_avg, cos_avg) * 180 / math.pi
        pure_homography = pure_homography[0].cpu().numpy()
        norm = np.linalg.norm(pure_homography[2, :2])
        print(f"{scale:.3f}\t{alpha:.3f}")


def run_experiments(detector_sets):

    detector_sets = [d.lower() for d in detector_sets]

    vlfeat_sift_descriptors = [
        HlocSiftDescriptor(HlocSiftDescriptor.opencv_like_conf),
        HlocSiftDescriptor(HlocSiftDescriptor.opencv_like_conf, [0.25, 0.25]),
        HlocSiftDescriptor(HlocSiftDescriptor.opencv_like_conf, [-0.25, -0.25]),
        HlocSiftDescriptor(HlocSiftDescriptor.default_conf),
        HlocSiftDescriptor(HlocSiftDescriptor.default_conf, [0.25, 0.25]),
        HlocSiftDescriptor(HlocSiftDescriptor.default_conf, [-0.25, -0.25]),
        # HlocSiftDescriptor([-0.5, -0.5]),
    ]

    lowe_sift_descriptors = [
        LoweSiftDescriptor(),
        LoweSiftDescriptor([0.25, 0.25]),
        LoweSiftDescriptor([-0.25, -0.25]),
    ]

    num_features = 8000
    nearest_fix_sp = MyScalePyramid(3, 1.6, 32, double_image=False, interpolation_mode='nearest', gauss_separable=True, every_2nd=True)
    nearest_fix_sp_d = MyScalePyramid(3, 1.6, 32, double_image=True, interpolation_mode='nearest', gauss_separable=True, every_2nd=True)
    nearest_sp = MyScalePyramid(3, 1.6, 32, double_image=False, interpolation_mode='nearest', gauss_separable=True, every_2nd=False)
    nearest_sp_d = MyScalePyramid(3, 1.6, 32, double_image=True, interpolation_mode='nearest', gauss_separable=True, every_2nd=False)

    kornia_sift_descriptors_nms_compensate = [
        NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp, scatter_fix=True,
                                  swap_xy_fix=True, compensate_nms_dim_minus_1=False),
        NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp_d, scatter_fix=True,
                                  swap_xy_fix=True, compensate_nms_dim_minus_1=False),
        NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp, scatter_fix=True,
                                  swap_xy_fix=True, compensate_nms_dim_minus_1=True),
        NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp_d, scatter_fix=True,
                                  swap_xy_fix=True, compensate_nms_dim_minus_1=True),
    ]

    kornia_sift_descriptors = []
    kornia_sift_descriptors_single_image = []

    # for i in range(5):
    #     adj = -0.125 * i

    for flip in [1, -1]:
        for adj_q in [0.0, 0.25, 0.5, 0.75]:
            for adj_a in [0.0, -0.25, -0.5, -0.75]:
                adj_q_r = adj_q * flip
                adj_a_r = adj_a * flip
                l = [
                    NumpyKorniaSiftDescriptor(adjustment=adj_a_r, num_features=num_features, scale_pyramid=nearest_fix_sp_d, scatter_fix=True, swap_xy_fix=True, conv_quad_interp_adjustment=adj_q_r),
                ]
                kornia_sift_descriptors.extend(l)


    for flip in [1, -1]:
        for adj_q in [0.0, 0.25, 0.5, 0.75]:
            for adj_a in [0.0, -0.25, -0.5, -0.75]:
                adj_q_r = adj_q * flip
                adj_a_r = adj_a * flip
                l = [
                    NumpyKorniaSiftDescriptor(adjustment=adj_a_r, num_features=num_features, scale_pyramid=nearest_fix_sp, scatter_fix=True, swap_xy_fix=True, conv_quad_interp_adjustment=adj_q_r),
                ]
                kornia_sift_descriptors_single_image.extend(l)


        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_sp_d),

        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp, scatter_fix=True, swap_xy_fix=True),
        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp_d, scatter_fix=True, swap_xy_fix=True),
        #
        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp, scatter_fix=True, swap_xy_fix=False),
        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp_d, scatter_fix=True, swap_xy_fix=False),
        #
        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp, scatter_fix=False, swap_xy_fix=True),
        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp_d, scatter_fix=False, swap_xy_fix=True),

        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp, scatter_fix=False, swap_xy_fix=False),
        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp_d, scatter_fix=False, swap_xy_fix=False),

        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp, scatter_fix=False, swap_xy_fix=False),
        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp_d, scatter_fix=False, swap_xy_fix=False),


        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_fix_sp_d),
        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_sp),
        # NumpyKorniaSiftDescriptor(num_features=num_features, scale_pyramid=nearest_sp_d),

        # NumpyKorniaSiftDescriptor(num_features=num_features),
        # NumpyKorniaSiftDescriptor(num_features=num_features, adjustment=[0.25, 0.25]),
        # NumpyKorniaSiftDescriptor(num_features=num_features, adjustment=[-0.25, -0.25]),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='bilinear'),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='bilinear', adjustment=[0.25, 0.25]),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='bilinear', adjustment=[-0.25, -0.25]),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='lanczos'),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='lanczos', adjustment=[0.25, 0.25]),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='lanczos', adjustment=[-0.25, -0.25]),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='bicubic'),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='bicubic', adjustment=[0.25, 0.25]),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='bicubic', adjustment=[-0.25, -0.25]),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='bilinear', adjustment=[-0.5, -0.5]),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='bilinear', adjustment=[0.4, 0.4]),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='nearest', adjustment=[0.4, 0.4]),
        # NumpyKorniaSiftDescriptor(num_features=num_features, interpolation_mode='bilinear', adjustment=[0.5, 0.5]),
    # ]

    test_descriptors = [
        AdjustedSiftDescriptor(adjustment=[0.0, 0.0], str="OpenCV; +0.25; bilinear"),
        AdjustedSiftDescriptor(adjustment=[0.0, 0.0], str="OpenCV; -0.25; nearest"),
        AdjustedSiftDescriptor(adjustment=[0.0, 0.0], str="VLFeat; +0.25; bilinear"),
        AdjustedSiftDescriptor(adjustment=[0.0, 0.0], str="VLFeat; -0.25; nearest"),
    ]

    cv_sift_descriptors = [
            AdjustedSiftDescriptor(adjustment=[0.0, 0.0]),
            AdjustedSiftDescriptor(adjustment=[0.25, 0.25]),
            AdjustedSiftDescriptor(adjustment=[-0.25, -0.25]),
            # AdjustedSiftDescriptor(adjustment=[-0.5, -0.5], q_adjustment=[0.09, 0.09]),
            # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.11, -0.11]),
            # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.2, -0.2]),
            # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.25, -0.25]),
        ]

        # AdjustedSiftDescriptor(adjustment=[-0.5, -0.5], q_adjustment=[0.0925, 0.0925]),
        # AdjustedSiftDescriptor(adjustment=[-0.5, -0.5], q_adjustment=[0.2, 0.2]),
        # AdjustedSiftDescriptor(adjustment=[-0.5, -0.5], q_adjustment=[0.0, 0.0]),
        # AdjustedSiftDescriptor(adjustment=[-0.5, -0.5], q_adjustment=[-0.1, -0.1]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[0.0, 0.0]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.05, -0.05]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.1, -0.1]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.15, -0.15]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.2, -0.2]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.05, -0.05]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.1, -0.1]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.15, -0.15]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.18, -0.18]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.19, -0.19]),
        # !!!
        # AdjustedSiftDescriptor(adjustment=[0.19, 0.19], q_adjustment=[-0.2, -0.2]),
        # AdjustedSiftDescriptor(adjustment=[0.21, 0.21], q_adjustment=[-0.2, -0.2]),
        # AdjustedSiftDescriptor(adjustment=[0.23, 0.23], q_adjustment=[-0.2, -0.2]),

        # AdjustedSiftDescriptor(adjustment=[0.27, 0.27], q_adjustment=[-0.2, -0.2]),
        # AdjustedSiftDescriptor(adjustment=[0.29, 0.29], q_adjustment=[-0.2, -0.2]),
        # AdjustedSiftDescriptor(adjustment=[0.31, 0.31], q_adjustment=[-0.2, -0.2]),

        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.21, -0.21]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.22, -0.22]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.23, -0.23]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.24, -0.24]),
        # AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.30, -0.30]),

    # https://www.robots.ox.ac.uk/~vgg/data/affine/

    Hs_bark = [
        [[0.7022029025774007, 0.4313737491020563, -127.94661199701689],
         [-0.42757325092889575, 0.6997834349758094, 201.26193857481698],
         [4.083733373964227E-6, 1.5076445750988132E-5, 1.0]],

        [[-0.48367041358997964, -0.2472935325077872, 870.2215120216712],
         [0.29085746679198893, -0.45733473891783305, 396.1604918833091],
         [-3.578663704630333E-6, 6.880007548843957E-5, 1.0]],

        [[-0.20381418476462312, 0.3510201271914591, 247.1085214229702],
         [-0.3499531830464912, -0.1975486500576974, 466.54576370699766],
         [-1.5735788289619667E-5, 1.0242951905091244E-5, 1.0]],

        [[0.30558415717792214, 0.12841186681168829, 200.94588793078017],
         [-0.12861248979242065, 0.3067557133397112, 133.77000196887894],
         [2.782320090398499E-6, 5.770764104061954E-6, 1.0]],

        [[-0.23047631546234373, -0.10655686701035443, 583.3200507850402],
         [0.11269946585180685, -0.20718914340861153, 355.2381263740649],
         [-3.580280012615393E-5, 3.2283960511548054E-5, 1.0]],
    ]

    Hs_bark = np.array(Hs_bark)
    files_bark = [f"demo_imgs/bark/img{i + 1}.ppm" for i in range(6)]
    imgs_bark = read_imgs(files_bark, show=False)
    imgs_bark_lowe = [f"demo_imgs/lowe_all/keys/bark_{i + 1}.pgm.key" for i in range(6)]

    print("BARK experiment hompographies decomposition")
    print_Hs_decomposition(Hs_bark)
    print()

    Hs_boat = [
        [[8.5828552e-01, 2.1564369e-01, 9.9101418e+00],
         [-2.1158440e-01, 8.5876360e-01, 1.3047838e+02],
         [2.0702435e-06, 1.2886110e-06, 1.0000000e+00]],

        [[5.6887079e-01, 4.6997572e-01, 2.5515642e+01],
         [-4.6783159e-01, 5.6548769e-01, 3.4819925e+02],
         [6.4697420e-06, -1.1704138e-06, 1.0000000e+00]],

        [[1.0016637e-01, 5.2319717e-01, 2.0587932e+02],
         [-5.2345249e-01, 8.7390786e-02, 5.3454522e+02],
         [9.4931475e-06, -9.8296917e-06, 1.0000000e+00]],

        [[4.2310823e-01, -6.0670438e-02, 2.6635003e+02],
         [6.2730152e-02, 4.1652096e-01, 1.7460201e+02],
         [1.5812849e-05, -1.4368783e-05, 1.0000000e+00]],

        [[2.9992872e-01, 2.2821975e-01, 2.2930182e+02],
         [-2.3832758e-01, 2.4564042e-01, 3.6767399e+02],
         [9.9064973e-05, -5.8498673e-05, 1.0000000e+00]]
    ]
    Hs_boat = np.array(Hs_boat)
    files_boat = [f"demo_imgs/boat/img{i + 1}.pgm" for i in range(6)]
    imgs_boat = read_imgs(files_boat, show=False)
    imgs_boat_lowe = [f"demo_imgs/lowe_all/keys/boat_{i + 1}.pgm.key" for i in range(6)]

    print("BOAT experiment hompographies decomposition")
    print_Hs_decomposition(Hs_boat)

    Hs_gt_rot, imgs_rot = Hs_imgs_for_rotation(files_bark[0], show=False)

    Hs_gt_rot_cropped, imgs_rot_cropped = Hs_imgs_for_rotation(files_bark[0], show=False, crop=1)

    imgs_rotation_lowe = [f"demo_imgs/lowe_all/keys/pure_rotation_0_rot_{i}.pgm.key" for i in range(4)]

    scales = [scale_int / 10 for scale_int in range(1, 10)]
    # Hs_gt_sc_backward, imgs_sc_backward = Hs_imgs_for_scaling(files_bark[0], scales, show=False, mod4=True)
    Hs_gt_sc_lanczos, imgs_sc_lanczos = Hs_imgs_for_scaling(files_bark[0], scales, mode='lanczos')
    Hs_gt_sc_lin, imgs_sc_lin = Hs_imgs_for_scaling(files_bark[0], scales, mode='linear')
    Hs_gt_sc_hom, imgs_sc_hom = Hs_imgs_for_scaling(files_bark[0], scales, mode='linear_homography')

    scale_indices = [10] + list(range(1, 10))
    # imgs_scale_lowe_linear_cv_mod4 = [f"demo_imgs/lowe_all/keys/pure_scale_lanczos_False_resized_lin_pil_False_mod4_True_0_{i}.pgm.key" for i in l]
    # imgs_scale_lowe_linear_pil_mod4 = [f"demo_imgs/lowe_all/keys/pure_scale_lanczos_False_resized_lin_pil_True_mod4_True_0_{i}.pgm.key" for i in l]
    # imgs_scale_lowe_linear_pil = [f"demo_imgs/lowe_all/keys/pure_scale_lanczos_False_resized_lin_pil_True_mod4_False_0_{i}.pgm.key" for i in l]
    # imgs_scale_lowe_lanczos_mod4 = [f"demo_imgs/lowe_all/keys/pure_scale_lanczos_True__mod4_True_0_{i}.pgm.key" for i in l]
    imgs_scale_lowe_linear_cv = [f"demo_imgs/lowe_all/keys/pure_scale_lanczos_False_resized_lin_pil_False_mod4_False_0_{i}.pgm.key" for i in scale_indices]
    imgs_scale_lowe_lanczos = [f"demo_imgs/lowe_all/keys/pure_scale_lanczos_True__mod4_False_0_{i}.pgm.key" for i in scale_indices]

    if 'opencv' in detector_sets:
        run_exp(cv_sift_descriptors, Hs_bark, imgs_bark, "bark")
        run_exp(cv_sift_descriptors, Hs_boat, imgs_boat, "boat")
        run_exp(cv_sift_descriptors, Hs_gt_rot, imgs_rot, "synthetic pi rotation")
        run_exp(cv_sift_descriptors, Hs_gt_sc_lanczos, imgs_sc_lanczos, "synthetic rescaling lanczos")
        run_exp(cv_sift_descriptors, Hs_gt_sc_hom, imgs_sc_hom, "synthetic rescaling homography")
        run_exp(cv_sift_descriptors, Hs_gt_sc_lin, imgs_sc_lin, "synthetic rescaling linear")

    if 'vlfeat' in detector_sets:
        run_exp(vlfeat_sift_descriptors, Hs_bark, imgs_bark, "bark")
        run_exp(vlfeat_sift_descriptors, Hs_boat, imgs_boat, "boat")
        run_exp(vlfeat_sift_descriptors, Hs_gt_rot, imgs_rot, "synthetic pi rotation")
        run_exp(vlfeat_sift_descriptors, Hs_gt_sc_lanczos, imgs_sc_lanczos, "synthetic rescaling lanczos")
        run_exp(vlfeat_sift_descriptors, Hs_gt_sc_hom, imgs_sc_hom, "synthetic rescaling homography")
        run_exp(vlfeat_sift_descriptors, Hs_gt_sc_lin, imgs_sc_lin, "synthetic rescaling linear")

    if 'lowe' in detector_sets:
        run_exp(lowe_sift_descriptors, Hs_bark, imgs_bark_lowe, "bark", imgs_bark)
        run_exp(lowe_sift_descriptors, Hs_boat, imgs_boat_lowe, "boat", imgs_boat)
        run_exp(lowe_sift_descriptors, Hs_gt_rot, imgs_rotation_lowe, "synthetic pi rotation", imgs_rot)
        run_exp(lowe_sift_descriptors, Hs_gt_sc_lanczos, imgs_scale_lowe_lanczos, "synthetic rescaling lanczos", imgs_sc_lanczos)
        run_exp(lowe_sift_descriptors, Hs_gt_sc_hom, imgs_scale_lowe_linear_cv, "synthetic rescaling linear/homography", imgs_sc_hom)
        run_exp(lowe_sift_descriptors, Hs_gt_sc_lin, imgs_scale_lowe_linear_cv, "synthetic rescaling linear", imgs_sc_lin)

    #if 'kornia' in detector_sets:
    run_exp(kornia_sift_descriptors, Hs_bark, imgs_bark, "bark")
    run_exp(kornia_sift_descriptors_single_image, Hs_bark, imgs_bark, "bark")

    run_exp(kornia_sift_descriptors, Hs_boat, imgs_boat, "boat")
    run_exp(kornia_sift_descriptors_single_image, Hs_boat, imgs_boat, "boat")

    run_exp(kornia_sift_descriptors, Hs_gt_rot, imgs_rot, "synthetic pi rotation")
    run_exp(kornia_sift_descriptors_single_image, Hs_gt_rot, imgs_rot, "synthetic pi rotation")

    run_exp(kornia_sift_descriptors, Hs_gt_sc_lanczos, imgs_sc_lanczos, "synthetic rescaling lanczos")
    run_exp(kornia_sift_descriptors_single_image, Hs_gt_sc_lanczos, imgs_sc_lanczos, "synthetic rescaling lanczos")

    run_exp(kornia_sift_descriptors, Hs_gt_sc_hom, imgs_sc_hom, "synthetic rescaling lanczos homography")
    run_exp(kornia_sift_descriptors_single_image, Hs_gt_sc_hom, imgs_sc_hom, "synthetic rescaling lanczos homography")

    run_exp(kornia_sift_descriptors, Hs_gt_sc_lin, imgs_sc_lin, "synthetic rescaling linear")
    run_exp(kornia_sift_descriptors_single_image, Hs_gt_sc_lin, imgs_sc_lin, "synthetic rescaling linear")

    # print("Original images")
    # run_exp(kornia_sift_descriptors, Hs_gt_rot, imgs_rot, "synthetic pi rotation", compensate=False)
    # print("nms compensation")
    # run_exp(kornia_sift_descriptors_nms_compensate, Hs_gt_rot[:1], imgs_rot[:2], "synthetic pi rotation", compensate=True)
    # print("Cropped images")
    # run_exp(kornia_sift_descriptors, Hs_gt_rot_cropped, imgs_rot_cropped, "synthetic pi rotation", compensate=False)


def rotate(img, sin_a, cos_a, rotation_index, show=False):
    h, w = img.shape[:2]
    center_x = (w - 1) / 2
    center_y = (h - 1) / 2

    H_gt = np.array([
        [cos_a, sin_a, 0.],  # (1 - cos_a) * center_x - sin_a * center_y
        [-sin_a, cos_a, 0.],  # (1 - cos_a) * center_y + sin_a * center_x],
        [0., 0., 1.],
    ])

    box = np.array([[0., 0., 1.], [0., h - 1, 1.], [w - 1, 0., 1.], [w - 1, h - 1, 1.]])
    box2 = (H_gt @ box.T).T
    min_x = box2[:, 0].min()
    min_y = box2[:, 1].min()

    H_gt[0, 2] = -min_x
    H_gt[1, 2] = -min_y
    box3 = (H_gt @ box.T).T

    # print(H_gt)

    bb = (w, h) if rotation_index == 2 else (h, w)
    # bb = (w, h)
    img_rot_h = cv.warpPerspective(img, H_gt, bb)
    if show:
        plt.figure()
        plt.imshow(img_rot_h)
        plt.title(f"rotated by H: {rotation_index}")
        plt.show()
        plt.close()

    img_rot_r = np.rot90(img, rotation_index, [0, 1]).copy()
    if show:
        plt.figure()
        plt.imshow(img_rot_r)
        plt.title(f"rotated plain np: {rotation_index}")
        plt.show()
        plt.close()

    assert np.all(img_rot_h == img_rot_r)
    # if not np.all(img_rot_h == img_rot_r):
    #     close = np.allclose(img_rot_h, img_rot_r)
    #     print()
    return H_gt, img_rot_h


def scale_img(img, scale, mode, show=False):

    assert mode in ['lanczos', 'linear', 'linear_homography']

    H_gt = np.array([
        [scale, 0., 0.],
        [0., scale, 0.],
        [0., 0., 1.],
    ])

    h, w = img.shape[:2]
    dsize = (round(w * scale), round(h * scale))
    img_scaled_h = cv.warpPerspective(img, H_gt, dsize, flags=cv.INTER_LANCZOS4)
    if show:
        plt.figure()
        plt.imshow(img_scaled_h)
        plt.title(f"scaled by H: {scale}")
        plt.show()
        plt.close()

    pil = Image.fromarray(img)
    pil_resized = pil.resize(size=dsize, resample=Image.LANCZOS)
    img_scaled_lanczos = np.array(pil_resized)
    img_scaled_resize = cv.resize(img, dsize=dsize, interpolation=cv.INTER_LINEAR)

    if show:
        plt.figure()
        plt.imshow(img_scaled_resize)
        plt.title(f"scaled by cv.resize np: {scale}")
        plt.show()
        plt.close()

        plt.figure()
        plt.imshow(img_scaled_resize - img_scaled_h)
        plt.title(f"diff: {scale}")
        plt.show()
        plt.close()

    # if not np.all(img_scaled_h == img_scaled_resize):
    #     close = np.allclose(img_scaled_h, img_scaled_resize)
    #     print(f"is it even close: {close}")

    if mode == 'lanczos':
        img_scaled = img_scaled_lanczos
    elif mode == 'linear':
        img_scaled = img_scaled_resize
    elif mode == 'linear_homography':
        img_scaled = img_scaled_h
    else:
        raise Exception("unexpected branch")

    return H_gt, img_scaled


def np_show(img, title=None):
    plt.figure()
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()
    plt.close()


def Hs_imgs_for_rotation(file, show=False, crop=None):

    img = Image.open(file)
    img = np.array(img)

    def modulo32(n, modulo):
        n_n = n - ((n - modulo) % 32)
        assert n_n % 32 == modulo
        return n_n

    if crop:
        h, w = img.shape[:2]
        w = modulo32(w, crop)
        h = modulo32(h, crop)
        img = img[:h, :w]

    if show:
        np_show(img, "original")

    cos_a = [0., -1., 0.]
    sin_a = [1., 0., -1.]

    rotations = 3
    Hs_gt_img = [rotate(img, sin_a[i], cos_a[i], i + 1, show) for i in range(rotations)]
    Hs_gt = [h[0] for h in Hs_gt_img]
    imgs = [img] + [h[1] for h in Hs_gt_img]
    return Hs_gt, imgs

    # h, w = img.shape[:2]
    # center_x = (w - 1) / 2
    # center_y = (h - 1) / 2
    #
    # H_gt_s = np.array([
    #     [-1., 0., 2 * center_x],
    #     [0., -1., 2 * center_y],
    #     [0., 0., 1.],
    # ])
    #
    # img_rot_h = cv.warpPerspective(img, H_gt_s, (w, h))
    # if show:
    #     plt.figure()
    #     plt.imshow(img_rot_h)
    #     plt.title(f"rotated by H: {i + 1}")
    #     plt.show()
    #     plt.close()
    #
    # img_rot_r = np.rot90(img, 2, [0, 1]).copy()
    # if show:
    #     plt.figure()
    #     plt.imshow(img_rot_r)
    #     plt.title(f"rotated plain np: {i + 1}")
    #     plt.show()
    #     plt.close()
    #
    # assert np.all(img_rot_h == img_rot_r)


def Hs_imgs_for_scaling(file, scales, mode, show=False, mod4=False):

    assert mode in ['lanczos', 'linear', 'linear_homography']

    img = Image.open(file)
    img = np.array(img)
    if mod4:
        img = img[:img.shape[0] // 4 * 4, :img.shape[1] // 4 * 4]
    if show:
        np_show(img, f"original, already cropped, shape: {img.shape}")

    h_i_tuples = [scale_img(img, scale, mode, show) for scale in scales]
    Hs_gt = [e[0] for e in h_i_tuples]
    imgs_r = [e[1] for e in h_i_tuples]
    imgs = [img] + imgs_r
    return Hs_gt, imgs


class Output:
    unformatted = ""
    formatted = ""
    latex = ""


def run_exp(detectors, Hs_gt, imgs, e_name, imgs_extra=None, compensate=False):

    print(f"running experiment: {e_name}")

    metric_names = ["MAE", "running time", "tentatives", "inliers"]

    data = [[] for _ in enumerate(metric_names)]
    data_formatted = [[] for _ in enumerate(metric_names)]

    ratio_threshold = 0.8

    ransac_th = 0.5
    ransac_conf = 0.9999
    ransac_iters = 100000

    for i_det, descriptor in enumerate(detectors):

        print(f"{i_det + 1}/{len(detectors)} detectors")
        metrics = []

        # FIXME remove me
        # if compensate:
        #     descriptor.set_rotate_gauss(0)
        #     descriptor.detector.compensate_nms = 0
        kpts_0, desc_0, time_0 = descriptor.detect_compute_measure(imgs[0], mask=None)
        # metrics.append([None, time_0] * 3 + [time_0])

        for other_i in range(1, len(imgs)):
            # print(f"other_i: {other_i}")

            # FIXME remove me
            descriptor.set_rotate_gauss(4 - other_i)
            if compensate:
                descriptor.detector.compensate_nms = (4 - other_i)
            kpts_other, desc_other, time_other = descriptor.detect_compute_measure(imgs[other_i], mask=None)
            descriptor.set_rotate_gauss(0)
            if compensate:
                descriptor.detector.compensate_nms = 0
                if descriptor.detector.compensate_nms_dim_minus_1:
                    for k in kpts_other:
                        k.pt = (k.pt[0], k.pt[1] - 1)

            time = time_0 + time_other
            src_pts, dst_pts, _, _, tentative_matches = get_tentatives(kpts_0, desc_0, kpts_other, desc_other, ratio_threshold)
            if len(src_pts) < 4:
                print(f"WARNING: less than 4 tentatives: {len(src_pts)}")
                na = "N/A"
                metrics.append([na, time, na, na])
                continue
            H_est, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,
                                                   maxIters=ransac_iters, ransacReprojThreshold=ransac_th,
                                                   confidence=ransac_conf)

            img_w = imgs[other_i].shape[0]
            img_w_1 = imgs[other_i].shape[0] - 1
            def get_mask(sum_c, eps=0.05):
                return np.logical_and(np.abs((src_pts[:, 0] + dst_pts[:, 1])[:, None] - sum_c) < eps, (np.abs((src_pts[:, 1] - dst_pts[:, 0])[:, None] - 0) < 0.01))
            mask = get_mask(img_w)
            print(f"exact w: {mask.sum()}/{len(mask)}")
            mask_1 = get_mask(img_w_1)
            print(f"exact w-1: {mask_1.sum()}/{len(mask_1)}")

            if imgs_extra:
                real_imgs = imgs_extra
            else:
                real_imgs = imgs

            H_gt = Hs_gt[other_i - 1]
            MAE = get_visible_part_mean_absolute_reprojection_error(real_imgs[0], real_imgs[other_i], H_gt, H_est, metric="L2")

            tent_count = len(src_pts)
            in_count = inlier_mask.sum()
            metrics.append([MAE, time, tent_count, in_count])

            show_matches = False
            if show_matches:
                plt.figure(figsize=(8, 8))
                info = f"tentatives: {tent_count} inliers: {in_count}, ratio: {in_count / tent_count}"
                plt.title(info)
                img = draw_matches(kpts_0, kpts_other, tentative_matches, H_est, H_gt, inlier_mask, real_imgs[0], real_imgs[other_i])
                plt.imshow(img)
                plt.show(block=False)

        detector_info = [s.strip() for s in str(descriptor).split(";")]
        if len(detector_info) < 3:
            detector_info + ([" "] * (3-len(detector_info)))

        for i_m, metric_name in enumerate(metric_names):
            metric_info = []
            metric_info_formatted = []
            if i_det == 0:
                metric_info.append(metric_name)
                metric_info_formatted.append(metric_name)
            else:
                metric_info.append(" ")
                metric_info_formatted.append(" ")
            # print(f"\nmetric: {metric_name}")
            sum = 0
            sum_formatted = 0
            for i in range(len(metrics)):
                val = metrics[i][i_m]
                if not val:
                    continue
                metric_info.append(val)
                # print(val)
                if type(val) == str:
                    metric_info_formatted.append(val)
                else:
                    val_formatted = val
                    if type(val_formatted) in [float, np.float, np.float32, np.float64]:
                        val_formatted = f"{val_formatted:.3f}"
                    metric_info_formatted.append(val_formatted)
                    if type(val_formatted) == str:
                        sum_formatted += float(val_formatted)
                    else:
                        sum_formatted += val_formatted
                    sum += val
            metric_info.append(sum)
            if type(sum_formatted) == float:
                metric_info_formatted.append(f"{sum_formatted:.3f}")
            else:
                metric_info_formatted.append(str(int(sum_formatted)))
            # print(f"{sum}")
            prepend = []
            if i_m == 0:
                prepend = detector_info
            data[i_m].append(prepend + metric_info)
            data_formatted[i_m].append(detector_info + metric_info_formatted)

    def format_data(d):
        s = ""
        for i in range(len(d[0])):
            s += "\t".join([str(d[j][i]) for j in range(len(d))]) + "\n"
        return s
    Output.unformatted += f"\n\n experiment: {e_name}\n\n"
    for d in data:
        Output.unformatted += f"\n{format_data(d)}"

    Output.formatted += f"\n\n experiment: {e_name}\n\n"
    for d in data_formatted:
        Output.formatted += f"\n{format_data(d)}"

    Output.latex += f"\n\n experiment: {e_name}\n\n"
    for d in data_formatted:
        Output.latex += f"\n{csv2latex(format_data(d))}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Homography experiment')
    parser.add_argument('--detectors', nargs='+', help="list of detector sets (choose from 'opencv', 'vlfeat', 'kornia' and 'lowe')", required=True)
    args = parser.parse_args()
    run_experiments(args.detectors)

    print("Unformatted data")
    print(Output.unformatted)
    print("Formatted data")
    print(Output.formatted)
    print("Latex data")
    print(Output.latex)
