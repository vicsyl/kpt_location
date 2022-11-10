import math

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from kornia_sift import NumpyKorniaSiftDescriptor
from scale_pyramid import MyScalePyramid
from sift_detectors import AdjustedSiftDescriptor
from superpoint_local import SuperPointDetector
from utils import get_tentatives

from hloc_sift import HlocSiftDescriptor


def decompose_homographies(Hs):
    """
    :param Hs:(B, 3, 3)
    :param device:
    :return: pure_homographies(B, 3, 3), affine(B, 3, 3)
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    B, three1, three2 = Hs.shape
    assert three1 == 3
    assert three2 == 3

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


def get_descriptor_by_key(key):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    key = key.lower()
    if key == 'sift':
        return cv.SIFT_create()
    elif key == 'adjusted_sift':
        return AdjustedSiftDescriptor(adjustment=[0.25, 0.25], nfeatures=8000)
    elif key == 'adjusted_sift_negative':
        return AdjustedSiftDescriptor(adjustment=[-0.25, -0.25])
    elif key == 'adjusted_sift_linear':
        return AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.11, -0.11])
    elif key == 'sift_kornia':
        num_features = 8000
        return NumpyKorniaSiftDescriptor(num_features=num_features)
    elif key == 'sift_kornia_negative':
        num_features = 8000
        return NumpyKorniaSiftDescriptor(num_features=num_features, adjustment=[-0.25, -0.25])
    elif key == 'adjusted_sift_kornia_negative':
        custom_scale_pyramid = MyScalePyramid(3, 1.6, 32, double_image=True)
        num_features = 8000
        return NumpyKorniaSiftDescriptor(num_features=num_features, nearest=False, adjustment=[-0.25, -0.25])
    elif key == 'adjusted_sift_kornia_double_negative':
        custom_scale_pyramid = MyScalePyramid(3, 1.6, 32, double_image=True)
        num_features = 8000
        return NumpyKorniaSiftDescriptor(num_features=num_features, nearest=False, adjustment=[-0.5, -0.5])
    elif key == 'adjusted_sift_kornia':
        custom_scale_pyramid = MyScalePyramid(3, 1.6, 32, double_image=True)
        num_features = 8000
        return NumpyKorniaSiftDescriptor(num_features=num_features, nearest=False)
    elif key == 'superpoint':
        return SuperPointDetector(device=device)
    elif key == 'adjusted_superpoint':
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
        return SuperPointDetector(device=device, const_adjustment=const_adjustment, translations=translations,
                                  rotations=rotations)
    else:
        raise "unrecognized detector: {}".format(key)


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


def read_imgs(file_paths, show=False):
    imgs = []
    for i, file in enumerate(file_paths):
        img = Image.open(file)
        img = np.array(img)
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

        pure_homography, affine = decompose_homographies(torch.from_numpy(H_gt[None]))

        affine = affine[0].numpy()
        # print(f"affine: {affine}")

        det = np.linalg.det(affine)
        scale = math.sqrt(det)
        affine = affine / scale

        cos_avg = (affine[0, 0] + affine[1, 1]) / 2.0
        sin_avg = (affine[0, 1] - affine[1, 0]) / 2.0
        alpha = math.atan2(sin_avg, cos_avg) * 180 / math.pi
        pure_homography = pure_homography[0].numpy()
        norm = np.linalg.norm(pure_homography[2, :2])
        print(f"{scale:.3f}\t{alpha:.3f}")


def main():

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
    files_bark = [f"bark/img{i + 1}.ppm" for i in range(6)]
    imgs_bark = read_imgs(files_bark, show=False)

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
    files_boat = [f"boat/img{i + 1}.pgm" for i in range(6)]
    imgs_boat = read_imgs(files_boat, show=False)

    print("BOAT experiment hompographies decomposition")
    print_Hs_decomposition(Hs_boat)

    hloc_sif_descriptors = [
        HlocSiftDescriptor(),
        HlocSiftDescriptor([-0.25, -0.25]),
        HlocSiftDescriptor([0.25, 0.25]),
        HlocSiftDescriptor([-0.5, -0.5]),
    ]

    num_features = 8000
    kornia_sift_descriptors = [
        NumpyKorniaSiftDescriptor(num_features=num_features),
        NumpyKorniaSiftDescriptor(num_features=num_features, nearest=False),
        NumpyKorniaSiftDescriptor(num_features=num_features, adjustment=[-0.25, -0.25]),
        NumpyKorniaSiftDescriptor(num_features=num_features, nearest=False, adjustment=[-0.25, -0.25]),
        NumpyKorniaSiftDescriptor(num_features=num_features, nearest=False, adjustment=[-0.5, -0.5]),
        NumpyKorniaSiftDescriptor(num_features=num_features, adjustment=[0.25, 0.25]),
        NumpyKorniaSiftDescriptor(num_features=num_features, nearest=False, adjustment=[0.25, 0.25]),
        NumpyKorniaSiftDescriptor(num_features=num_features, nearest=False, adjustment=[0.4, 0.4]),
        NumpyKorniaSiftDescriptor(num_features=num_features, nearest=True, adjustment=[0.4, 0.4]),
        NumpyKorniaSiftDescriptor(num_features=num_features, nearest=False, adjustment=[0.5, 0.5]),
    ]

    # !!!
    cv_sift_descriptors = [
            "SIFT",
            AdjustedSiftDescriptor(adjustment=[0.25, 0.25]),
            AdjustedSiftDescriptor(adjustment=[-0.25, -0.25]),
            AdjustedSiftDescriptor(adjustment=[-0.5, -0.5], q_adjustment=[0.09, 0.09]),
            AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.11, -0.11]),
            AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.2, -0.2]),
            AdjustedSiftDescriptor(adjustment=[0.25, 0.25], q_adjustment=[-0.25, -0.25]),
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

    for other in cv_sift_descriptors:
        run_exp(other, Hs_bark, imgs_bark, "bark")
    for other in kornia_sift_descriptors:
        run_exp(other, Hs_bark, imgs_bark, "bark")
    # for other in hloc_sif_descriptors:
    #     run_exp(other, Hs_bark, imgs_bark, "bark")

    for other in cv_sift_descriptors:
        run_exp(other, Hs_boat, imgs_boat, "boat")
    for other in kornia_sift_descriptors:
        run_exp(other, Hs_boat, imgs_boat, "boat")
    # for other in hloc_sif_descriptors:
    #     run_exp(other, Hs_boat, imgs_boat, "boat")

    Hs_gt, imgs = Hs_imgs_for_rotation(files_bark[0], show=False)
    for other in cv_sift_descriptors:
        run_exp(other, Hs_gt, imgs, "synthetic pi rotation")
    for other in kornia_sift_descriptors:
        run_exp(other, Hs_gt, imgs, "synthetic pi rotation")
    # for other in hloc_sif_descriptors:
    #     run_exp(other, Hs_gt, imgs, "synthetic pi rotation")

    scales = [scale_int / 10 for scale_int in range(1, 10)]
    Hs_gt, imgs = Hs_imgs_for_scaling(files_bark[0], scales, show=False)
    for other in cv_sift_descriptors:
        run_exp(other, Hs_gt, imgs, "synthetic pi rotation")
    for other in kornia_sift_descriptors:
        run_exp(other, Hs_gt, imgs, "synthetic rescaling")
    # for other in hloc_sif_descriptors:
    #     run_exp(other, Hs_gt, imgs, "synthetic rescaling")

    bark_img = Image.open(files_bark[0])
    bark_img = np.array(bark_img)
    boat_img = Image.open(files_boat[0])
    boat_img = np.array(boat_img)

    detectors_cv = ["sift", "adjusted_sift", "adjusted_sift_negative", "adjusted_sift_linear"]
    run_identity_exp(detectors_cv, bark_img, "identity sift cv vanilla vs other variants on bark")
    run_identity_exp(detectors_cv, boat_img, "identity sift cv vanilla vs other variants on boat")

    detectors_kornia = ["sift_kornia", "adjusted_sift_kornia", "sift_kornia_negative",
                      "adjusted_sift_kornia_negative", "adjusted_sift_kornia_double_negative"]
    run_identity_exp(detectors_kornia, bark_img, "identity sift kornia vanilla vs other variants on bark")
    run_identity_exp(detectors_kornia, boat_img, "identity sift kornia vanilla vs other variants on boat")


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


def scale_img(img, scale, show=False):

    H_gt = np.array([
        [scale, 0., 0.],
        [0., scale, 0.],
        [0., 0., 1.],
    ])

    h, w = img.shape[:2]
    img_scaled_h = cv.warpPerspective(img, H_gt, (round(w * scale), round(h * scale)), flags=cv.INTER_LINEAR)
    if show:
        plt.figure()
        plt.imshow(img_scaled_h)
        plt.title(f"scaled by H: {scale}")
        plt.show()
        plt.close()

    img_scaled_resize = cv.resize(img, dsize=(round(w * scale), round(h * scale)), interpolation=cv.INTER_LINEAR)
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

    # assert np.all(img_scaled_h == img_scaled_resize)
    # if not np.all(img_rot_h == img_rot_r):
    #     close = np.allclose(img_rot_h, img_rot_r)
    #     print()
    return H_gt, img_scaled_resize


def np_show(img, title=None):
    plt.figure()
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()
    plt.close()


def Hs_imgs_for_rotation(file, show=False):

    img = Image.open(file)
    img = np.array(img)
    if show:
        np_show(img, "original")

    cos_a = [0., -1., 0.]
    sin_a = [1., 0., -1.]

    Hs_gt_img = [rotate(img, sin_a[i], cos_a[i], i + 1, show) for i in range(3)]
    Hs_gt = [h[0] for h in Hs_gt_img]
    imgs = [img] + [h[1] for h in Hs_gt_img]
    return Hs_gt, imgs

    # h, w = img.shape[:2]
    # center_x = (w - 1) / 2
    # center_y = (h - 1) / 2
    #
    # H_gt = np.array([
    #     [-1., 0., 2 * center_x],
    #     [0., -1., 2 * center_y],
    #     [0., 0., 1.],
    # ])
    #
    # img_rot_h = cv.warpPerspective(img, H_gt, (w, h))
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
    #
    # print()


def Hs_imgs_for_scaling(file, scales, show=False):

    img = Image.open(file)
    img = np.array(img)
    img = img[:img.shape[0] // 4 * 4, :img.shape[1] // 4 * 4]
    if show:
        np_show(img, f"original, already cropped, shape: {img.shape}")

    h_i_tuples = [scale_img(img, scale, show) for scale in scales]
    Hs_gt = [e[0] for e in h_i_tuples]
    imgs_r = [e[1] for e in h_i_tuples]
    imgs = [img] + imgs_r
    return Hs_gt, imgs


def get_detector_special(det, prefix=""):
    if type(det) == str:
        detector = get_descriptor_by_key(det)
        detector_name = det
    else:
        detector = det
        detector_name = str(detector)
    print(f"{prefix}{detector_name}")
    return detector


def run_identity_exp(detector_names, img, name):

    print(f"\n\nexperiment: {name}")

    ratio_threshold = 0.8
    ransac_th = 0.5
    ransac_conf = 0.9999
    ransac_iters = 100000

    detector = get_detector_special(detector_names[0], prefix="base detector:")

    kpts_0, desc_0 = detector.detectAndCompute(img, mask=None)
    print(f"base line kpts: {len(kpts_0)}")
    # angle, octave, response, size, pt, class_id
    # metric_names = ["MAE", "tentatives", "inliers"]
    # metrics = []

    for other_i in range(1, len(detector_names)):

        H_est = None
        cur_ratio_threshold = ratio_threshold
        while H_est is None:
            detector = get_detector_special(detector_names[other_i])

            # print(f"other_i: {other_i}")
            kpts_other, desc_other = detector.detectAndCompute(img, mask=None)
            print(f"kpts: {len(kpts_other)}")
            src_pts, dst_pts, kpts_00, kpts_11, tentative_matches = get_tentatives(kpts_0, desc_0, kpts_other, desc_other, cur_ratio_threshold)
            if len(tentative_matches) < 4:
                print(f"WARNING: couldn't find at least 4 tentatives with ratio threshold of {cur_ratio_threshold}")
                cur_ratio_threshold = math.sqrt(cur_ratio_threshold)
                if cur_ratio_threshold < 0.99:
                    print(f"will continue with threshold={cur_ratio_threshold}")
                    continue
                else:
                    print(f"experiment unsuccessful, quitting ...\n")
                    break
            H_est, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,
                                                   maxIters=ransac_iters, ransacReprojThreshold=ransac_th,
                                                   confidence=ransac_conf)

            def print_stats(kpts_000, kpts_111):
                sizes_0 = [d.size for d in kpts_000]
                sizes_1 = [d.size for d in kpts_111]
                size_ratio = np.array([sizes_0[i]/sizes_1[i] for i, _ in enumerate(sizes_0)])
                print(f"size ratio mean and std_dev: {size_ratio.mean()}, {np.sqrt(size_ratio.var())}")

                # class id = -1
                # response = 0.0

                ang_0 = [d.angle for d in kpts_000]
                ang_1 = [d.angle for d in kpts_111]
                and_diff = np.array([ang_0[i] - ang_1[i] for i, _ in enumerate(ang_0)])
                print(f"angular difference mean and std_dev: {and_diff.mean()}, {np.sqrt(and_diff.var())}")

                oct_0 = [d.octave for d in kpts_000]
                oct_1 = [d.octave for d in kpts_111]
                oct_diff = np.array([oct_0[i] - oct_1[i] for i, _ in enumerate(oct_0)])
                print(f"octave difference mean and std_dev: {oct_diff.mean()}, {np.sqrt(oct_diff.var())}")
                un = np.unique(oct_0, return_counts=True)
                print(f"octaves: {un[0]}, {un[1]}")

                pt_0 = np.float32([k.pt for k in kpts_000]).reshape(-1, 2)
                pt_1 = np.float32([k.pt for k in kpts_111]).reshape(-1, 2)
                loc_diff = (pt_0 - pt_1)
                print(f"location difference mean and std_dev: {loc_diff.mean(axis=0)}, {np.sqrt(loc_diff.var(axis=0))}")

            print("stats for all tentatives: ")
            print_stats(kpts_00, kpts_11)

            sizes_00 = np.array([k.size for k in kpts_00])
            hist = np.histogram(sizes_00)
            print()
            print(f"histogram[0]: {hist[0]}")
            print(f"histogram[1]: {hist[1]}")

            for i in range(1, len(hist[1])):
                lower = hist[1][i-1]
                upper = hist[1][i]
                if i != len(hist[1]) - 1:
                    mask = np.logical_and(lower <= sizes_00, sizes_00 < upper)
                else:
                    mask = np.logical_and(lower <= sizes_00, sizes_00 <= upper)
                count = hist[0][i-1]
                if count == 0:
                    print(f"\nno kpts between sizes [{lower}, {upper}] ({count} kpts.)")
                    continue
                print(f"\nstats for kpts between sizes [{lower}, {upper}] ({count} kpts.)")
                kpts_000 = [kpts_00[i] for i, m in enumerate(mask) if m]
                kpts_111 = [kpts_11[i] for i, m in enumerate(mask) if m]
                print_stats(kpts_000, kpts_111)

            MAE = get_visible_part_mean_absolute_reprojection_error(img, img, np.eye(3), H_est, metric="vec")
            tent_count = len(src_pts)
            in_count = inlier_mask.sum()
            print(f"MAE(vec): {MAE}")
            print(f"tent_count: {tent_count}")
            print(f"in_count: {in_count}")


def run_exp(detector_name, Hs_gt, imgs, name):
    print(f"\n\nexperiment: {name}")
    detector = get_detector_special(detector_name)
    ratio_threshold = 0.8

    ransac_th = 0.5
    ransac_conf = 0.9999
    ransac_iters = 100000

    kpts_0, desc_0 = detector.detectAndCompute(imgs[0], mask=None)

    metric_names = ["MAE", "tentatives", "inliers"]
    metrics = []

    for other_i in range(1, len(imgs)):
        # print(f"other_i: {other_i}")
        kpts_other, desc_other = detector.detectAndCompute(imgs[other_i], mask=None)
        src_pts, dst_pts, _, _, tentative_matches = get_tentatives(kpts_0, desc_0, kpts_other, desc_other, ratio_threshold)
        if len(src_pts) < 4:
            print(f"WARNING: less than 4 tentatives: {len(src_pts)}")
            metrics.append(["N/A"] * 3)
            continue
        H_est, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,
                                               maxIters=ransac_iters, ransacReprojThreshold=ransac_th,
                                               confidence=ransac_conf)

        H_gt = Hs_gt[other_i - 1]
        MAE = get_visible_part_mean_absolute_reprojection_error(imgs[0], imgs[other_i], H_gt, H_est, metric="L2")

        tent_count = len(src_pts)
        in_count = inlier_mask.sum()
        metrics.append([MAE, tent_count, in_count])

        show_matches = True
        if show_matches:
            plt.figure(figsize=(8, 8))
            info = f"tentatives: {tent_count} inliers: {in_count}, ratio: {in_count / tent_count}"
            plt.title(info)
            img = draw_matches(kpts_0, kpts_other, tentative_matches, H_est, H_gt, inlier_mask, imgs[0], imgs[other_i])
            plt.imshow(img)
            plt.show(block=False)

    for i_m, metric_name in enumerate(metric_names):
        print(f"\nmetric: {metric_name}")
        sum = 0
        for i in range(len(metrics)):
            val = metrics[i][i_m]
            print(val)
            if type(val) == str:
                pass
            else:
                sum += val
        print(f"{sum}")


if __name__ == "__main__":
    main()
