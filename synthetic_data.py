import math
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from dataclasses import dataclass
from typing import Callable
from config import get_detector_by_key
from kornia_sift import NumpyKorniaSiftDescriptor
from sift_detectors import AdjustedSiftDescriptor

@dataclass(init=True, repr=True, eq=False, order=False, unsafe_hash=False, frozen=False)
class SyntheticConf:
    img_size: tuple
    kpt_loc: tuple
    dist_to_grey_fnc: Callable
    #name


def simple_corner_synth_builder(corner_type, radius, max_int=255.0):

    assert corner_type in ["SE", "SW", "NW", "NE"]

    def to_ret(conf):
        max_dst = max(conf.img_size[0], conf.img_size[1]) * 10
        w = np.indices(conf.img_size)
        ys, xs = w[0], w[1]
        # TODO just use conf.kpt_loc[0]
        kpt_center_y = np.ones(conf.img_size) * conf.kpt_loc[0]
        kpt_center_x = np.ones(conf.img_size) * conf.kpt_loc[1]
        distances_y = (np.abs(ys - kpt_center_y) > radius).astype(dtype=int)
        fact_distances_x = (xs <= conf.kpt_loc[1] - radius if corner_type[1] == "E" else xs >= conf.kpt_loc[1] - radius).astype(int) * max_dst
        distances_y = distances_y + fact_distances_x
        distances_x = (np.abs(xs - kpt_center_x) > radius).astype(dtype=int)
        fact_distances_y = (ys <= conf.kpt_loc[0] - radius if corner_type[0] == "S" else ys >= conf.kpt_loc[0] + radius).astype(int) * max_dst
        distances_x = distances_x + fact_distances_y
        return (np.minimum(distances_y, distances_x) > 1).astype(dtype=np.uint8) * int(max_int)

    return to_ret


def corner_synth_builder(corner_type, inner_fce, radius):

    assert corner_type in ["SE", "SW", "NW", "NE"]

    def to_ret(conf):
        max_dst = max(conf.img_size[0], conf.img_size[1]) * 10
        w = np.indices(conf.img_size)
        ys, xs = w[0], w[1]
        kpt_center_y = np.ones(conf.img_size) * conf.kpt_loc[0]
        kpt_center_x = np.ones(conf.img_size) * conf.kpt_loc[1]
        distances_y = np.abs(ys - kpt_center_y)
        fact_distances_x = (xs <= conf.kpt_loc[1] - radius if corner_type[1] == "E" else xs >= conf.kpt_loc[1] - radius).astype(int) * max_dst
        distances_y = distances_y + fact_distances_x
        distances_x = np.abs(xs - kpt_center_x)
        fact_distances_y = (ys <= conf.kpt_loc[0] - radius if corner_type[0] == "S" else ys >= conf.kpt_loc[0] + radius).astype(int) * max_dst
        distances_x = distances_x + fact_distances_y
        distances = np.minimum(distances_y, distances_x)

        # slices = [conf.kpt_loc[0] - radius,
        #           conf.kpt_loc[0] + radius,
        #           conf.kpt_loc[1] - radius,
        #           conf.kpt_loc[1] + radius]
        # slices = [int(s) for s in slices]
        # np_max = np.maximum(distances_y, distances_x)[slices[0]: slices[1], slices[2]: slices[3]]
        # distances[slices[0]: slices[1], slices[2]: slices[3]] = np_max

        img = inner_fce(distances)
        img = img.astype(dtype=np.uint8)
        return img

    return to_ret


def synth_from_distance(inner_fce):

    def to_ret(conf):
        w = np.indices(conf.img_size)
        ys, xs = w[0], w[1]
        kpt_center_y = np.ones(conf.img_size) * conf.kpt_loc[0]
        kpt_center_x = np.ones(conf.img_size) * conf.kpt_loc[1]
        distances = np.sqrt((ys - kpt_center_y) ** 2 + (xs - kpt_center_x) ** 2)
        img = inner_fce(distances)
        img = img.astype(dtype=np.uint8)
        return img

    return to_ret


def pyramid_distance_builder(radius=10.0, max=255.0):

    def pyramid_distance_numpy(distances):
        return max * (1 - np.minimum(distances / radius, 1.0))

    return pyramid_distance_numpy


def hyperbolic_spike_builder(eps=0.1, max=255.0):

    def hyperbolic_spike_numpy(distances):
        return 1 / (distances + eps)

    def hyperbolic_spike(distances):
        factor = 1.0
        if max:
            max_0 = hyperbolic_spike_numpy(np.array([0.0]))[0]
            factor = max / max_0
        return hyperbolic_spike_numpy(distances) * factor

    return synth_from_distance(hyperbolic_spike)


def gauss_dist_to_grey_builder(sigma, max=255.0):

    def gauss_numpy(distances):
        return np.exp((distances / sigma) ** 2 / -2) / (sigma * math.sqrt(2 * math.pi))

    def gauss_dist_to_grey(distances):
        factor = 1.0
        if max:
            max_0 = gauss_numpy(np.array([0.0]))[0]
            factor = max / max_0
        return gauss_numpy(distances) * factor

    return synth_from_distance(gauss_dist_to_grey)


def get_synthetic_image(conf: SyntheticConf):

    return conf.dist_to_grey_fnc(conf)

    # w = np.indices(conf.img_size)
    # ys, xs = w[0], w[1]
    # kpt_center_y = np.ones(conf.img_size) * conf.kpt_loc[0]
    # kpt_center_x = np.ones(conf.img_size) * conf.kpt_loc[1]
    # distances = np.sqrt((ys - kpt_center_y) ** 2 + (xs - kpt_center_x) ** 2)
    # img = conf.dist_to_grey_fnc(distances)
    # img = img.astype(dtype=np.uint8)
    # return img


def get_and_show(conf):
    img = get_synthetic_image(conf)
    plt.figure()
    plt.imshow(img)
    plt.show()
    plt.close()


def analyze_kpts(kpts, img, conf, detector_name):

    img_draw = img.copy()
    img_draw = np.repeat(img_draw[:, :, None], 3, axis=2)
    cv.drawKeypoints(img_draw, kpts, img_draw, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure()
    plt.title("{}: {} kpts with kpts".format(detector_name, len(kpts)))
    plt.imshow(img_draw)
    plt.show()
    plt.close()

    if len(kpts) == 0:
        return None

    kpt_locs = np.array([[kpt.pt[1], kpt.pt[0]] for kpt in kpts])
    kpt_scales = np.array([kpt.size for kpt in kpts])
    dists = np.linalg.norm(kpt_locs - conf.kpt_loc, axis=1)
    sorted_indices = np.argsort(dists)
    dists = dists[sorted_indices]
    print("dists: {}".format(dists))
    kpt_locs = kpt_locs[sorted_indices]
    print("kpts locations: {}".format(kpt_locs))
    kpt_scales = kpt_scales[sorted_indices]
    print("kpts scales: {}".format(kpt_scales))

    return kpts[sorted_indices[0]]


def main():
    img_size = (101, 101)
    kpt_loc = (50.0, 50.0)
    dist_to_grey_fncs = [
        simple_corner_synth_builder("SE", radius=2),
        corner_synth_builder("SE", pyramid_distance_builder(radius=2.5), radius=2.5),

        # gauss_dist_to_grey_builder(sigma=0.1),
        # gauss_dist_to_grey_builder(sigma=0.2),
        #
        # gauss_dist_to_grey_builder(sigma=0.5),
        # gauss_dist_to_grey_builder(sigma=1),

        gauss_dist_to_grey_builder(sigma=3.0),

        # gauss_dist_to_grey_builder(sigma=4),
        # gauss_dist_to_grey_builder(sigma=6),
        # gauss_dist_to_grey_builder(sigma=8),
        # hyperbolic_spike_builder(eps=0.1),
        # hyperbolic_spike_builder(eps=1.0),
        # hyperbolic_spike_builder(eps=5.0),

        # hyperbolic_spike_builder(eps=0.5),
        # hyperbolic_spike_builder(eps=0.2),
        # hyperbolic_spike_builder(eps=0.1),
    ]
    confs = [SyntheticConf(
        img_size=img_size,
        kpt_loc=kpt_loc,
        dist_to_grey_fnc=f
    ) for f in dist_to_grey_fncs]

    detectors = [AdjustedSiftDescriptor(adjustment=[0., 0.]),
                 NumpyKorniaSiftDescriptor(num_features=10)]

    errors = []

    for shift_index in range(0, 1):
        shift = shift_index * 5
        for conf in confs:
            for detector in detectors:

                if type(detector) == "str":
                    detector = get_detector_by_key(detector)

                conf.kpt_loc = (kpt_loc[0] + shift, kpt_loc[1] + shift)
                img = get_synthetic_image(conf)

                kpts = detector.detect(img, None)

                print(f"detector: {detector}")
                print(f"conf: {conf}")
                best_kpt = analyze_kpts(kpts, img, conf, detector)

                print("kpts: {}".format(len(kpts)))
                if not best_kpt:
                    print("NO KPT DETECTED")
                    continue
                kpt_loc_det = np.array([best_kpt.pt[1], best_kpt.pt[0]])
                print("kpt loc as detected: {}".format(kpt_loc_det))
                kpt_loc_gt = np.array(list(conf.kpt_loc))
                print("kpt loc gt: {}".format(kpt_loc_gt))
                err = kpt_loc_det - kpt_loc_gt
                print("err: {}".format(err))
                errors.append(err)

    print("All errors: {}".format(errors))


if __name__ == "__main__":
    main()
