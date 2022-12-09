import os
import numpy as np
import torch

from PIL import Image

import torch.nn.functional as F

from kornia.filters import gaussian_blur2d
import kornia.utils as KU
import matplotlib.pyplot as plt


def prepare(title, rows=2, cols=2):
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    fig.tight_layout()
    fig.suptitle(title)
    return axs


def add(axs, r, c, img, title):
    img = img[0, 0]
    axis = axs[r, c]
    axis.set_title(title)
    axis.imshow(img)


def get_kernel_size(sigma: float):
    ksize = int(2.0 * 4.0 * sigma + 1.0)

    if ksize % 2 == 0:
        ksize += 1
    return ksize


def kornia_gaussian_blur2d(x, sigma, separable=True):
    ksize = get_kernel_size(sigma)
    return gaussian_blur2d(x, (ksize, ksize), (sigma, sigma), separable=separable)


def check_mod(n):
    for i in range(5):
        assert n % 2 == 1
        n //= 2


def check_pil_mod(img_pil):
    h, w = img_pil.size[:2]
    check_mod(h)
    check_mod(w)


def check_np_mod(img_np):
    h, w = img_np.shape[:2]
    check_mod(h)
    check_mod(w)


def modulo32(n):
    n_n = n - ((n + 1) % 32)
    assert n_n % 32 == 31
    return n_n


def potentially_crop(img_np, crop):
    if crop:
        h, w = img_np.shape[:2]
        h = modulo32(h)
        w = modulo32(w)
        img_np = img_np[:h, :w]
        check_np_mod(img_np)
    return img_np


def get_image_grid():

    #size = 767, 1023

    size = 32
    x = np.arange(0, size, step=1, dtype=int)
    y = np.arange(0, size, step=1, dtype=int)
    xv, yv = np.meshgrid(x, y)

    img_np = np.zeros((size, size, 3), dtype=int)
    img_np[:, :, 0] = yv
    img_np[:, :, 1] = xv
    ret = potentially_crop(img_np, Cfg.crop)
    return ret


def get_image_path(file_path, crop):
    img_pil = Image.open(file_path)
    img_np_o = np.array(img_pil)
    potentially_crop(img_np_o, crop)
    return img_np_o


def rotate_experiment_interpolation(img_np_o, rotations_90_deg):

    img_t_o = KU.image_to_tensor(img_np_o.copy(), False).float() / 255.


    img_t_r = torch.rot90(img_t_o, k=rotations_90_deg, dims=(2, 3))

    axs = prepare(f"interpolation rotational symmetry - crop: {Cfg.crop}", rows=2, cols=2)
    add(axs, 0, 0, img_t_o, "original")

    img_t_o_int_2 = F.interpolate(img_t_o, scale_factor=2.0, mode='bilinear', align_corners=False)
    img_t_r_int_2 = F.interpolate(img_t_r, scale_factor=2.0, mode='bilinear', align_corners=False)
    img_t_b_int_2 = torch.rot90(img_t_r_int_2, k=4 - rotations_90_deg, dims=(2, 3))
    img_t_d_int_2 = img_t_o_int_2 - img_t_b_int_2
    add(axs, 0, 1, img_t_d_int_2, "diff. interpolate bilinear scale = 2.0")

    img_t_o_int_size_nearest = F.interpolate(img_t_o, size=(img_t_o.size(-2) // 2, img_t_o.size(-1) // 2),
                                             mode='nearest')
    img_t_r_int_size_nearest = F.interpolate(img_t_r, size=(img_t_r.size(-2) // 2, img_t_r.size(-1) // 2),
                                             mode='nearest')

    img_t_b_int_size_nearest = torch.rot90(img_t_r_int_size_nearest, k=4 - rotations_90_deg, dims=(2, 3))
    img_t_d_int_size_nearest = (img_t_o_int_size_nearest - img_t_b_int_size_nearest).abs()
    add(axs, 1, 0, img_t_d_int_size_nearest, "diff. interpolate nearest scale = 0.5")

    img_t_o_int_size_e2nd = img_t_o[:, :, ::2, ::2]
    img_t_r_int_size_e2nd = img_t_r[:, :, ::2, ::2]

    img_t_b_int_size_e2nd = torch.rot90(img_t_r_int_size_e2nd, k=4 - rotations_90_deg, dims=(2, 3))
    img_t_d_int_size_e2nd = (img_t_o_int_size_e2nd - img_t_b_int_size_e2nd).abs()
    add(axs, 1, 1, img_t_d_int_size_e2nd, "diff. interpolate ::2  (nearest, scale = 0.5)")
    print(f"exp. max: {img_t_d_int_size_e2nd.max()}")

    #
    # img_t_o_int_size_bilinear = F.interpolate(img_t_o, size=(img_t_o.size(-2) // 2, img_t_o.size(-1) // 2),
    #                                           mode='bilinear')
    # img_t_r_int_size_bilinear = F.interpolate(img_t_r, size=(img_t_r.size(-2) // 2, img_t_r.size(-1) // 2),
    #                                           mode='bilinear')
    # img_t_b_int_size_bilinear = torch.rot90(img_t_r_int_size_bilinear, k=4 - rotations_90_deg, dims=(2, 3))
    # img_t_d_int_size_bilinear = img_t_o_int_size_bilinear - img_t_b_int_size_bilinear
    # add(axs, 1, 1, img_t_d_int_size_bilinear, "diff. interpolate bilinear scale = 0.5")

    plt.show()
    pass


class Cfg:
    crop = True


def rotate_experiment_loop_gauss(img_to_show):
    Cfg.crop = True
    img_dir = "demo_imgs/hypersim"
    files = ["{}/{}".format(img_dir, fn) for fn in os.listdir(img_dir)][:img_to_show]
    for file_path in files:
        print(f"\n\nFILE: {file_path}\n")
        for rots in range(1, 4):
            print(f"rotation by {rots * 90} degrees")
            # img_np = get_image_path(file_path, crop)
            img_np = get_image_grid()
            rotate_experiment_interpolation(img_np, rots)


if __name__ == "__main__":
    rotate_experiment_loop_gauss(img_to_show=1)
