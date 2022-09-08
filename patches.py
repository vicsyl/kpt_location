import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import PIL
import torch
import os


def log_me(s):
    print(s)


def mnn(kpts, kpts_r, scale, th):

    kpts_reproj = kpts_r / scale
    d_mat = torch.cdist(kpts, kpts_reproj)
    min0 = torch.min(d_mat, dim=0)
    min1 = torch.min(d_mat, dim=1)

    mask = min1[1][min0[1]] == torch.arange(0, min0[1].shape[0])

    mask_th = mask & (min0[0] < th)

    verify = True
    if verify:
        for i in range(min0[1].shape[0]):
            if mask_th[i]:
                assert min1[1][min0[1][i]] == i
                assert min0[0][i] < th

    kpts0 = kpts[min0[1][mask_th]]
    kpts1 = kpts_r[mask_th]
    dists = min0[0][mask_th]
    diffs = (kpts0 - kpts_reproj[mask_th]) * scale
    if verify:
        ds = torch.diag(torch.cdist(kpts0, kpts_reproj[mask_th]))
        assert torch.allclose(ds, dists)

    return kpts0, kpts1, diffs


def scale_pil(img, scale, show=False):
    """
    :param img:
    :param scale: in [0, 1]
    :return:
    """
    h, w = img.size
    gcd = gcd_euclid(w, h)

    real_scale_gcd = round(gcd * scale)
    # log_me("gcd: {}".format(gcd))
    # log_me("real_scale_gcd: {}".format(real_scale_gcd))

    w_sc = int(w / gcd * real_scale_gcd)
    h_sc = int(h / gcd * real_scale_gcd)
    img_r = img.resize((h_sc, w_sc), resample=PIL.Image.LANCZOS)
    log_me("scaled to: {}".format(img_r.size))
    if show:
        show_pil(img_r)

    real_scale = real_scale_gcd / gcd
    print("real scale: {}".format(real_scale))
    return img_r, real_scale


def show_pil(img):
    npa = np.array(img)
    plt.figure()
    plt.imshow(npa)
    plt.show()


def gcd_euclid(a, b):

    c = a % b
    if c == 0:
        return b
    else:
        return gcd_euclid(b, c)


def get_default_detector():
    detector = cv.SIFT_create()
    return detector


def detect(img_pil, patch_size, detector=get_default_detector(), show=False):

    npa = np.array(img_pil)
    h, w, c = npa.shape

    kpts = detector.detect(npa, mask=None)

    kpt_f = np.array([kp.pt for kp in kpts])

    kpt_i = np.round(kpt_f).astype(np.int)
    margin = patch_size // 2
    mask = (kpt_i[:, 0] >= margin) & (kpt_i[:, 1] >= margin)
    mask = mask & (kpt_i[:, 0] < h - margin) & (kpt_i[:, 1] < w - margin)
    kpt_f = kpt_f[mask]

    if show:
        npac = npa.copy()
        cv.drawKeypoints(npa, kpts, npac, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure()
        plt.imshow(npac)
        plt.show()

    return torch.from_numpy(kpt_f)


def get_patches(img_pil, patch_size, kpt_f, exclude_near_edge=True, show_few=False):

    assert patch_size % 2 == 1, "uncentered patches"
    if not exclude_near_edge:
        raise "Not implemented"

    img_t = torch.tensor(np.array(img_pil))
    kpt_i = torch.round(kpt_f).to(torch.int)
    margin = patch_size // 2

    patches_l = [img_t[kp_i[0] - margin: kp_i[0] + margin + 1, kp_i[1] - margin: kp_i[1] + margin + 1][None] for kp_i in kpt_i]
    patches = torch.cat(patches_l, dim=0)
    # patches = np.array([img_t[kp_i[0] - margin: kp_i[0] + margin,
    #                         kp_i[1] - margin: kp_i[1] + margin] for kp_i in kpt_i])

    if show_few:
        cols = 4
        rows = 4
        fig, axs = plt.subplots(rows, cols, figsize=(5, 5))
        fig.suptitle("Few patches")

        for ix in range(rows):
            for iy in range(cols):
                #axs[ix, iy].set_title("foo")
                axs[ix, iy].set_axis_off()
                axs[ix, iy].imshow(patches[ix * cols + iy].numpy())
        plt.show()

    return patches


def compare_patches(patches0, patches1, diffs):

    cols = 8
    cols = min(cols, patches0.shape[0])

    fig, axs = plt.subplots(2, cols, figsize=(12, 4))
    fig.suptitle("Patches comparison")

    for n in range(cols):
        for r in range(2):
            if r == 0:
                axs[r, n].set_title("diff=\n{:.02f},{:.02f}".format(diffs[n, 0].item(), diffs[n, 1].item()))
            axs[r, n].set_axis_off()
            patch = patches0[n] if r == 0 else patches1[n]
            axs[r, n].imshow(patch.numpy())
    plt.show()


def get_img_tuple(path, scale, show=False):

    img = Image.open(path)
    if show:
        show_pil(img)
    log_me("original size: {}".format(img.size))

    img_r, real_scale = scale_pil(img, scale=scale, show=show)
    return img, img_r, real_scale


def process_patches_for_file(file_path,
                             out_dir,
                             out_dict,
                             patch_size=33,
                             scale=0.3,
                             th=2.0,
                             compare=True,
                             show=False):

    print("Processing: {}".format(file_path))

    # convert and show the image
    img, img_r, real_scale = get_img_tuple(file_path, scale)

    kpts = detect(img, patch_size=patch_size, show=show)
    kpts_r = detect(img_r, patch_size=patch_size, show=show)

    # TODO check integer th (original value of 2)
    kpts, kpts_r, diffs = mnn(kpts, kpts_r, real_scale, th)

    patches = get_patches(img, patch_size, kpt_f=kpts, show_few=show)
    patches_r = get_patches(img_r, patch_size, kpt_f=kpts_r, show_few=show)

    if compare:
        compare_patches(patches, patches_r, diffs)

    file_name_prefix = file_path[file_path.rfind("/") + 1:file_path.rfind(".")]
    for i in range(patches.shape[0]):
        diff = diffs[i]
        patch = patches_r[i]
        file_name = "{}_{}.jpg".format(file_name_prefix, i)
        out_dict[file_name] = diff
        out_path = "{}/{}".format(out_dir, file_name)
        cv.imwrite(out_path, patch.numpy())


def prepare_data():

    root_dir = "./work"
    out_dir = "./dataset"

    all = len([fn for fn in os.listdir(root_dir) if fn.endswith(".tonemap.jpg")])

    data_dict = {}
    counter = 0
    for file_name in os.listdir(root_dir):
        if not file_name.endswith(".tonemap.jpg"):
            continue
        counter += 1
        path = "{}/{}".format(root_dir, file_name)
        print("{}/{}".format(counter, all))
        process_patches_for_file(file_path=path,
                                 out_dir=out_dir,
                                 out_dict=data_dict,
                                 patch_size=33,
                                 scale=0.3,
                                 th=2.0,
                                 compare=False)

    with open("{}/values.txt".format(out_dir), "w") as f:
        for k in data_dict:
            data = data_dict[k].numpy()
            f.write("{}, {}, {}\n".format(k, data[0], data[1]))


if __name__ == "__main__":
    prepare_data()
