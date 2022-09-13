import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import PIL
import torch
import os


def log_me(s):
    print(s)


def mnn(kpts, kpts_scales, kpts_r, kpts_r_scales, scale, th):

    # TODO skip if len(kpts) == 0

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
    kpts_scales = kpts_scales[min0[1][mask_th]]

    kpts1 = kpts_r[mask_th]
    kpts_r_scales = kpts_r_scales[mask_th]

    dists = min0[0][mask_th]
    diffs = (kpts0 - kpts_reproj[mask_th]) * scale
    if verify:
        ds = torch.diag(torch.cdist(kpts0, kpts_reproj[mask_th]))
        assert torch.allclose(ds, dists)

    return kpts0, kpts_scales, kpts1, kpts_r_scales, diffs


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
    img_r = img.resize((h_sc, w_sc), resample=PIL.Image.Resampling.LANCZOS)
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


def detect(img_pil, scale_th, detector=get_default_detector(), show=False):

    # TODO remove patch size

    npa = np.array(img_pil)
    h, w, c = npa.shape

    kpts = detector.detect(npa, mask=None)

    kpt_f = np.array([[kp.pt[1], kp.pt[0]] for kp in kpts])
    kpt_i = np.round(kpt_f).astype(int)

    scales = np.array([kp.size for kp in kpts])
    # TODO fixme - not consistent
    margin = np.ceil(scale_th).astype(int) + 3
    #margin = patch_size // 2

    mask = scales > scale_th
    mask = mask & (kpt_i[:, 0] >= margin) & (kpt_i[:, 1] >= margin)
    mask = mask & (kpt_i[:, 0] < h - margin) & (kpt_i[:, 1] < w - margin)
    kpt_f = kpt_f[mask]
    scales = scales[mask]

    if show:
        npac = npa.copy()
        cv.drawKeypoints(npa, kpts, npac, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure()
        plt.imshow(npac)
        plt.show()

    return torch.from_numpy(kpt_f), torch.from_numpy(scales)


def get_patches(img_pil, kpt_f, kpt_scales, scale_th, show_few=False):

    #assert scale_th % 2 == 1, "uncentered patches"

    img_n = np.array(img_pil)
    img_t = torch.tensor(img_n)

    kpt_i = torch.round(kpt_f).to(torch.int)
    #margins = torch.tensor(np.ceil(kpt_scales / 2.0))
    # margins_np = margins.to(torch.int).numpy()
    margins_np = np.ones(kpt_scales.shape[0]) * scale_th
    margins_np = np.ceil(margins_np).astype(int) // 2

    print()
    for i, kp_i in list(enumerate(kpt_i)):
        print(kp_i[0] - margins_np[i],
              kp_i[0] + margins_np[i] + 1,
              kp_i[1] - margins_np[i],
              kp_i[1] + margins_np[i] + 1,
              img_t.shape)

    patches_l = [img_t[kp_i[0] - margins_np[i]: kp_i[0] + margins_np[i] + 1, kp_i[1] - margins_np[i]: kp_i[1] + margins_np[i] + 1][None] for i, kp_i in enumerate(kpt_i)]
    patches_l = [patch[0] for patch in patches_l]

    if show_few:
        cols = 5
        rows = 5

        detector = get_default_detector()
        kpts = detector.detect(img_n, mask=None)
        npac = img_n.copy()
        cv.drawKeypoints(img_n, kpts, npac, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        patches_o = [npac[kp_i[0] - margins_np[i]: kp_i[0] + margins_np[i] + 1,
                     kp_i[1] - margins_np[i]: kp_i[1] + margins_np[i] + 1][None] for i, kp_i in enumerate(kpt_i)]
        patches_o = [p[0] for p in patches_o]

        fig, axs = plt.subplots(rows, cols, figsize=(5, 5))
        fig.suptitle("Few patches - original")

        for ix in range(rows):
            for iy in range(cols):
                if ix * cols + iy >= len(patches_l):
                    break
                #axs[ix, iy].set_title("foo")
                axs[ix, iy].set_axis_off()
                patch_to_show = patches_o[ix * cols + iy].copy()
                axs[ix, iy].imshow(patch_to_show)
        plt.show()
        plt.close()

        fig, axs = plt.subplots(rows, cols, figsize=(5, 5))
        fig.suptitle("Few patches - redetected")

        for ix in range(rows):
            for iy in range(cols):
                if ix * cols + iy >= len(patches_l):
                    break
                #axs[ix, iy].set_title("foo")
                axs[ix, iy].set_axis_off()
                patch_to_show = patches_l[ix * cols + iy].numpy().copy()
                kpts = detector.detect(patch_to_show, mask=None)
                cv.drawKeypoints(patch_to_show, kpts, patch_to_show, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                axs[ix, iy].imshow(patch_to_show)
        plt.show()
        plt.close()

    return patches_l


def compare_patches(patches0, patches1, diffs):

    cols = 8
    cols = min(cols, len(patches0))

    fig, axs = plt.subplots(2, cols, figsize=(12, 4))
    fig.suptitle("Patches comparison")

    for n in range(cols):
        for r in range(2):
            if r == 0:
                axs[r, n].set_title("err=\n{:.02f},{:.02f}".format(diffs[n, 0].item(), diffs[n, 1].item()))
            axs[r, n].set_axis_off()
            patch_o = patches0[n] if r == 0 else patches1[n]
            patch = patch_o.clone().detach()
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
                             scale,
                             err_th,
                             compare=True,
                             show=False):

    print("Processing: {}".format(file_path))

    # convert and show the image
    img, img_r, real_scale = get_img_tuple(file_path, scale)

    scale_th = 15.0
    # TODO check the consistency of scales
    kpts, scales = detect(img, scale_th, show=show)
    kpts_r, scales_r = detect(img_r, scale_th*real_scale, show=show)

    if len(kpts) == 0 or len(kpts_r) == 0:
        return

    # TODO check integer th (original value of 2)
    kpts, scales, kpts_r, scales_r, diffs = mnn(kpts, scales, kpts_r, scales_r, real_scale, err_th)

    patches = get_patches(img, kpts, scales, scale_th=scale_th, show_few=show)
    patches_r = get_patches(img_r, kpts_r, scales_r, scale_th=scale_th, show_few=show)

    if compare:
        compare_patches(patches, patches_r, diffs)

    file_name_prefix = file_path[file_path.rfind("/") + 1:file_path.rfind(".")]
    for i in range(len(patches)):
        diff = diffs[i]
        patch = patches_r[i]
        file_name = "{}_{}.png".format(file_name_prefix, i)
        out_dict[file_name] = diff
        out_path = "{}/{}".format(out_dir, file_name)
        cv.imwrite(out_path, patch.numpy())


def prepare_data():

    repr_err_th = 2.0
    down_scale = 0.3

    max_items = None

    in_dir = "./dataset/raw_data"
    out_dir = "./dataset"

    if max_items:
        all = max_items
    else:
        all = len([fn for fn in os.listdir(in_dir) if fn.endswith(".tonemap.jpg")])

    data_dict = {}
    counter = 0
    for file_name in os.listdir(in_dir):

        if not file_name.endswith(".tonemap.jpg"):
            continue
        counter += 1
        if max_items and counter > max_items:
            break

        path = "{}/{}".format(in_dir, file_name)
        print("{}/{}".format(counter, all))
        process_patches_for_file(file_path=path,
                                 out_dir=out_dir,
                                 out_dict=data_dict,
                                 scale=down_scale,
                                 err_th=repr_err_th,
                                 compare=counter == 1,
                                 show=counter == 1)

    with open("{}/a_values.txt".format(out_dir), "w") as f:
        for k in data_dict:
            data = data_dict[k].numpy()
            f.write("{}, {}, {}\n".format(k, data[0], data[1]))


# continue: encapsulate the params -> in some configurable object (torch-lightning)
if __name__ == "__main__":
    prepare_data()
