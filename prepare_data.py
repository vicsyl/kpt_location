import copy
import glob
import os
import time

import PIL
import matplotlib.pyplot as plt
import wandb
from PIL import Image

from patch_dataset import *
from wand_utils import wand_log_me


def mnn(kpts, kpts_scales, kpts_r, kpts_r_scales, scale, config):

    err_th = config['err_th']
    err_th = err_th / scale
    scale_ratio_th = config['scale_ratio_th']
    half_pixel_adjusted = config['half_pixel_adjusted']
    #kpts_reprojected_or = kpts_r / scale
    if half_pixel_adjusted:
        magic_scale = scale / 2 # hypothesis
        adjustment = torch.ones(2) * magic_scale
        kpts_r = kpts_r + adjustment

    kpts_reprojected = kpts_r / scale

    d_mat = torch.cdist(kpts, kpts_reprojected)

    # min0 => minima of resized
    min0 = torch.min(d_mat, dim=0)
    k = 3
    up_to_k_min = torch.topk(d_mat, k, largest=False, axis=0).values # [k, dim]
    min1 = torch.min(d_mat, dim=1)

    mask = min1[1][min0[1]] == torch.arange(0, min0[1].shape[0])
    mask_th = mask & (min0[0] < err_th)

    verify = True
    if verify:
        for i in range(min0[1].shape[0]):
            if mask_th[i]:
                assert min1[1][min0[1][i]] == i
                assert min0[0][i] < err_th

    kpts0 = kpts[min0[1][mask_th]]
    kpts_scales = kpts_scales[min0[1][mask_th]]

    kpts1 = kpts_r[mask_th]
    kpts_r_scales = kpts_r_scales[mask_th]

    dists = min0[0][mask_th]
    diffs = (kpts_reprojected[mask_th] - kpts0) * scale

    if verify:
        ds = torch.diag(torch.cdist(kpts0, kpts_reprojected[mask_th]))
        assert torch.allclose(ds, dists)

    # now filter based on the scale ratio threshold
    kpts_r_scales_backprojected = kpts_r_scales / scale
    scale_ratios = kpts_r_scales_backprojected / kpts_scales
    if scale_ratio_th is not None:
        mask_ratio_th = 1 + torch.abs(1 - scale_ratios) < scale_ratio_th
        diffs = diffs[mask_ratio_th]
        kpts0 = kpts0[mask_ratio_th]
        kpts1 = kpts1[mask_ratio_th]
        kpts_scales = kpts_scales[mask_ratio_th]
        kpts_r_scales = kpts_r_scales[mask_ratio_th]
        scale_ratios = scale_ratios[mask_ratio_th]

    min_distances_reprojected = min0[0] * scale
    up_to_k_min = up_to_k_min * scale
    assert torch.all(up_to_k_min[0, :] == min_distances_reprojected)
    # distances for reprojected
    return kpts0, kpts_scales, kpts1, kpts_r_scales, diffs, scale_ratios, up_to_k_min


def scale_pil(img, scale, config, show=False):
    """
    :param img:
    :param scale: in [0, 1]
    :param show:
    :return:
    """
    real_scale = scale
    h, w = img.size
    integer_scale = config['integer_scale']
    if integer_scale:
        gcd = gcd_euclid(w, h)

        real_scale_gcd = round(gcd * scale)
        real_scale = real_scale_gcd / gcd

        fall_back = False
        if real_scale == 0.0 or math.fabs(real_scale - scale) > 0.1:
            if fall_back:
                wand_log_me("WARNING: scale={} => {}".format(real_scale, scale), config)
                real_scale = scale
            else:
                raise Exception("scale {} cannot be effectively realized for w, h = {}, {} in integer domain".format(scale, w, h))

    w_sc = round(w * real_scale)
    h_sc = round(h * real_scale)
    img_r = img.resize((h_sc, w_sc), resample=PIL.Image.LANCZOS)
    if show:
        show_pil(img_r)

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


def check_grey(img_np, detector, kpt_f):
    # For SIFT only the check_COLOR_BGR2GRAY is true
    npa_grey = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY);
    kpts_grey = detector.detect(npa_grey, mask=None)
    kpt_f_g = np.array([[kp.pt[1], kp.pt[0]] for kp in kpts_grey])
    check_COLOR_BGR2GRAY_all = kpt_f_g == kpt_f
    check_COLOR_BGR2GRAY = np.alltrue(check_COLOR_BGR2GRAY_all)
    npa_grey = cv.cvtColor(img_np, cv.COLOR_RGB2GRAY);
    kpts_grey = detector.detect(npa_grey, mask=None)
    kpt_f_g = np.array([[kp.pt[1], kp.pt[0]] for kp in kpts_grey])
    check_COLOR_RGB2GRAY_all = kpt_f_g == kpt_f
    check_COLOR_RGB2GRAY = np.alltrue(check_COLOR_RGB2GRAY_all)


def detect_kpts(img_np, scale_th, const_patch_size, config):

    detector = get_detector(config)

    if const_patch_size is not None:
             assert const_patch_size % 2 == 1, "doesn't work that way"

    h, w = img_np.shape[:2]

    heatmap = None
    kpts = detector.detect(img_np, mask=None)
    if config['detector'].lower() == "superpoint":
        kpts, heatmap = kpts[0], kpts[1]
        heatmap = (heatmap * 255).astype(dtype=np.uint8)

    # NOTE show the keypoints
    # img_kpts = cv.cvtColor(img_np, cv.COLOR_GRAY2RGB)
    # cv.drawKeypoints(img_kpts, kpts, img_kpts, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.figure()
    # plt.title("kpts")
    # plt.imshow(img_kpts)
    # plt.show()
    # plt.close()

    if len(kpts) == 0:
        return [], []

    kpt_f = np.array([[kp.pt[1], kp.pt[0]] for kp in kpts])
    kpt_i = np.round(kpt_f).astype(int)

    # this is to check against grey scale handling in SuperPoint
    # check_grey(img_np, detector, kpt_f)

    scales = np.array([kp.size for kp in kpts])
    # NOTE in original image it just means it's not detected on the edge
    # even though the patch will not be put into the ds, which is still reasonable
    if const_patch_size is not None:
        margin = np.ones(scales.shape[0], dtype=int) * const_patch_size // 2
    else:
        margin = np.ceil(scales / 2).astype(int)

    mask = scales > scale_th
    mask = mask & (kpt_i[:, 0] >= margin) & (kpt_i[:, 1] >= margin)
    mask = mask & (kpt_i[:, 0] < h - margin) & (kpt_i[:, 1] < w - margin)
    kpt_f = kpt_f[mask]
    scales = scales[mask]

    show = config['show_kpt_patches']
    if show:
        npac = img_np.copy()
        cv.drawKeypoints(img_np, kpts, npac, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure()
        plt.title("kpts")
        plt.imshow(npac)
        plt.show()
        plt.close()

    # in [0, 360]
    orientations = torch.tensor([kpt.angle for kpt in kpts])[mask]
    return torch.from_numpy(kpt_f), torch.from_numpy(scales), orientations, heatmap


def print_and_check_margins(kpt_i, margins_np, img_t):
    print()
    for i, kp_i in list(enumerate(kpt_i)):
        t = ((kp_i[0] - margins_np[i]).item(),
              (kp_i[0] + margins_np[i]).item() + 1,
              (kp_i[1] - margins_np[i]).item(),
              (kp_i[1] + margins_np[i]).item() + 1)
        print(t, img_t.shape)


def show_patches(patches, label, config, detect):

    cols = 5
    rows = 5

    if detect:
        detector = get_detector(config)

    fig, axs = plt.subplots(rows, cols, figsize=(5, 5))
    fig.suptitle(label)

    for ix in range(rows):
        for iy in range(cols):
            axs[ix, iy].set_axis_off()
            if ix * cols + iy >= len(patches):
                axs[ix, iy].imshow(np.ones((32, 32, 3), np.uint8) * 255)
                continue
            patch_to_show = patches[ix * cols + iy].numpy().copy()
            if detect:
                kpts = detector.detect(patch_to_show, mask=None)
                cv.drawKeypoints(patch_to_show, kpts, patch_to_show, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            axs[ix, iy].imshow(patch_to_show)
    plt.show()
    plt.close()


def slice_patches(img_t, kpts, margins, heatmap_t):
    def slice_np(np_data):
        return [np_data[kp_i[0] - margins[i]: kp_i[0] + margins[i] + 1,
         kp_i[1] - margins[i]: kp_i[1] + margins[i] + 1] for i, kp_i in enumerate(kpts)]

    patches = slice_np(img_t)
    if heatmap_t is not None:
        patches_hm = slice_np(heatmap_t)
        return [patches, patches_hm]
    else:
        return [patches]


def get_patches_list(img_np, kpt_f, kpt_scales, const_patch_size, config, heatmap_np):
    """
    :param img_np:
    :param kpt_f:
    :param kpt_scales:
    :param scale_th:
    :param show_few:
    :return: patches: List[Tensor]
    """

    kpt_i = torch.round(kpt_f).to(torch.int)
    if const_patch_size is not None:
        assert const_patch_size % 2 == 1, "doesn't work that way"
        margins_np = np.ones(kpt_scales.shape[0], dtype=int) * const_patch_size // 2
    else:
        margins = torch.ceil(kpt_scales / 2.0)
        margins_np = margins.to(torch.int).numpy()
    # print_and_check_margins(kpt_i, margins_np, img_t)

    img_t = torch.tensor(img_np)
    heatmap_t = torch.tensor(heatmap_np) if heatmap_np is not None else None
    patches_list = slice_patches(img_t, kpt_i, margins_np, heatmap_t)

    show = config['show_kpt_patches']
    if show:
        detector = get_detector(config)
        kpts = detector.detect(img_np, mask=None)
        npac = img_np.copy()
        cv.drawKeypoints(img_np, kpts, npac, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_tc = torch.tensor(npac)
        ps_kpts = slice_patches(img_tc, kpt_i, margins_np)

        show_patches(ps_kpts, "Patches - original", config, detect=False)
        show_patches(patches_list, "Patches - redetected", config, detect=True)

    return patches_list


def compare_patches(patches0, patches1, diffs):

    cols = 8
    cols = min(cols, max(2, len(patches0)))

    fig, axs = plt.subplots(2, cols, figsize=(12, 4))
    fig.suptitle("Patches comparison")

    for n in range(min(cols, len(patches0))):
        for r in range(2):
            if r == 0:
                title = "err=\n{:.02f},{:.02f}".format(diffs[n, 0].item(), diffs[n, 1].item())
                axs[r, n].set_title(title)
            axs[r, n].set_axis_off()
            patch_o = patches0[n] if r == 0 else patches1[n]
            patch = patch_o.clone().detach()
            axs[r, n].imshow(patch.numpy())
    plt.show()


def get_pil_img(path, show=False):

    img = Image.open(path)
    if show:
        show_pil(img)
    #log_me("original size: {}".format(img.size))
    return img


def possibly_refl_image(config, img):
    refl = config.get('reflection', None)
    if refl:
        refl = refl.upper()
    if refl == "XY":
        img = np.flip(img, 0).copy()
        img = np.flip(img, 1).copy()
    elif refl == "Y":
        img = np.flip(img, 0).copy()
    elif refl == "X":
        img = np.flip(img, 1).copy()
    elif not refl:
        pass
    else:
        raise Exception("unknown value for reflection: {}".format(refl))
    return img


def possibly_to_grey_scale(config, img):
    to_grey_scale = config['to_grey_scale']
    if to_grey_scale:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    return img


def pil_img_transforms(img, config):
        img = np.array(img)
        img = possibly_to_grey_scale(config, img)
        img = possibly_refl_image(config, img)
        # FIXME a bit of a hack
        if config['detector'].lower() == "superpoint":
            new_h = (img.shape[0] // 8) * 8
            new_w = (img.shape[1] // 8) * 8
            img = img[:new_h, :new_w]
        return img


def get_img_tuple(path, scale, config, show=False):

    img = get_pil_img(path, show)
    img_r, real_scale = scale_pil(img, scale=scale, config=config, show=show)
    img = pil_img_transforms(img, config)
    img_r = pil_img_transforms(img_r, config)

    return img, img_r, real_scale


def apply_mask(mask, np_list):
    return tuple([npa[mask] for npa in np_list])


def augment_and_write_patch(patch, dr, file_name_prefix, out_map, out_dir, ds_config):

    write_imgs = ds_config['write_imgs']
    augment_mode = ds_config['augment'].lower()
    if "eager" == augment_mode:
        raise NotImplemented
        #patches_aug, diffs_aug, augmented_keys = augment_patch(patches, (dr.dy, dr.dx))
    else:
        patches_aug, diffs_aug, augmented_keys = [patch], [(dr.dy, dr.dx)], ["original"]
    for index, patch_aug in enumerate(patches_aug):
        dy, dx = diffs_aug[index]
        if index > 0:
            dr = copy.copy(dr)
        else:
            assert dr.dy == dy and dr.dx == dx
        dr.dy, dr.dx = dy, dx
        dr.augmented = augmented_keys[index]
        file_name = "{}_{}.png".format(file_name_prefix, index)
        out_map["metadata"][file_name] = dr
        img_out_path = "{}/data/{}".format(out_dir, file_name)
        if write_imgs:
            cv.imwrite(img_out_path, patch_aug.numpy())


# NOTE - not supported!!!
# def process_patches_for_file_dynamic(file_path,
#                                      config,
#                                      out_map,
#                                      key=""):
#
#     out_dir = get_full_ds_dir(config)
#     min_scale_th = config['min_scale_th']
#     const_patch_size = config.get('const_patch_size')
#     compare = config['compare_patches']
#     scale_ratio_th = config['scale_ratio_th']
#
#     # convert and show the image
#     img_orig_pil = get_pil_img(file_path)
#     img_orig = pil_img_transforms(img_orig_pil)
#
#     kpts, kpt_scales, _, _ = detect_kpts(img_orig, min_scale_th, const_patch_size=None, config=config)
#     if len(kpts) == 0:
#         return
#
#     counter = 0
#     max_scales = None # NOTE not implemented
#     matched = set()
#     for kpt_scale_index, kpt_scale in enumerate(kpt_scales):
#
#         if max_scales and counter > max_scales:
#             break
#
#         kpts_orig = kpts
#         kpt_scales_orig = kpt_scales
#
#         # FIXME test *2?
#         scale = (const_patch_size / scale_ratio_th) / kpt_scale.item()
#
#         img_r, real_scale = scale_pil(img_orig_pil, scale, config=config)
#         img_r = pil_img_transforms(img_r, config)
#
#         kpts_r, kpt_scales_r, _, _ = detect_kpts(img_r, min_scale_th*real_scale, const_patch_size, config)
#         if len(kpts_r) == 0:
#             continue
#
#         kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, diffs, scale_ratios, up_to_k_min = mnn(kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, real_scale, config)
#
#         #check kpt_scales_r
#         mask = (kpt_scales_r < const_patch_size) & (kpt_scales_r > const_patch_size - 2)
#         kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, diffs, scale_ratios = apply_mask(mask, [kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, diffs, scale_ratios])
#         mask = np.zeros(kpts_orig.shape[0], dtype=bool)
#         for i, kpt in enumerate(kpts_orig):
#             if not matched.__contains__(kpt):
#                 matched.add(kpt)
#                 mask[i] = True
#         kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, diffs, scale_ratios = apply_mask(mask, [kpts_orig, kpt_scales_orig, kpts_r, kpt_scales_r, diffs, scale_ratios])
#         patches = get_patches(img_orig, kpts_orig, kpt_scales_orig, const_patch_size, config)
#         patches_r = get_patches(img_r, kpts_r, kpt_scales_r, const_patch_size, config)
#         if compare:
#             compare_patches(patches, patches_r, diffs)
#
#         if key != "":
#             key = key + "_"
#         file_name_prefix = key + file_path[file_path.rfind("/") + 1:file_path.rfind(".")] + "_" + str(kpt_scale_index)
#         for i in range(len(patches)):
#             counter += 1
#             if max_scales and counter > max_scales:
#                 break
#             patch = patches_r[i]
#             dr = DataRecord(
#                 dy=diffs[i][0].item(),
#                 dx=diffs[i][1].item(),
#                 patch_size=patch.shape[0],
#                 kpt_orig_scale=kpt_scales_orig[i].item(),
#                 kpt_resize_scale=kpt_scales_r[i].item(),
#                 scale_ratio=scale_ratios[i].item(),
#                 real_scale=real_scale,
#                 img_scale_y=img_r.shape[0]/img_orig.shape[0],
#                 img_scale_x=img_r.shape[0]/img_orig.shape[0],
#                 original_img_size_y=img_orig.shape[0],
#                 original_img_size_x=img_orig.shape[1],
#                 resized_img_size_y=img_r.shape[0],
#                 resized_img_size_x=img_r.shape[1],
#                 augmented=None
#             )
#             augment_and_write_patch(patch,
#                                     dr,
#                                     "{}_{}".format(file_name_prefix, i),
#                                     out_map,
#                                     out_dir,
#                                     config)


def process_patches_for_file(file_path,
                             ds_config,
                             out_map,
                             key=""):

    dynamic_resizing = ds_config['dynamic_resizing']
    if dynamic_resizing:
        pass
        #process_patches_for_file_dynamic(file_path, config, out_map, key)
    else:
        process_patches_for_file_simple(file_path, ds_config, out_map, key)


def ensure_keys(m: map, keys: list):
    for key in keys:
        if not m.__contains__(key):
            m[key] = {}
        m = m[key]


def update_min_dists(min_distances_reprojected, out_map):
    ensure_keys(out_map, ["other", "minimal_dists"])
    if len(out_map["other"]["minimal_dists"]) == 0:
        out_map["other"]["minimal_dists"] = min_distances_reprojected.numpy()
    else:
        out_map["other"]["minimal_dists"] = np.hstack((out_map["other"]["minimal_dists"], min_distances_reprojected))


def process_patches_for_file_simple(file_path,
                                    ds_config,
                                    out_map,
                                    key=""):

    scale = ds_config['down_scale']
    out_dir = get_full_ds_dir(ds_config)
    min_scale_th = ds_config['min_scale_th']
    const_patch_size = ds_config.get('const_patch_size')
    compare = ds_config['compare_patches']

    # convert and show the image
    img, img_r, real_scale = get_img_tuple(file_path, scale, ds_config)

    kpts, kpt_scales, _, heatmap = detect_kpts(img, min_scale_th, const_patch_size, ds_config)
    kpts_r, kpt_scales_r, _, heatmap_r = detect_kpts(img_r, min_scale_th * real_scale, const_patch_size, ds_config)

    if len(kpts) == 0 or len(kpts_r) == 0:
        return

    kpts, kpt_scales, kpts_r, kpt_scales_r, diffs, scale_ratios, up_to_k_min = mnn(kpts, kpt_scales, kpts_r, kpt_scales_r, real_scale, ds_config)
    update_min_dists(up_to_k_min, out_map)

    patches_list = get_patches_list(img, kpts, kpt_scales, const_patch_size, ds_config, heatmap)
    patches_r_list = get_patches_list(img_r, kpts_r, kpt_scales_r, const_patch_size, ds_config, heatmap_r)

    if compare:
        # TODO test
        compare_patches(patches_list, patches_r_list, diffs)

    if key != "":
        key = key + "_"
    file_name_prefix = key + file_path[file_path.rfind("/") + 1:file_path.rfind(".")]
    for i in range(len(patches_list[0])):

        def concat_patches(p_l):
            if len(p_l) > 1:
                return torch.hstack(tuple(p_l))
            else:
                # TODO test this (e.g. with SIFT)
                return pl[0]
        patch = concat_patches([patch_r[i] for patch_r in patches_r_list])

        dr = DataRecord(
            dy=diffs[i][0].item(),
            dx=diffs[i][1].item(),
            # TODO becomes a bit irrelevant...
            patch_size=patch.shape[0],
            kpt_orig_scale=kpt_scales[i].item(),
            kpt_resize_scale=kpt_scales_r[i].item(),
            scale_ratio=scale_ratios[i].item(),
            real_scale=real_scale,
            img_scale_y=img_r.shape[0] / img.shape[0],
            img_scale_x=img_r.shape[0] / img.shape[0],
            original_img_size_y=img.shape[0],
            original_img_size_x=img.shape[1],
            resized_img_size_y=img_r.shape[0],
            resized_img_size_x=img_r.shape[1],
            augmented=None
        )
        augment_and_write_patch(patch,
                                dr,
                                "{}_{}".format(file_name_prefix, i),
                                out_map,
                                out_dir,
                                ds_config)


def get_ds_stats(entries):
    def adjust_min_max(min_max_stat, value):
        if min_max_stat[0] is None or min_max_stat[0] > value:
            min_max_stat[0] = value
        if min_max_stat[1] is None or min_max_stat[1] < value:
            min_max_stat[1] = value
        return min_max_stat

    patch_size_min_max = [None, None]
    scale_min_max = [None, None]
    scale_ratio_min_max = [None, None]
    for _, data_record in entries:
        patch_size_min_max = adjust_min_max(patch_size_min_max, data_record.patch_size)
        scale_min_max = adjust_min_max(scale_min_max, data_record.kpt_orig_scale)
        scale_ratio_min_max = adjust_min_max(scale_ratio_min_max, data_record.scale_ratio)

    return patch_size_min_max, scale_min_max, scale_ratio_min_max


def write_st(dataset_conf):
    write_metadata = dataset_conf['write_metadata']
    write_data = dataset_conf['write_data']
    return write_metadata or write_data


def prepare_and_clean_dir(dataset_config):
    out_dir = get_full_ds_dir(dataset_config)
    clean = dataset_config['clean_out_dir']
    if write_st(dataset_config):
        data_dir_path = "{}/data".format(out_dir)
        if not os.path.exists(data_dir_path):
            os.makedirs(data_dir_path, exist_ok=True)
        if clean:
            try:
                path = '{}/a_other.npz'.format(out_dir)
                os.remove(path)
                path = '{}/a_metadata.npz'.format(out_dir)
                os.remove(path)
            except:
                print("couldn't remove {}".format(path))
            glob_str = '{}/data/*.png'.format(out_dir)
            files = glob.glob(glob_str)
            print("will be removing {} files for glob '{}'".format(len(files), glob_str))
            for path in files:
                try:
                    os.remove(path)
                except:
                    print("couldn't remove {}".format(path))


def log_table(data, column, table_ref=None, title=None):
    if not title:
        title = column
    if not table_ref:
        table_ref = column
    t_d = wandb.Table(data=data, columns=[column])
    wandb.log({table_ref: wandb.plot.histogram(t_d, column, title=title)})


def write_data(dataset_config, orig_out_map):
    write_metadata = dataset_config['write_metadata']
    if write_metadata:
        out_map = orig_out_map["metadata"]
        out_dir = get_full_ds_dir(dataset_config)
        with open("{}/a_metadata.txt".format(out_dir), "w") as md_file:
            log_metada(out_map, dataset_config, log_wand=False, file=md_file)

    write_data = dataset_config['write_data']
    if write_data:
        with open("{}/a_values.txt".format(out_dir), "w") as md_file:
            log_metada(out_map, dataset_config, log_wand=False, file=md_file)

            md_file.write("# schema: {}\n".format(DataRecord.schema()))
            for (fn, value) in out_map.items():
                to_write = "{}, {}\n".format(fn, value.line_str())
                md_file.write(to_write)

    write_other = dataset_config['write_other']
    if write_other:
        other = orig_out_map["other"]
        np.savez("{}/a_other".format(out_dir), **other)


def read_img(file_path, config):
    img_pil = get_pil_img(file_path)
    img_np = pil_img_transforms(img_pil, config)
    return img_np, img_pil


def prepare_data(dataset_config, in_dirs, keys):

    ends_with = dataset_config['ends_with']
    max_files = dataset_config['max_files']
    const_patch_size = dataset_config.get('const_patch_size')
    if const_patch_size is not None:
            assert const_patch_size % 2 == 1, "doesn't work that way"

    prepare_and_clean_dir(dataset_config)

    out_map = {"metadata": {}}
    all = str(max_files) if max_files is not None else None

    for i, in_dir in enumerate(in_dirs):

        counter = 0

        if not max_files:
            all = len([fn for fn in os.listdir(in_dir) if fn.endswith(ends_with)])

        print("Processing dir {}: {}/{}".format(in_dir, i + 1, len(in_dirs)))
        key = keys[i]

        for file_name in sorted(os.listdir(in_dir)):

            if not file_name.endswith(ends_with):
                continue
            counter += 1
            if max_files and counter > max_files:
                break

            path = "{}/{}".format(in_dir, file_name)
            print("Processing file {}: {}/{}, learning examples: {}".format(path, counter, all, len(out_map["metadata"])))
            process_patches_for_file(file_path=path,
                                     ds_config=dataset_config,
                                     out_map=out_map,
                                     key=key)

    write_data(dataset_config, out_map)
    return list(out_map["metadata"].items())


def log_metada(out_map, dataset_conf, log_wand=False, file=None, conf_to_log=None):

    if not conf_to_log:
        conf_to_log = dataset_conf

    def log_all(str):
        print(str)
        if file:
            file.write(f"{str}\n")
        # if log_wand:
        #     wandb.log(str)

    def print_min_max_stat(stat, name):
        log_all("# {} (min, max): ({}, {})".format(name, stat[0], stat[1]))

    def print_m_am_stat(stat, name, skip_abs_mean=False):
        mean, abs_mean = mean_abs_mean(stat)
        std_dev = np.sqrt(stat.var(axis=0))

        log_all("# mean {} error: {}".format(name, mean))
        if not skip_abs_mean:
            log_all("# absolute mean {} error: {}".format(name, abs_mean))
        log_all("# std dev of {} error: {}".format(name, std_dev))

    log_all("# entries: {}".format(len(out_map)))
    detector_name = dataset_conf['detector'].upper()
    log_all(f"# detector: {detector_name}")

    distances, errors, angles = get_error_stats(out_map.items())
    print_m_am_stat(distances, "distance", skip_abs_mean=True)
    print_m_am_stat(distances ** 2, "distance squared", skip_abs_mean=True)
    print_m_am_stat(errors, "")
    print_m_am_stat(angles, "angle")

    patch_size_min_max, scale_min_max, scale_ratio_min_max = get_ds_stats(out_map.items())
    print_min_max_stat(patch_size_min_max, "patch size")
    print_min_max_stat(scale_min_max, "original scale")
    print_min_max_stat(scale_ratio_min_max, "scale ratio")

    log_all("### CONFIG ###")
    for k, v in list(dataset_conf.items()):
        log_all("#\t\t\t{}: {}".format(k, v))
    log_all("### CONFIG ###")


def prepare_data_by_scale(scales, wandb_project="mean_std_dev"):

    dataset_conf = get_config()['dataset']
    dataset_conf["tags"] = ["development"]
    in_dirs = dataset_conf['in_dirs']
    keys = dataset_conf['keys']

    print("### CONFIG ###")
    for k, v in list(dataset_conf.items()):
        print("#\t\t\t{}: {}".format(k, v))
    print("### CONFIG ###")

    means = []
    std_devs = []

    for i, scale in enumerate(scales):
        start_time = time.time()
        dataset_conf['down_scale'] = scale
        set_config_dir_scale_scheme(dataset_conf)
        print("Scale={}".format(scale))
        entry_list = prepare_data(dataset_conf, in_dirs, keys)
        _, errors, _ = get_error_stats(entry_list)
        mean, _ = mean_abs_mean(errors)
        means.append(mean.tolist())
        std_dev = np.sqrt(errors.var(axis=0)).tolist()
        std_devs.append(std_dev)
        end_time = time.time()
        print("{}. out of {} scales took {:.4f} seconds.".format(i + 1, len(scales), end_time - start_time))

    std_devs = np.array(std_devs)
    means = np.array(means)

    # NOTE this can also be done by iterating (over scale) through already prepared data
    if wandb_project:

        # tags...
        wandb.init(project=wandb_project, name=get_wand_name(dataset_conf, entry_list), tags=dataset_conf["tags"])

        data = [[x, y] for (x, y) in zip(scales, means[:, 1])]
        table = wandb.Table(data=data, columns=["scale", "mean error x"])
        wandb.log({"mean error x": wandb.plot.line(table, "scale", "mean error x", title="Mean error(x) as a function of scale")})

        data = [[x, y] for (x, y) in zip(scales, means[:, 0])]
        table = wandb.Table(data=data, columns=["scale", "mean error y"])
        wandb.log({"mean error y": wandb.plot.line(table, "scale", "mean error y", title="Mean error(y) as a function of scale")})

        data = [[x, y] for (x, y) in zip(scales, std_devs[:, 1])]
        table = wandb.Table(data=data, columns=["scale", "std dev x"])
        wandb.log({"std dev(x)": wandb.plot.line(table, "scale", "std dev x", title="Std dev(x) of error as a function of scale")})

        data = [[x, y] for (x, y) in zip(scales, std_devs[:, 0])]
        table = wandb.Table(data=data, columns=["scale", "std dev y"])
        wandb.log({"std dev(y)": wandb.plot.line(table, "scale", "std dev y", title="Std dev(y) of error as a function of scale")})


def simple_prepare_data():
    start_time = time.time()
    config = get_config()['dataset']
    in_dirs = config['in_dirs']
    keys = config['keys']
    prepare_data(config, in_dirs, keys)
    end_time = time.time()
    print("it took {:.4f} seconds.".format(end_time - start_time))


if __name__ == "__main__":

    def tenths(_from, to):
        return [scale_int / 10 for scale_int in range(_from, to)]

    #scales = tenths(1, 10)
    scales = [0.3]

    prepare_data_by_scale(scales, wandb_project=None)
    #simple_prepare_data()
