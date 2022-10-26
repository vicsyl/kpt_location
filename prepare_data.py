import copy
import glob
import os
import sys
import time

import PIL
from PIL import Image

from patch_dataset import *
from wand_utils import wand_log_me

sys.path.append('/content/kpt_location')
from prepare_data_files import run_command
from dataset_utils import get_dirs_and_keys
from config import *


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
        # FIXME
        #assert torch.allclose(ds, dists)

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
    # FIXME temporary fix (use is hm_relevant)
    if config['detector'].lower().__contains__("superpoint"):
        kpts, heatmap = kpts[0], kpts[1]
        if heatmap is None:
            heatmap = np.zeros_like(img_np).astype(dtype=np.uint8)
        else:
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
        return [], [], [], None

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
        npac = cv.cvtColor(img_np, cv.COLOR_GRAY2RGB)
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


def crop_patches(img_t, kpts, margins, heatmap_t):
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
    patches_list = crop_patches(img_t, kpt_i, margins_np, heatmap_t)

    show = config['show_kpt_patches']
    if show:
        detector = get_detector(config)
        kpts = detector.detect(img_np, mask=None)
        if len(kpts) == 2:
            kpts = kpts[0]
        npac = img_np.copy()
        cv.drawKeypoints(img_np, kpts, npac, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_tc = torch.tensor(npac)
        ps_kpts = crop_patches(img_tc, kpt_i, margins_np, None)

        #show_patches(ps_kpts, "Patches - original", config, detect=False)
        #show_patches(patches_list, "Patches - redetected", config, detect=True)

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
        # FIXME temporary fix (use is hm_relevant)
        if config['detector'].lower().__contains__("superpoint"):
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


def write_patch(patch, dr, file_name_prefix, out_map, out_dir, ds_config, augment_index):

    write_imgs = ds_config['write_imgs']
    dr.augmented = "original" if augment_index == 0 else "augmented"
    file_name = "{}_{}.png".format(file_name_prefix, augment_index)
    out_map["metadata"][file_name] = dr
    img_out_path = "{}/data/{}".format(out_dir, file_name)
    if write_imgs:
        cv.imwrite(img_out_path, patch.numpy())


# def process_patches_for_file(file_path,
#                              ds_config,
#                              out_map,
#                              key=""):
#
#     dynamic_resizing = ds_config['dynamic_resizing']
#     if dynamic_resizing:
#         pass
#         #process_patches_for_file_dynamic(file_path, config, out_map, key)
#     else:
#         process_patches_for_file_simple(file_path, ds_config, out_map, key)
#
#
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
                                    key):

    scale = ds_config['down_scale']
    # convert and show the image
    img_or, img_r_or, real_scale = get_img_tuple(file_path, scale, ds_config)

    augment_index = 0
    process_patches_for_images(img_or,
                               img_r_or,
                               real_scale,
                               ds_config,
                               out_map,
                               file_path,
                               key,
                               augment_index)

    augment_type = ds_config['augment'].lower()
    if augment_type != "eager" and augment_type != "eager_averaged":
        return

    if augment_type == "eager_averaged":
        # give ...
        # kpts, kpt_scales, _, heatmap = detect_kpts(img, min_scale_th, const_patch_size, ds_config)
        # ... with the avg'd kpts
        raise NotImplemented

    img = img_or
    img_r = img_r_or
    for rot_index in range(3):
        augment_index += 1
        img = np.rot90(img, rot_index, [0, 1]).copy()
        img_r = np.rot90(img_r, rot_index, [0, 1]).copy()
        process_patches_for_images(img,
                                   img_r,
                                   real_scale,
                                   ds_config,
                                   out_map,
                                   file_path,
                                   key,
                                   augment_index)
    for axis in range(2):
        augment_index += 1
        img = np.flip(img_or, axis=axis).copy()
        img_r = np.flip(img_r_or, axis=axis).copy()
        process_patches_for_images(img,
                                   img_r,
                                   real_scale,
                                   ds_config,
                                   out_map,
                                   file_path,
                                   key,
                                   augment_index)


def process_patches_for_images(img,
                               img_r,
                               real_scale,
                               ds_config,
                               out_map,
                               file_path,
                               key,
                               augment_index):

    out_dir = get_full_ds_dir(ds_config)
    min_scale_th = ds_config['min_scale_th']
    const_patch_size = ds_config.get('const_patch_size')
    compare = ds_config['compare_patches']

    # either here or given
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
                return p_l[0]
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
        write_patch(patch,
                    dr,
                    "{}_{}".format(file_name_prefix, i),
                    out_map,
                    out_dir,
                    ds_config,
                    augment_index)


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
            log_metada(out_map, dataset_config, file=md_file)

    write_data = dataset_config['write_data']
    if write_data:
        with open("{}/a_values.txt".format(out_dir), "w") as md_file:
            log_metada(out_map, dataset_config, file=md_file)

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

        if not max_files:
            all = len([fn for fn in os.listdir(in_dir) if fn.endswith(ends_with)])

        print("Processing dir {}: {}/{}".format(in_dir, i + 1, len(in_dirs)))
        key = keys[i]

        file_names = [fn for fn in sorted(os.listdir(in_dir)) if fn.endswith(ends_with)]
        if max_files:
            file_names = file_names[:max_files]

        for i, file_name in enumerate(file_names):

            path = "{}/{}".format(in_dir, file_name)
            print("Processing file {}: {}/{}, learning examples: {}".format(path, i, all, len(out_map["metadata"])))
            #process_patches_for_file
            process_patches_for_file_simple(file_path=path,
                                            ds_config=dataset_config,
                                            out_map=out_map,
                                            key=key)

    write_data(dataset_config, out_map)
    return list(out_map["metadata"].items())


def log_metada(out_map, dataset_conf, file=None, conf_to_log=None):

    if not conf_to_log:
        conf_to_log = dataset_conf
        loss = None
    else:
        loss = conf_to_log["train"]["loss"].upper()

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
        return mean

    log_all("# entries: {}".format(len(out_map)))
    detector_name = dataset_conf['detector'].upper()
    log_all(f"# detector: {detector_name}")

    distances, errors, angles = get_error_stats(out_map.items())
    distances_mean = print_m_am_stat(distances, "distance", skip_abs_mean=True)
    squared_distances_mean = print_m_am_stat(distances ** 2, "squared distance", skip_abs_mean=True)
    # TODO check the 1/2
    if loss is None:
        expected_loss = 0
    elif loss == "L1":
        expected_loss = distances_mean / 2
    elif loss == "L2":
        expected_loss = squared_distances_mean / 2

    print_m_am_stat(errors, "")
    print_m_am_stat(angles, "angle")

    patch_size_min_max, scale_min_max, scale_ratio_min_max = get_ds_stats(out_map.items())
    print_min_max_stat(patch_size_min_max, "patch size")
    print_min_max_stat(scale_min_max, "original scale")
    print_min_max_stat(scale_ratio_min_max, "scale ratio")

    log_all("### CONFIG ###")
    log_all("#" + OmegaConf.to_yaml(conf_to_log).replace("\n", "\n#"))
    log_all("### CONFIG ###")
    return expected_loss


def prepare_data_by_scale(scales, wandb_project="mean_std_dev"):

    conf = get_config()
    dataset_conf = conf['dataset']
    dataset_conf["tags"] = ["development"]
    enable_wandb = dataset_conf.get('enable_wandlog', False)

    if enable_wandb:
        # tags...
        wandb.init(project=wandb_project, name=get_wand_name(dataset_conf), tags=dataset_conf["tags"])
        wandb.config = OmegaConf.to_container(conf)
    start_time = time.time()

    in_dirs = dataset_conf['in_dirs']
    keys = dataset_conf['keys']

    print("### CONFIG ###")
    print("#" + OmegaConf.to_yaml(dataset_conf).replace("\n", "\n#"))
    print("### CONFIG ###")

    means = np.zeros((len(scales), 2))
    sq_error = np.zeros(len(scales))
    std_devs = np.zeros((len(scales), 2))

    for i, scale in enumerate(scales):
        start_time_scale = time.time()
        dataset_conf['down_scale'] = scale
        set_config_dir_scale_scheme(dataset_conf)
        print("Scale={}".format(scale))
        entry_list = prepare_data(dataset_conf, in_dirs, keys)
        _, errors, _ = get_error_stats(entry_list)
        means[i] = errors.mean(axis=0)
        sq_error[i] = (errors ** 2).sum(axis=1).mean()
        std_devs[i] = np.sqrt(errors.var(axis=0)).tolist()
        end_time = time.time()
        print("{}. out of {} scales took {:.4f} seconds.".format(i + 1, len(scales), end_time - start_time_scale))

    end_time = time.time()
    print("it took {:.4f} seconds.".format(end_time - start_time))

    # NOTE this can also be done by iterating (over scale) through already prepared data
    if enable_wandb:

        data = [[x, y] for (x, y) in zip(scales, means[:, 1])]
        table = wandb.Table(data=data, columns=["scale", "mean error x"])
        wandb.log({"mean error x": wandb.plot.line(table, "scale", "mean error x", title="Mean error(x) as a function of scale")})

        data = [[x, y] for (x, y) in zip(scales, means[:, 0])]
        table = wandb.Table(data=data, columns=["scale", "mean error y"])
        wandb.log({"mean error y": wandb.plot.line(table, "scale", "mean error y", title="Mean error(y) as a function of scale")})

        data = [[x, y] for (x, y) in zip(scales, sq_error)]
        table = wandb.Table(data=data, columns=["scale", "mean square distance error"])
        wandb.log({"mean squared distance error": wandb.plot.line(table, "scale", "mean square distance error", title="Mean square distance error as a function of scale")})

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


def prepare_data_all(base_dir, config_path, filter_list):

    config = get_config(path=config_path)['dataset']
    dirs_to_process = 1000
    in_dirs, keys = get_dirs_and_keys(dirs_to_process, base_dir=base_dir)

    if filter_list is not None and len(filter_list) > 0:
        # SIMPLE filtering
        in_dirs_2 = []
        keys_2 = []
        for i, dir in enumerate(in_dirs):
            for cont in filter_list:
                # if dir.__contains__("001_001") or dir.__contains__("001_002") or dir.__contains__("001_003"):
                if dir.__contains__(cont):
                    in_dirs_2.append(dir)
                    keys_2.append(keys[i])
                    break
        in_dirs = in_dirs_2
        keys = keys_2

    print("final keys: {}".format(keys))
    print("final indirs: {}".format(in_dirs))
    prepare_data(config, in_dirs, keys)


def prepare_data_from_files():

    # prepare_data_all(base_dir="/content")
    # list_contains = ["001_001", "001_002"]

    #list_contains = ["ai_001_001", "ai_001_002", "ai_001_003", "ai_001_004", "ai_001_005", "ai_001_006", "ai_001_007", "ai_001_008", "ai_001_009", "ai_001_010"]
    list_contains = []
    prepare_data_all(base_dir="./unzips",
                     config_path="./config/config_train_cluster.yaml",
                     filter_list=list_contains)

    print("dataset ls")

    #run_command("ls - alh  / content / dataset / *")
    print("dataset data ls | head")
    #!ls - alhtr / content / dataset / * / data | head - n  20
    print("a_values.txt | head")
    #!head / content / dataset / * / a_values.txt - n 50
    print("data consumption")
    # !du - d 0 - h / content / dataset / * / data


if __name__ == "__main__":

    simple_prepare_data()

    #prepare_data_from_files()

    # def tenths(_from, to):
    #     return [scale_int / 10 for scale_int in range(_from, to)]
    #
    # scales = tenths(1, 10)
    # #scales = [0.3]
    #
    # prepare_data_by_scale(scales)
    # #simple_prepare_data()
