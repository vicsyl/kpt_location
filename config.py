from omegaconf import OmegaConf


def get_config(path = 'config/config.yaml'):
    return OmegaConf.load(path)


def get_full_ds_dir(config):
    path_size = config.get('const_patch_size')
    out_dir = config['out_dir']
    if path_size is None:
        return out_dir
    else:
        return out_dir + str(path_size)


def get_config_map():
    return {
        'err_th': 2.0,
        'down_scale': 0.3,
        'in_dir': "./dataset/raw_data",
        #'out_dir': "./dataset/var_sizes",
        'out_dir': "./dataset/const_size_",
        'const_patch_size': 33,
        'max_items': None,
        #'ends_with': '.jpg',
        'ends_with': '.tonemap.jpg',
        'min_scale_th': 15.0,
        'clean_out_dir': True
    }
