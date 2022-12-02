import cv2 as cv
import kornia.utils as KU
import numpy as np
from hloc.extractors.dog import DoG
from sift_detectors import BaseDescriptor


class HlocSiftDescriptor(BaseDescriptor):

    opencv_like_conf = {
        'options': {
            'first_octave': -1,
            'peak_threshold': 0.04,
        },
        'descriptor': 'sift',
        'max_keypoints': -1,
        'patch_size': 32,
        'mr_size': 12,
    }

    default_conf = {
        'options': {
            'first_octave': 0,
            'peak_threshold': 0.01,
        },
        'descriptor': 'rootsift',
        'max_keypoints': -1,
        'patch_size': 32,
        'mr_size': 12,
    }

    def __str__(self) -> str:

        return f"HLOC SIFT: adj={self.abs_adjustment[0]}, conf={self.conf_name}"

    def __init__(self, conf, adjustment=[0.0, 0.0]):
        # self.dog = None # use the line below
        if conf == HlocSiftDescriptor.default_conf:
            self.conf_name = "default_conf"
        elif conf == HlocSiftDescriptor.opencv_like_conf:
            self.conf_name = "opencv_like_conf"
        else:
            self.conf_name = "custom_conf"
        self.dog = DoG(conf)
        self.abs_adjustment = np.array(adjustment)

    def adjust_cv_kpts(self, cv_kpts):
        for cv_kpt in cv_kpts:
            # x, y!!
            # CONTINUE
            pt = np.array(cv_kpt.pt) + self.abs_adjustment
            cv_kpt.pt = tuple(pt)

    def create_cv_kpts(self, keypoints, scales, oris):
        keypoints = keypoints[0]
        kpts = []
        for i, kpt in enumerate(keypoints):
            x = kpt[0].item() + self.abs_adjustment[1]
            y = kpt[1].item() + self.abs_adjustment[0]
            size = scales[0, i].item()
            angle = oris[0, i].item()
            kp = cv.KeyPoint(x, y, size=size, angle=angle)
            kpts.append(kp)
        return kpts

    def detectAndCompute(self, img_np, mask):

        if len(img_np.shape) == 2:
            img_np = img_np[:, :, None]
        else:
            img_np = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)

        img_t = KU.image_to_tensor(img_np, False).float() / 255.
        ret_dict = self.dog({"image": img_t})
        cv_kpts = self.create_cv_kpts(ret_dict['keypoints'], ret_dict['scales'], ret_dict['oris'])
        cv_descs = ret_dict['descriptors'][0].T.numpy()
        return cv_kpts, cv_descs
