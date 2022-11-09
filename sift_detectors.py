import math

import cv2 as cv
import numpy as np


class AdjustedSiftDescriptor:

    def __str__(self) -> str:
        return f"{self.abs_adjustment[0]}/{self.q_adjustment[0]}"

    def __init__(self, adjustment, nfeatures=None, lin_adjustment=[0.0, 0.0], sqrt_adjustment=[0.0, 0.0], q_adjustment=[0.0, 0.0]):
        self.cv_sift = cv.SIFT_create() if not nfeatures else cv.SIFT_create(nfeatures=nfeatures)
        self.abs_adjustment = np.array(adjustment)
        self.lin_adjustment = np.array(lin_adjustment)
        self.sqrt_adjustment = np.array(sqrt_adjustment)
        self.q_adjustment = np.array(q_adjustment)

    def adjust_cv_kpts(self, cv_kpts):
        for cv_kpt in cv_kpts:
            # x, y!!
            # CONTINUE
            pt = np.array(cv_kpt.pt) + self.abs_adjustment + cv_kpt.size * self.lin_adjustment + \
                 math.sqrt(cv_kpt.size) * self.sqrt_adjustment + math.sqrt(cv_kpt.size)**2 * self.q_adjustment
            cv_kpt.pt = tuple(pt)

    def detect(self, img_np, mask=None):

        cv_kpts = self.cv_sift.detect(img_np, mask)
        self.adjust_cv_kpts(cv_kpts)
        return cv_kpts

    def detectAndCompute(self, img_np, mask):

        cv_kpts, cv_desc = self.cv_sift.detectAndCompute(img_np, mask)
        self.adjust_cv_kpts(cv_kpts)
        return cv_kpts, cv_desc
