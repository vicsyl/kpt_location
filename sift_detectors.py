import cv2 as cv
import numpy as np


class AdjustedSiftDetector:

    def __init__(self, adjustment=[0.25, 0.25]):
        self.cv_sift = cv.SIFT_create()
        self.ajustment = adjustment

    def detect(self, img_np, mask=None):

        cv_kpts = self.cv_sift.detect(img_np, mask)
        for cv_kpt in cv_kpts:
            # x, y!!
            pt = np.array(cv_kpt.pt) + self.ajustment
            cv_kpt.pt = tuple(pt)

        return cv_kpts
