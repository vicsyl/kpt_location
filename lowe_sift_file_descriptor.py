import cv2 as cv
import numpy as np
from sift_detectors import BaseDescriptor


class LoweSiftDescriptor(BaseDescriptor):

    def __str__(self):
        return f"Lowe SIFT: {self.adjustment}"

    def __init__(self, adjustment=[0., 0.]):
        self.adjustment = adjustment

    def detectAndCompute(self, f_n, mask=None):

        assert mask is None
        with open(f_n, "r") as f:
            lines = list(f.readlines())
            kpts_c = int(lines[0].split(" ")[0])
            kpts_cv = []
            kpts_geo = []
            descs = []
            counter = 1
            for i in range(kpts_c):
                floats = [float(i) for i in lines[counter].split(" ")]
                counter += 1
                all_ints = []
                for j in range(7):
                    ints = [int(i) for i in lines[counter].split(" ") if len(i.strip()) > 0]
                    counter += 1
                    all_ints.extend(ints)
                kpts_cv.append(cv.KeyPoint(floats[1] + self.adjustment[1], floats[0] + self.adjustment[0], size=floats[2], angle=floats[3]))
                kpts_geo.append([floats[0], floats[1]])
                descs.append(np.array(all_ints, dtype=np.float32))
        descs = np.array(descs)
        kpts_geo = np.array(kpts_geo)
        # NOTE kpts_geo thrown away
        return kpts_cv, descs
