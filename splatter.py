import cv2
import numpy as np
import random
import os

import utils
import perturber


class NoSplatsException(Exception):
    pass

class Splatter(object):
    PERCENTAGE = 5.0 / 100 # Size of splat/image
    splats = []
    def __init__(self, imgShape=(480, 640), splats_dir='.'):
        shape_per = (int(imgShape[0] * self.PERCENTAGE), int(imgShape[1] * self.PERCENTAGE))
        for f in os.listdir(splats_dir):
            splat = cv2.imread(os.path.join(splats_dir, f), 0)
            if splat is not None:
                splat = cv2.resize(splat, shape_per, interpolation=cv2.cv.CV_INTER_NN)
                self.splats.append(splat)
        if not self.splats:
            raise NoSplatsException(os.path.realpath(splats_dir))
        s = random.choice(self.splats)


    def _splat_once(self, t):
        s = random.choice(self.splats)
        s = perturber.rand_rotate(perturber.rand_e_d(s))
        utils.safe_random_embed(t, s, mask=True)

    
    def splat_mask(self, m):
        "Gets a mask and splats it"
        t = np.zeros(m.shape, np.uint8)
        for i in xrange(300):
            self._splat_once(t)
        
        return t & m

        