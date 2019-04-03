import logging

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
import rawpy
import imageio


logger = logging.getLogger(__name__)


def adjust_gamma(image):
    GAMMA = np.random.normal(0.9, 0.1)
    return image**GAMMA

def add_noise(image):
    return random_noise(image, mode='poisson') * 255.0

def remove_white_balance(image):
    WB = [np.random.normal(0.8, 0.04),
          np.random.normal(0.8, 0.04),
          np.random.normal(0.7, 0.04)]

    image[... ,0] = WB[0] * image[... ,0]
    image[... ,1] = WB[1] * image[... ,1]
    image[... ,2] = WB[2] * image[... ,2]

    return image


class SimulateCondition(object):
    
    def __init__(self):

        self.GAMMA = np.random.normal(0.9, 0.1)
        self.NOISE_PEAK = 0.5
        self.WB = [np.random.normal(0.8, 0.04), np.random.normal(0.8, 0.04),
                   np.random.normal(0.7, 0.04)]
        self.BL = 30
        
    def adjust_gamma(self, image):
        invGamma = 1.0 / self.GAMMA
        table = np.array([((i / 255.0) ** invGamma) * 255 for i
                          in np.arange(0, 256)]).astype("uint8")
        logger.debug("> Applied gamma correction")
        return cv2.LUT(image, table)
    
    def add_noise(self, image):
        noisy = np.uint8(random_noise(image, mode='poisson') * 255.0)
        logger.debug("> Noise Added")
        return noisy

    def remove_white_balance(self, image):
        white_unbalanced =  np.zeros((self.img_rows,self.img_cols,
                                      self.img_colors), dtype='uint8')
        white_unbalanced[:,:,0] = np.int8(self.WB[0] * image[:,:,0])
        white_unbalanced[:,:,1] = np.int8(self.WB[1] * image[:,:,1])
        white_unbalanced[:,:,2] = np.int8(self.WB[2] * image[:,:,2])
        logger.debug("> White unbalanced")
        return white_unbalanced
    
    def increase_black_level(self, image):
        bl_image = np.copy(image)
        bl_image[bl_image<self.BL] = self.BL
        bl_image = bl_image - self.BL
        logger.debug("> Black level increased")
        return bl_image
