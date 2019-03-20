""" Tiny Imagenet

https://tiny-imagenet.herokuapp.com/
"""
import os
from urllib.parse import urljoin

import numpy as np
from tensorflow.keras.preprocessing import image


def image_array(img_path: str):
    """ np.NDArray Shape (64,64,3) """
    img = image.load_img(img_path, target_size=(64,64))
    x = image.img_to_array(img)
    return x

def ndarray_inmemory(imgdir: str):

    items = []
    for file_name in os.listdir(imgdir):

        fpath = urljoin(imgdir, file_name)
        x = image_array(fpath)
        items.append(x)

    X = np.array(items)
    return X


