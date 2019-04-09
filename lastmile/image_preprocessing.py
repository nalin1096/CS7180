"""
Handle image preprocessing tasks
"""
from itertools import product
import logging
import math
import os
import pickle
from urllib.parse import urljoin

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from tensorflow.keras.utils import Sequence

from skimage.io import imread
from skimage.transform import resize


logger = logging.getLogger(__name__)


class ImageDataPipeline(object):

    def __init__(self, 
                 preprocessing_function=None,
                 stride=1,
                 batch_size=32,
                 patch_size=None,
                 random_seed=None,
                 meanm_fpath='',
                 covm_fpath='',
                 image_dims=(512,512,3),
                 num_images=10
    ):
        
        self.preprocessing_function = preprocessing_function
        self.stride = stride
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.random_seed = random_seed
        self.meanm = self.read_pickle(meanm_fpath)
        self.covm = self.read_pickle(covm_fpath)
        self.num_images = num_images

        self.prepfuncs = {
            'bl': self.bl,
            'bl_cd': self.bl_cd,
            'bl_cd_pn': self.bl_cd_pn,
            'bl_cd_pn_ag': self.bl_cd_pn_ag,
            'sony': None,
        }

        if preprocessing_function not in self.prepfuncs:
            raise TypeError("preprocessing function not available: {}".\
                            format(preprocessing_function))

    def read_pickle(self, fpath):
        try:
            with open(fpath, "rb") as infile:
                m = pickle.load(infile)
            return m

        except FileNotFoundError:
            logger.warning("Filepath not set for pickle")

    def reformat_imgpath(self, img_path: str):
        rfmt = img_path[7:-4]
        rfmt = 'Sony/rgb/' + rfmt + ".png"
        return rfmt

    def parse_sony_list(self, sony_list: str):
        try:
            with open(sony_list, "r") as infile:
                sony_pairs = infile.readlines()

            np.random.seed(self.random_seed)
            np.random.shuffle(sony_pairs)

            pairs = []
            for idx, pair in enumerate(sony_pairs):

                items = pair.strip().split(" ")

                if len(items) != 4:
                    raise TypeError("Corrupted list: {}".format(sony_list))

                pairs.append((self.reformat_imgpath(items[0]),
                              self.reformat_imgpath(items[1])))

            return pairs

        except Exception as exc:
            logger.exception(exc)
            raise exc

    def _train_val_sony(self, sony_list: str):
        sony_pairs = self.parse_sony_list(sony_list)
        
        batch = []
        batch_count = 0
        for X_file_path, Y_file_path in sony_pairs:

            X = cv2.imread(X_file_path)
            Y = cv2.imread(Y_file_path)

            X = self.crop(X)
            Y = self.crop(Y)

            for X_patch, Y_patch in zip(self.extract_patches(X),
                                        self.extract_patches(Y)):

                if batch_count < self.batch_size:

                    batch.append((X_patch, Y_patch))
                    batch_count += 1

                else:

                    yield np.array(batch)

                    batch = []
                    batch.append((X_patch, Y_patch))
                    batch_count = 1

        if len(batch) > 0:
            yield np.array(batch)


    def dirflow_test_sony(self, sony_test_list):
        """ We want uneven batch sizes so the image can be patched back. """
        sony_pairs = self.parse_sony_list(sony_test_list)

        for X_file_path, Y_file_path in sony_pairs:

            X = cv2.imread(X_file_path)
            Y = cv2.imread(Y_file_path)

            X = self.crop(X)
            Y = self.crop(Y)

            batch = []
            for X_patch, Y_patch in zip(self.extract_patches(X),
                                        self.extract_patches(Y)):

                batch.append((X_patch, Y_patch))

            yield np.array(batch)

    def raise_pipeline(self, Y):

        # Crop each image, do not resize
                
        Y = self.crop(Y)
        logger.debug("Y crop dimensions: {}".format(Y.shape))

        # Extract patches from each image

        logger.debug("Y shape: {}".format(Y.shape))
        for Y_patch in self.extract_patches(Y):

            # Apply relevant noise function

            X_patch = np.copy(Y_patch)
            X_patch = self.prepfuncs[self.preprocessing_function](X_patch)
            yield (X_patch, Y_patch)

    def valid_sample(self):
        np.random.seed(self.random_seed)
        sample = [-1, -1, -1, -1, -1]   # Make sure sample isn't negative
        while not(sample[0]>0 and sample[1]>0 and sample[2]>0 \
                  and sample[3]>0 and sample[4]>0):
                sample = np.random.multivariate_normal(self.meanm, self.covm)
        return sample

    def bl(self, image, sample=False):
        """ Apply black level """
        if not np.all(sample):
            sample = self.valid_sample()

        BL = int(sample[0])
        image[image < BL] = BL
        image = image - BL
        return image

    def bl_cd(self, image, sample=False):
        """ Apply black level with color distortion """

        if not np.all(sample):
            sample = self.valid_sample()

        image = self.bl(image, sample)

        WB = [ sample[1], sample[2], sample[3] ]
    
        image[... ,0] = WB[0] * image[... ,0]
        image[... ,1] = WB[1] * image[... ,1]
        image[... ,2] = WB[2] * image[... ,2]

        return image

    def bl_cd_pn(self, image, sample=False):
        """ Apply black level with color distortion and poisson noise. """
    
        if not np.all(sample):
            sample = self.valid_sample()

        noise_param = 10

        image = self.bl_cd(image, sample)

        np.random.seed(self.random_seed)
        noise = lambda x : np.random.poisson(x / 255.0 * noise_param) / \
            noise_param * 255

        func = np.vectorize(noise)
        image = func(image)
        return image

    def bl_cd_pn_ag(self, image, sample=False):
        """ 
        Apply black level, color distortion, poisson noise, adjust gamma. 
        """

        if not np.all(sample):
            sample = self.valid_sample()

        image = self.bl_cd_pn(image, sample)
        image = image**sample[4]
        return image

    def extract_patches(self, data):
    
        def _compute_n_patches(i_h, i_w, p_h, p_w):

            n_h = i_h - p_h + 1
            n_w = i_w - p_w + 1
            all_patches = n_h * n_w

            return all_patches
    
        def get_patches(arr, patch_shape):
            arr_ndim = arr.ndim

            extraction_step = tuple([self.stride] * arr_ndim)

            patch_strides = arr.strides

            slices = tuple(slice(None, None, st) for st in extraction_step)
            indexing_strides = arr[slices].strides

            patch_indices_shape = ((np.array(arr.shape) - \
                                    np.array(patch_shape)) //
                                   np.array(extraction_step)) + 1

            shape = tuple(list(patch_indices_shape) + list(patch_shape))
            strides = tuple(list(indexing_strides) + list(patch_strides))

            patches = as_strided(arr, shape=shape, strides=strides)
            return patches

        def _ex_pt(image):
            logger.debug("img shape: {}".format(image[0].shape))
            i_h, i_w = image.shape[:2]
            p_h, p_w = self.patch_size

            image = image.reshape((i_h, i_w, -1))
            n_colors = image.shape[-1]

            extracted_patches = get_patches(image,
                                            patch_shape=(p_h, p_w, n_colors))
            
            n_patches = _compute_n_patches(i_h, i_w, p_h, p_w)

            patches = extracted_patches

            patches = patches.reshape(-1, p_h, p_w, n_colors)
            # remove the color dimension if useless
            if patches.shape[-1] == 1:
                return patches.reshape((n_patches, p_h, p_w))
            else:
                return patches

        batch = data.shape[0]
        pair = data.shape[1]

        logger.debug("shape of data: {}".format(data.shape))
        #image_patches = []
        #for idx in range(batch):
        #    image_patches.extend(np.array(list(map(_ex_pt,
        #                                           data[idx]))).\
        #                         reshape((-1, pair, self.patch_size[0],
        #                                  self.patch_size[1], 3)))
        #
        #return np.array(image_patches)
        return _ex_pt(data)

    def crop(self, image):
        i_h, i_w = image.shape[:2]
        p_h, p_w = self.patch_size

        crop_h, crop_w = i_h-((i_h-p_h)%self.stride), \
            i_w-((i_w-p_w)%self.stride)

        return image[:crop_h, :crop_w]
        # return image[:512, :512] #TODO: hard coded dims


    def crop_images(self, data):
    
        batch = data.shape[0]
        pair = data.shape[1]
    
        cropped_images = []
        for idx in range(batch):
            cropped_images.append(np.array(list(map(self.crop, data[idx]))))
    
        return np.array(cropped_images)
    
    def reconstruct_patches(self, patches, image_size):
        i_h, i_w = image_size[:2]
        p_h, p_w = patches.shape[1:3]
        img = np.zeros(image_size)
        img_map = np.zeros(image_size)

        n_h = i_h - p_h + 1
        n_w = i_w - p_w + 1
        for p, (i, j) in zip(patches, product(range(0, n_h, self.stride),
                                              range(0, n_w, self.stride))):
            img_map[i:i + p_h, j:j + p_w] += 1
            img[i:i + p_h, j:j + p_w] += p
    
        return np.divide(img,img_map,
                         out=np.zeros_like(img, dtype=np.float32),
                         where=img_map!=0)

    def image_to_arr(self, img_path):
        Y = cv2.imread(img_path)
        return Y

class RiseDataGenerator(Sequence):
    
    def __init__(self, y_set, idp):
        self.y = y_set
        self.batch_size = idp.batch_size
        self.idp = idp
        
    def __len__(self):
        # ASSUMES image size is (512,512), stride is 128
        return int(np.ceil(len(self.y)  / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        X_batch = []
        Y_batch = []
        for Y_filepath in batch_y:

            Y = cv2.imread(Y_filepath)
            logger.debug("Y input dim: {}".format(Y.shape))

            for X_patch, Y_patch in self.idp.raise_pipeline(Y):

                X_batch.append(X_patch)
                Y_batch.append(Y_patch)

        item = (np.array(X_batch), np.array(Y_batch))
        return item

    def __iter__(self):
        logger.debug("len of self: {}".format(len(self)))
        for item in (self[i] for i in range(len(self))):
            logger.debug("item type from dunder iter: {}".format(type(item)))

            if item[0].shape[0] != 0:
                yield item

class SonyDataGenerator(Sequence):

    def __init__(self):
        pass


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/utils/data_utils.py


