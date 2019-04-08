"""
Handle image preprocessing tasks
"""
import logging

import numpy as np


logger = logging.getLogger(__name__)

class ImageDataGenerator(object):

    def __init__(self,
                 preprocessing_function=None,
                 stride=1,
                 batch_size=32,
                 patch_size=None)
        
        self.preprocessing_function = preprocessing_function
        self.stride = stride
        self.batch_size = batch_size
        self.patch_size = patch_size

        self.prepfuncs = {
            'bl': self.bl,
            'bl_cd': self.bl_cd,
            'bl_cd_pn': self.bl_cd_pn,
            'bl_cd_pn_ag': self.bl_cd_pn_ag,
        }

        if preprocessing_function not in self.prepfuncs:
            raise TypeError("preprocessing function not available: {}".\
                            format(preprocessing_function))

    def parse_sony_list(self, sony_list: str):
        try:
            with open(sony_list, "r") as infile:
                sony_pairs = infile.readlines()

            np.random.shuffle(sony_pairs)

            pairs = []
            for idx, pair in enumerate(sony_pairs):

                shrt, lng, = pair.split(" ")

                pairs.append((shrt, lng))

            return pairs

        except Exception as exc:
            logger.exception(exc)
            raise exc

    def _train_val_sony(self, dirpath: str, sony_list: str):
        sony_pairs = self.parse_sony_list(sony_list)
        
        batch = []
        batch_count = 0
        for X_file_name, Y_file_name in sony_pairs:

            X_file_path = urljoin(dirpath, X_file_name)
            Y_file_path = urljoin(dirpath, Y_file_name)

            X_img = image.load_img(X_file_path)
            Y_img = image.load_img(Y_file_path)

            X = X_img.img_to_array(X_img)
            Y = Y_img.img_to_array(Y_img)

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

    def dirflow_val_sony(self, sony_val_list: str):
        """ Yield validation set for sony. """
        yield self._train_val_sony(sony_val_list)

    def dirflow_train_sony(self, sony_train_list: str):
        """ Yield train set for sony. """
        yield self._train_val_sony(sony_train_list)

    def dirflow_test_sony(self, dirpath, sony_test_list):
        """ We want uneven batch sizes so the image can be patched back. """
        sony_pairs = self.parse_sony_list(sony_test_list)

        for X_file_name, Y_file_name in sony_pairs:

            X_file_path = urljoin(dirpath, X_file_name)
            Y_file_path = urljoin(dirpath, Y_file_name)

            X_img = image.load_img(X_file_path)
            Y_img = image.load_img(Y_file_path)

            X = X_img.img_to_array(X_img)
            Y = Y_img.img_to_array(Y_img)

            X = self.crop(X)
            Y = self.crop(Y)

            batch = []
            for X_patch, Y_patch in zip(self.extract_patches(X),
                                        self.extract_patches(Y)):

                batch.append((X_patch, Y_patch))

            yield np.array(batch)

    def dirflow_train_raise(self, dirpath):
        """ Generator for RAISE dataset during training

        Equal batch sizes are generated during training except
        for the last one.
        """
        fnames = os.listdir(dirpath)

        if len(fnames) == 0:
            raise TypeError("No file names found in directory")

        batch = []
        batch_count = 0
        for file_name in fnames:
            
            file_path = urljoin(dirpath, file_name)
            img = image.load_img(file_path)

            Y = img.img_to_array(img)

            # Crop each image, do not resize
                
            Y = self.crop(Y)

            # Extract patches from each image

            for Y_patch in self.extract_patches(Y):

                # Apply relevant noise function

                X_patch = np.copy(Y_patch)

                X_patch = self.prepfuncs[self.preprocessing_function](X_patch)

                if batch_count < self.batch_size:

                    batch.append((X_patch, Y_patch))
                    batch_count += 1

                else:
                    XY_batch = np.array(batch)
                    yield XY_batch

                    batch = []
                    batch.append((X_patch, Y_patch))
                    batch_count = 1

        # Final batch includes all remaining patches

        if len(batch) > 0:
            XY_batch = np.array(batch)
            yield XY_batch

    def dirflow_test_raise(self. dirpath):
        """ Generator for RAISE dataset during testing

        Here we don't care about the batch size because we
        aren't effecting the loss function and need to stitch 
        the predicted image together for model review.
        """
        fnames = os.listdir(dirpath)
        if len(fnames) == 0:
            raise TypeError("No file names found in directory")
        
        for file_name in fnames:

            uneven_batch = []
            file_path = urljoin(dirpath, file_name)
            img = image.load_img(file_path)

            Y = img.img_to_array(img)

            # Crop each image, do not resize
                
            Y = self.crop(Y)

            # Extract patches from each image

            uneven_batch = []
            for Y_patch in self.extract_patches(Y):

                # Apply relevant noise function

                X_patch = np.copy(Y_patch)

                X_patch = self.prepfuncs[self.preprocessing_function](X_patch)

                uneven_batch.append((X_patch, Y_patch))

            yield np.array(uneven_batch)

    def valid_sample(self):
        sample = [-1, -1, -1, -1, -1]   # Make sure sample isn't negative
        while not(sample[0]>0 and sample[1]>0 and sample[2]>0 \
                  and sample[3]>0 and sample[4]>0):
                sample = np.random.multivariate_normal(MEANM, COVM)
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

        image = bl(image, sample)

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

        image = bl_cd(image, sample)

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

        image = bl_cd_pn(image, sample)
        image = image**sample[4]
        return image


    def extract_patches(self, data, patch_size, stride = 1,
                         random_state=None):
    
        def _compute_n_patches(i_h, i_w, p_h, p_w):

            n_h = i_h - p_h + 1
            n_w = i_w - p_w + 1
            all_patches = n_h * n_w

            return all_patches
    
        def get_patches(arr, patch_shape):
            arr_ndim = arr.ndim

            extraction_step = tuple([stride] * arr_ndim)

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
            i_h, i_w = image.shape[:2]
            p_h, p_w = patch_size

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
    
        image_patches = []
        for idx in range(batch):
            image_patches.extend(np.array(list(map(_ex_pt,
                                                   data[idx]))).\
                                 reshape((-1, pair, patch_size[0],
                                          patch_size[1], 3)))
    
        return np.array(image_patches)

    def crop(self, image):
        i_h, i_w = image.shape[:2]
        p_h, p_w = self.patch_size

        crop_h, crop_w = i_h-((i_h-p_h)%self.stride), \
            i_w-((i_w-p_w)%self.stride)

        return img[:crop_h, :crop_w]


    def crop_images(self, data):
    
        batch = data.shape[0]
        pair = data.shape[1]
    
        cropped_images = []
        for idx in range(batch):
            cropped_images.append(np.array(list(map(self.crop, data[idx]))))
    
        return np.array(cropped_images)
    
    def reconstruct_patches(self, patches, image_size, stride=1):
        i_h, i_w = image_size[:2]
        p_h, p_w = patches.shape[1:3]
        img = np.zeros(image_size)
        img_map = np.zeros(image_size)

        n_h = i_h - p_h + 1
        n_w = i_w - p_w + 1
        for p, (i, j) in zip(patches, product(range(0, n_h, stride),
                                              range(0, n_w, stride))):
            img_map[i:i + p_h, j:j + p_w] += 1
            img[i:i + p_h, j:j + p_w] += p
    
        return np.divide(img,img_map,
                         out=np.zeros_like(img, dtype=np.float32),
                         where=img_map!=0)


