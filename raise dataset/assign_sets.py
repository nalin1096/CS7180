""" Assign RAISE images to train/test/val sets. """
import logging
import os
import shutil
from urllib.parse import urljoin

import numpy as np

logger = logging.getLogger(__name__)

def copy_raise_file(num, indir, outdir):
    png = str(num) + ".png"
    src = urljoin(indir, png)
    dst = urljoin(outdir, png)

    shutil.copyfile(src, dst)

def assign_images(dist: tuple, fpaths: dict, size: int):

    train, test, val = dist

    imgs = np.array([i for i in range(1, size + 1)])

    np.random.shuffle(imgs)

    train_size = np.floor(size * train)
    test_size = np.floor(size * test)
    val_size = size - train_size + test_size
    indir = fpaths.get('indir', None)
    
    count = 0
    for img_num in imgs.tolist():

        if train_size > 0:
            outdir = fpaths.get('train', None)
            copy_raise_file(img_num, indir, outdir)
            train_size -= 1
            
        elif test_size > 0:
            outdir = fpaths.get('test', None)
            copy_raise_file(img_num, indir, outdir)
            test_size -= 1

        elif val_size > 0:
            outdir = fpaths.get('val', None)
            copy_raise_file(img_num, indir, outdir)
            val_size -= 1

        else:
            print("error: {}".format(img_num))


        if count % 100 == 0:
            logger.info("Processed {} images".format(count))

        count += 1


if __name__ == "__main__":

    logging.basicConfig(filename='assign_sets.log',
                        format='%(asctime)s %(message)s',
                        level=logging.INFO)
    logger.info("STARTED assigning sets")
    dist = (0.94, 0.03, 0.03)
    fpaths = {
        'indir': 'rgb/',
        'train': 'rgb/train/',
        'test': 'rgb/test/',
        'val': 'rgb/val/',
    }
    assign_images(dist, fpaths, size=1000)
    logger.info("FINISHED assigning sets")
