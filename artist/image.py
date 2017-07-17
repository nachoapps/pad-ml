"""
"""

from glob import glob
import operator
import os

from PIL import Image
import PIL
from sklearn.cross_validation import train_test_split

import numpy as np

from .model import *


portraits_dir = 'images/portraits'

portrait_files_path = os.path.join(portraits_dir, '*.png')
portrait_files = sorted(glob(portrait_files_path))
n_files = len(portrait_files)


DEFAULT_IMAGE_SIZE = 100


class ModelImg:
    def __init__(self, file_path, image_size=DEFAULT_IMAGE_SIZE):
        self.file_path = file_path
        self.image_file_name = os.path.basename(file_path)
        self.image_name = os.path.splitext(self.image_file_name)[0]
        self.image_size = image_size

        img = Image.open(file_path).convert('RGB')
        if max(img.size) > min(img.size):
            img = pad_to_square(img)

        self.img = img.resize((image_size, image_size), PIL.Image.ANTIALIAS)

        self.image_label = None

    def as_np_array(self):
        return np.array(self.img)


def pad_to_square(img):
    longer_side = max(img.size)
    horizontal_padding = (longer_side - img.size[0]) / 2
    vertical_padding = (longer_side - img.size[1]) / 2
    result = img.crop((
        -horizontal_padding,
        -vertical_padding,
        img.size[0] + horizontal_padding,
        img.size[1] + vertical_padding))

    result.show()

    return result


def load_all_training_img(image_size=DEFAULT_IMAGE_SIZE):
    results = list()
    count = 0
    for f in portrait_files:
        try:
            img = ModelImg(f, image_size=image_size)
            img.image_label = count
            results.append(img)
            count += 1
        except Exception as ex:
            print("failed to load image file:", f, ex)
    return results
