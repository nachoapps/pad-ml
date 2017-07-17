"""
"""

from glob import glob
import os

from PIL import Image
import PIL

import numpy as np


def load_image_files(img_dir):
    print(img_dir)
    img_glob_path = os.path.join(img_dir, '*.png')
    print(img_glob_path)
    result = sorted(glob(img_glob_path))
    print(len(result))
    return result


portrait_files = load_image_files('images/portraits')
fullimg_files = load_image_files('images/full')


class ModelImg:
    def __init__(self, file_path, image_size):
        self.file_path = file_path
        self.image_file_name = os.path.basename(file_path)
        self.image_name = os.path.splitext(self.image_file_name)[0]
        self.image_id = int(self.image_name)
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
    # This crop seems to add a black border, whereas the argb->rgb puts a white background
    # probably should fix that
    result = img.crop((
        -horizontal_padding,
        -vertical_padding,
        img.size[0] + horizontal_padding,
        img.size[1] + vertical_padding))

    return result


def load_all_training_img(src_files, image_size):
    results = list()
    count = 0
    for f in src_files:
        try:
            img = ModelImg(f, image_size=image_size)
            img.image_label = count
            results.append(img)
            count += 1
        except Exception as ex:
            print("failed to load image file:", f, ex)
    return results
