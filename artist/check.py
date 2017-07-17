"""
Based on the tflearn CIFAR-10 example at:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""


import operator

from boobs import boobs_model
import numpy as np
import shared.image as tf_image
import shared.model as tf_model


###################################
# Import picture files
###################################
size_image = boobs_model.get_image_size()
loaded_data = tf_image.load_all_training_img(tf_image.fullimg_files, size_image)
category_count = 2


model_config = boobs_model.create_model(size_image, category_count)

model = tf_model.build_model(model_config)
model.load(boobs_model.get_model_name())

count = 0
for img in loaded_data:
    allX = np.zeros((1, size_image, size_image, 3), dtype='float64')
    allX[0] = img.as_np_array()
    results = model.predict(allX)

    index, value = max(enumerate(results[0]), key=operator.itemgetter(1))
    print('{},{},{}'.format(img.image_name, index + 1, value))
