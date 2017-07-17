"""
Based on the tflearn CIFAR-10 example at:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""

# from __future__ import division, print_function, absolute_import

from tflearn.data_utils import to_categorical

from boobs import boobs_model
import numpy as np
import shared.image as tf_image
import shared.model as tf_model


training_data = {
    2078: 4,
    3198: 4,
    3242: 0,
    3243: 0,
    3383: 2,
    3251: 4,
    2959: 4,
    3581: 3,
    3491: 1,
    2970: 2,
    3152: 4,
    3159: 2,
    1463: 1,
    2985: 1,
    2382: 1,
    1747: 4,
    3496: 2,
    2206: 4,
    3238: 2,
    3607: 4,
    2686: 3,
    2680: 3,
    2692: 4,
    2969: 1,
    2967: 2,
    494: 0,
    2574: 0,
    3087: 1,
    1830: 0,
    1835: 0,
    241: 0,
    1116: 0,
    1117: 0,
}


###################################
# Import picture files
###################################
# size_image = 100
size_image = boobs_model.get_image_size()
all_data = tf_image.load_all_training_img(tf_image.fullimg_files, size_image)

loaded_data = list()
for img in all_data:
    if img.image_id in training_data:
        img.image_label = training_data[img.image_id]
        loaded_data.append(img)
    elif img.image_id < 1500 and img.image_id % 10 == 0:
        img.image_label = 0
        loaded_data.append(img)


print('identified', len(loaded_data))
data_count = len(loaded_data)
category_count = 2

allX = np.zeros((data_count, size_image, size_image, 3), dtype='float64')
ally = np.zeros(data_count)

for idx, img in enumerate(loaded_data):
    allX[idx] = img.as_np_array()
    ally[idx] = img.image_label


###################################
# Prepare train & test samples
###################################

# test-train split
X, X_test, Y, Y_test = allX, allX, ally, ally

# encode the Ys
Y = to_categorical(Y, category_count)
Y_test = to_categorical(Y_test, category_count)


model_config = boobs_model.create_model(size_image, category_count)

model = tf_model.build_model(model_config)


###################################
# Train model for 100 epochs
###################################
model.fit(X, Y,
          #           validation_set=(X_test, Y_test),
          validation_set=0,
          batch_size=64,
          n_epoch=100, run_id='boobs', show_metric=True)

model.save(boobs_model.get_model_name())
