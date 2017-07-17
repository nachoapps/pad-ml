"""
Based on the tflearn CIFAR-10 example at:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""

# from __future__ import division, print_function, absolute_import

from tflearn.data_utils import to_categorical

import numpy as np
from portraits import portrait_model
import shared.image as tf_image
import shared.model as tf_model


###################################
# Import picture files
###################################
# size_image = 100
size_image = 96
loaded_data = tf_image.load_all_training_img(
    tf_image.portrait_files, size_image)
category_count = len(loaded_data)

allX = np.zeros((category_count, size_image, size_image, 3), dtype='float64')
ally = np.zeros(category_count)

for idx, img in enumerate(loaded_data):
    allX[idx] = img.as_np_array()
    ally[idx] = img.image_label


###################################
# Prepare train & test samples
###################################

# test-train split
X, X_test, Y, Y_test = allX, allX, ally, ally
# X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

# X = np.reshape(X, (-1, size_image, size_image, 3))
# X_test = np.reshape(X_test, (-1, size_image, size_image, 3))
# Y = np.reshape(Y, (-1, size_image, size_image, 3))
# Y_test = np.reshape(Y_test, (-1, size_image, size_image, 3))

# encode the Ys
Y = to_categorical(Y, category_count)
Y_test = to_categorical(Y_test, category_count)


model_config = portrait_model.create_model(size_image, category_count)


model = tf_model.build_model(model_config)

# model = tf_model.build_nin(size_image, category_count)


###################################
# Train model for 100 epochs
###################################
model.fit(X, Y,
          #           validation_set=(X_test, Y_test),
          validation_set=0,
          batch_size=64,
          n_epoch=70, run_id='portrait_model_run', show_metric=True)

model.save(portrait_model.get_model_name())
