"""
Based on the tflearn CIFAR-10 example at:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""


import operator

import numpy as np
from portraits import portrait_model
import shared.image as tf_image
import shared.model as tf_model


###################################
# Import picture files
###################################
size_image = 96
loaded_data = tf_image.load_all_training_img(
    tf_image.portrait_files, size_image)
category_count = len(loaded_data)


model_config = portrait_model.create_model(size_image, category_count)

model = tf_model.build_model(model_config)
model.load(portrait_model.get_model_name())

count = 0
for img in loaded_data:
    allX = np.zeros((1, size_image, size_image, 3), dtype='float64')
    allX[0] = img.as_np_array()
    results = model.predict(allX)

    index, value = max(enumerate(results[0]), key=operator.itemgetter(1))
    if int(img.image_label) == int(index):
        pass
#       print("good", index)
    else:
        print("bad match for", img.image_name, "got",
              loaded_data[index].image_name, "confidence", value)
#     print("for", img.image_label, "max was", index, "value", value)
