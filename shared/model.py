"""
Based on the tflearn CIFAR-10 example at:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""


from enum import Enum, auto

import tflearn
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy


class MatchType(Enum):
    # Indicates we're looking for the exact image
    exact = auto()
    # Indicates we're looking for some feature in the image
    feature = auto()


class ModelConfig:
    def __init__(self,
                 image_size, category_count,
                 ip_zero_center=True,
                 ip_stdnorm=True,
                 match_type: MatchType=MatchType.exact,
                 ia_random_blur=3.0,
                 ia_random_crop=-1,
                 network_conv_layer_1_size=(32, 3),
                 network_conv_layer_2_size=(32, 3),
                 network_conv_layer_3_size=(32, 3),
                 network_fullyconnected_layer_1_size=512,
                 network_fullyconnected_layer_2_size=0,
                 network_conv_dropout=0,
                 network_final_dropout=.5,
                 learning_rate=0.0003):

        self.image_size = image_size
        self.category_count = category_count
        self.ip_zero_center = ip_zero_center
        self.ip_stdnorm = ip_stdnorm
        self.match_type = match_type
        self.ia_random_blur = ia_random_blur
        self.ia_random_crop = ia_random_crop
        self.network_conv_layer_1_size = network_conv_layer_1_size
        self.network_conv_layer_2_size = network_conv_layer_2_size
        self.network_conv_layer_3_size = network_conv_layer_3_size
        self.network_fullyconnected_layer_1_size = network_fullyconnected_layer_1_size
        self.network_fullyconnected_layer_2_size = network_fullyconnected_layer_2_size
        self.network_conv_dropout = network_conv_dropout
        self.network_final_dropout = network_final_dropout
        self.learning_rate = learning_rate


def build_model(model_config: ModelConfig):
    # Image preprocessing steps
    img_prep = ImagePreprocessing()
    if model_config.ip_zero_center:
        img_prep.add_featurewise_zero_center()
    if model_config.ip_stdnorm:
        img_prep.add_featurewise_stdnorm()

    # Create extra synthetic training data by flipping & rotating images
    img_aug = ImageAugmentation()
    if model_config.ia_random_blur > 0:
        img_aug.add_random_blur(sigma_max=model_config.ia_random_blur)
    if model_config.match_type == MatchType.exact:
        #     if model_config.ia_random_crop >= 0:
        #         img_aug.add_random_crop((image_size, image_size), ia_random_crop)
        pass
    elif model_config.match_type == MatchType.feature:
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=25.)

    ###################################
    # Define network architecture
    ###################################

    size_image = model_config.image_size

    # Input is a 32x32 image with 3 color channels (red, green and blue)
    network = input_data(shape=[None, size_image, size_image, 3],
                         # network = input_data(shape=[None, size_image,
                         # size_image, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)

    # 1: Convolution layer with 32 filters, each 3x3x3
    if model_config.network_conv_layer_1_size:
        conv_layer = model_config.network_conv_layer_1_size
        network = conv_2d(
            network, conv_layer[0], conv_layer[1], activation='relu', name='conv_1')

    # 2: Max pooling layer
    network = max_pool_2d(network, 2)

    if model_config.network_conv_dropout > 0:
        network = dropout(network, model_config.network_conv_dropout)

    # 3: Convolution layer with 64 filters
    if model_config.network_conv_layer_2_size:
        conv_layer = model_config.network_conv_layer_2_size
        network = conv_2d(
            network, conv_layer[0], conv_layer[1], activation='relu', name='conv_2')

    # 4: Convolution layer with 64 filters
    if model_config.network_conv_layer_3_size:
        conv_layer = model_config.network_conv_layer_3_size
        network = conv_2d(
            network, conv_layer[0], conv_layer[1], activation='relu', name='conv_3')

    # 5: Max pooling layer
    network = max_pool_2d(network, 2)

    # 6: Fully-connected 512 node layer
    fc_layer1_size = model_config.network_fullyconnected_layer_1_size
    if fc_layer1_size > 0:
        network = fully_connected(network, fc_layer1_size, activation='relu')

    fc_layer2_size = model_config.network_fullyconnected_layer_2_size
    if fc_layer2_size > 0:
        network = fully_connected(network, fc_layer2_size, activation='relu')

    # 7: Dropout layer to combat overfitting
    network = dropout(network, model_config.network_final_dropout)

    # 8: Fully-connected layer with two outputs
    network = fully_connected(
        network, model_config.category_count, activation='softmax')

    # Configure how the network will be trained
    acc = Accuracy(name="Accuracy")
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=model_config.learning_rate,
                         metric=acc)

    # Wrap the network in a model object
    model = tflearn.DNN(network, checkpoint_path='model_pad.tflearn', max_checkpoints=3,
                        tensorboard_verbose=3, tensorboard_dir='tmp/tflearn_logs/')

    return model
