import shared.model as tf_model


def get_image_size():
    return 640 // 4


def get_model_name():
    return 'model_boobs.tflearn'


def create_model(size_image, category_count):
    # 36% after 70 epocs
    #     model_config = tf_model.ModelConfig(size_image, category_count,
    #                                         match_type=tf_model.MatchType.feature,
    #                                         network_conv_layer_1_size=(20, 20),
    #                                         network_conv_layer_2_size=None,
    #                                         network_conv_layer_3_size=None,
    #                                         network_fullyconnected_layer_2_size=512)

    # No blur - add second layer - two conv, 20/20, 30/30, 30/30 -> 94%
    model_config = tf_model.ModelConfig(size_image, category_count,
                                        ia_random_blur=0.0,
                                        network_conv_layer_1_size=(8, 2),
                                        network_conv_layer_2_size=(24, 2),
                                        network_conv_layer_3_size=(24, 2),
                                        learning_rate=.0003
                                        )

    return model_config
