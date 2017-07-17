import shared.model as tf_model


def get_model_name():
    return 'model_pad_final.tflearn'


def create_model(size_image, category_count):
    # Base model - 92%
    #model_config = tf_model.ModelConfig(size_image, category_count)

    # Increase blur -  90%
    #model_config = tf_model.ModelConfig(size_image, category_count, ia_random_blur=5.0)

    # No blur - 95%
    # model_config = tf_model.ModelConfig(size_image, category_count, ia_random_blur=0.0)

    # No blur, lower dropout - bad, keep .5
    # model_config = tf_model.ModelConfig(size_image, category_count,
    #                                     ia_random_blur=0.0,
    #                                     network_final_dropout=.1)

    # No blur - add second layer - 93%
    # model_config = tf_model.ModelConfig(size_image, category_count,
    #                                    ia_random_blur=0.0,
    #                                    network_fullyconnected_layer_2_size=512)

    # No blur - add second layer - one conv
    # model_config = tf_model.ModelConfig(size_image, category_count,
    #                                     ia_random_blur=0.0,
    #                                     network_conv_layer_2_size=None,
    #                                     network_conv_layer_3_size=None,
    #                                     network_fullyconnected_layer_2_size=512)

    # # No blur - add second layer - one conv, 20/20 -> 96%!
    # model_config = tf_model.ModelConfig(size_image, category_count,
    #                                     ia_random_blur=0.0,
    #                                     network_conv_layer_1_size=(20, 20),
    #                                     network_conv_layer_2_size=None,
    #                                     network_conv_layer_3_size=None,
    #                                     network_fullyconnected_layer_2_size=512)
    #
    # # No blur - add second layer - one conv, 20/20, with mid dropout -> 95%
    # model_config = tf_model.ModelConfig(size_image, category_count,
    #                                     ia_random_blur=0.0,
    #                                     network_conv_layer_1_size=(20, 20),
    #                                     network_conv_layer_2_size=None,
    #                                     network_conv_layer_3_size=None,
    #                                     network_conv_dropout=.5,
    #                                     network_fullyconnected_layer_2_size=512)
    #
    # # No blur - add second layer - two conv, 20/20, 30/30 -> 96%
    # model_config = tf_model.ModelConfig(size_image, category_count,
    #                                     ia_random_blur=0.0,
    #                                     network_conv_layer_1_size=(20, 20),
    #                                     network_conv_layer_2_size=None,
    #                                     network_conv_layer_3_size=None,
    #                                     network_fullyconnected_layer_2_size=512)

    # No blur - add second layer - two conv, 20/20, 30/30, 30/30 -> 94%
    model_config = tf_model.ModelConfig(size_image, category_count,
                                        ia_random_blur=0.0,
                                        network_conv_layer_1_size=(8, 2),
                                        network_conv_layer_2_size=(24, 2),
                                        network_conv_layer_3_size=(24, 2),
                                        # network_fullyconnected_layer_2_size=512,
                                        learning_rate=.0003
                                        #                                     network_conv_layer_2_size=(30, 30),
                                        #                                     network_conv_layer_3_size=(30, 30),
                                        # network_fullyconnected_layer_2_size=512)
                                        )

    return model_config
