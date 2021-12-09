# This class will contain the backbone module
import tensorflow as tf
import tensorflow_hub as hub


class Backbone:

    # Start by downloading pretrained weights from the Tensorflow hub
    def __init__(self):

        # Instantiate the feature extractor.
        # This is the feature of this class we will actually want to use
        self.feature_extractor = hub.KerasLayer(
            "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5",
            trainable=True,
        )

        # TODO standardize the thumbansil sizes, I just hard coded in the feature sizes here and in data_utils.py
        self.thumbnail_detector = tf.keras.Sequential(
            [self.feature_extractor, tf.keras.layers.Dense(2, activation='softmax')]
        )
        self.thumbnail_detector.build([None, 128, 128, 3])

    def fine_tune(dataloader, epochs):

        # This method will take the thumbnail dataloader class and train for some number
        # of epochs to fine tune the weights.
        # We need to figure out what fine tuning parameters to use.

        pass
