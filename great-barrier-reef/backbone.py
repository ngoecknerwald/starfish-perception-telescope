# This class will contain the backbone module
import tensorflow as tf
import tensorflow_hub as hub


class Backbone:

    # Start by downloading pretrained weights from the Tensorflow hub
    def __init__(self, mode='classify', input_size=(128,128)):

        # Instantiate the feature extractor.
        # This is the feature of this class we will actually want to use
        self.feature_model = hub.load("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5")
        self.feature_extractor = hub.KerasLayer(
            feature_model,
            trainable=True,
        )

        self.thumbnail_detector = tf.keras.Sequential([self.feature_extractor])

        if mode == 'classify':
        # TODO standardize the thumbansil sizes, I just hard coded in the feature sizes here and in data_utils.py
          self.thumbnail_detector.add(
              tf.keras.layers.Dense(1, activation=None)
        ) 
        input_h, input_w = input_size
        self.thumbnail_detector.build([None, input_h, input_w, 3])

    def fine_tune(dataloader, epochs):

        # This method will take the thumbnail dataloader class and train for some number
        # of epochs to fine tune the weights.
        # We need to figure out what fine tuning parameters to use.

        pass

    def save_backbone(self, path):
        tf.saved_model.save(self.feature_model, path)
