# This class will contain the backbone module
import tensorflow as tf
import tensorflow_hub as hub


class Backbone:

    # Start by downloading pretrained weights from the Tensorflow hub
    def __init__(self, pretrained_model='https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5', **kwargs):
        '''
        Initialize the network backbone. By default downloads the Inception Resnet V2 pretrained on ImageNet.
        Provides methods to fine tune the training on thumbnail images and convert the full image into feature maps.

        Arguments:

        model : str, 'https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5' default
            Pretrained convolutional weights to download from the TF hub.
        **kwargs : {}
            Arguments passed to hub.KerasLayer() when downloading the model
        '''


        # Instantiate the feature extractor. This is the feature of this class we will actually want to use
        self.feature_model = hub.load(pretrained_model)
        
        # Feature extractor, TODO need to figure out the total stride of this network
        # It looks like we can set lots of tunable params, like weight decay from the documentation
        # I'm trying to pull things we want to tune as hyperparams out of the lower classes to the high level code
        self.feature_extractor = hub.KerasLayer(
            feature_model,
            trainable=True,
            **kwargs
        )
        # The things connected to this model will hand down the input image size,
        # and then need to know how 
        self.output_layers = 99999999 # TODO figure this out
        self.output_stride = 99999999 # TODO figure this out

    def save_backbone(self, path):
        '''
        Save the trained convolutional layers of the backbone to a file path.

        Arguments:

        path: str
            Save path for the (tuned) backbone model.
        '''

        tf.saved_model.save(self.feature_model, path)

    def load_backbone(self, path)
        '''
        Load the tuned backbone from a path.

        path: str
            Load path for the (tuned) backbone model.
        '''

        self.feature_model = tf.saved_model.load(path)

    def build_backbone(self, full_size, channels=3):

        '''
        Builds the backbone model given the input image size.

        full_size: (int, int)
            Size of the input images in pixels.
        channels: int, defaults to 3
            Number of (RGB) channels in the input image.

        Creates the attribute self.backbone
        '''

        self.backbone = tf.keras.Sequential([self.feature_extractor])
        self.backbone.build(None, full_size[0], full_size[1], channels)

    def build_classifier(self, thumbnail_size, channels=3):
        '''
        Appends a simple fully connected layer to the downloded convolutional weights
        and builds for a set thumbnail size. This model can then be fine tuned with
        self.fine_tune()

        thumbnail_size: (int, int)
            Size of the input images in pixels, 128x128 by fault.
        channels: int, defaults to 3
            Number of (RGB) channels in the input image. 

        '''

        # Build a classification model to train the feature extractor
        # Note to self - is this degenerate with training the RPN?
        self._thumbnail_detector = tf.keras.Sequential([self.feature_extractor])
        self._thumbnail_detector.add(tf.keras.layers.Dense(1, activation=None))
        self._thumbnail_detector.build([None, thumbnail_size[0], thumbnail_size[1], 3])

    def fine_tune(dataloader, epochs):

        # This method will take the thumbnail dataloader class and train for some number
        # of epochs to fine tune the weights.
        # We need to figure out what fine tuning parameters to use.

        pass
