# This class will contain the backbone module
import tensorflow as tf
import tensorflow_hub as hub


class Backbone:
    def __init__(self):
        '''
        Superclass for different backbone models.
        '''
        self.extractor = None
        self.input_shape = None
        self.output_shape = None

    def save_backbone(self, path):
        '''
        Save the trained convolutional layers of the backbone to a file path.

        Arguments:

        path: str
            Save path for the (tuned) backbone model.
        '''

        tf.saved_model.save(self.extractor, path)

    def load_backbone(self, path):
        '''
        Load the tuned backbone from a path.

        path: str
            Load path for the (tuned) backbone model.
        '''

        self.extractor = tf.saved_model.load(path)


class Backbone_InceptionResNetV2(Backbone):

    # Start by downloading pretrained weights from the Tensorflow hub
    def __init__(self, input_shape=(720, 1280, 3), **kwargs):
        '''
        Initialize the network backbone. Downloads the Inception Resnet V2 pretrained on ImageNet.

        Arguments:

        input_shape: tuple
            Shape of the input images.
        '''

        super().__init__()

        # Feature extractor,
        self.extractor = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )

        # The things connected to this model will need to know output geometry
        self.input_shape = input_shape
        self.output_shape = self.extractor.output_shape[1:]
