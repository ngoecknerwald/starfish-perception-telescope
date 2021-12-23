# This class will contain the backbone module
import tensorflow as tf
import tensorflow_hub as hub

# NOTE, from here on all variables denoted xx, yy, hh, ww refer to *feature space*
# and all variables denoted x, y, h, w denote *image space*


class Backbone:
    def __init__(self):
        """
        Superclass for different backbone models.
        """
        self.extractor = None
        self.input_shape = None
        self.output_shape = None

    def save_backbone(self, path):
        """
        Save the trained convolutional layers of the backbone to a file path.

        Arguments:

        path: str
            Save path for the (tuned) backbone model.
        """

        tf.saved_model.save(self.extractor, path)

    def load_backbone(self, path):
        """
        Load the tuned backbone from a path.

        path: str
            Load path for the (tuned) backbone model.
        """

        self.extractor = tf.saved_model.load(path)

    def feature_coords_to_image_coords(self, xx, yy):
        """
        Naively maps coordinates x,y in extracted feature space to
        coordinates in map space.

        Arguments:

        xx : int or numpy array
            Pixel coordinate in the feature map.
        yy : int or numpy array
            Pixel corrdinate in the feature map.

        TODO this probably isn't actually right, because of edge effect.
        Come back to this if the boxes are all systematically offset.
        """

        return (
            xx * float(self.input_shape[1] / self.output_shape[1]),
            yy * float(self.input_shape[0] / self.output_shape[0]),
        )

    def image_coords_to_feature_coords(self, x, y):
        """
        Naively map coordinates in image space to feature space.

        Arguments:

        x : int or numpy array
            Pixel coordinate in the image map.
        y : int or numpy array
            Pixel corrdinate in the image map.

        TODO this probably isn't actually right, because of edge effect.
        Come back to this if the boxes are all systematically offset.
        """

        return (
            x * float(self.output_shape[1] / self.input_shape[1]),
            y * float(self.output_shape[0] / self.input_shape[0]),
        )

    def pretrain(self, dataloader, epochs=10, optimizer='adam', optimizer_kwargs={}):
        """
        Pretrain a backbone to do classification on starfish / not starfish thumbnails.
        Requires that self.extractor be an instantiated valid keras model and trains
        on the thumbnail classification task.

        Arguments:

        dataloader : data_utils.DataLoaderThumbnail
            Thumbnail data loading class. Handles file I/O and batching.
        epochs : int
            Number of epochs to train the backbone.
        optimizer : string or tf.keras.optimizer
            Either the name of an optimizer ('adam') or an optimizer known to TensorFlow.
        optimizer_kwargs : dict
            Set of keyword arguments to pass to the optimizer.

        """

        # Check to make sure the backbone has actually been instantiated
        assert isinstance(self.extractor, tf.keras.Model)

        # Now we have to compile the model
        pass


class Backbone_InceptionResNetV2(Backbone):

    # Start by downloading pretrained weights from the Tensorflow hub
    def __init__(self, input_shape=(720, 1280, 3), **kwargs):
        """
        Initialize the network backbone. Downloads the Inception Resnet V2 pretrained on ImageNet.

        Arguments:

        input_shape: tuple
            Shape of the input images.
        """

        super().__init__()

        # Feature extractor,
        self.network = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )

        # The things connected to this model will need to know output geometry
        self.input_shape = input_shape
        self.output_shape = self.network.output_shape[1:]

        # Fold the image preprocessing into the model
        self.extractor = tf.keras.Sequential(
            [tf.keras.applications.inception_resnet_v2.preprocess_input, self.network]
        )


class Backbone_VGG16(Backbone):
    def __init__(self, input_shape=(720, 1280, 3), **kwargs):
        """
        Same arguments as Backbone_InceptionResNetV2,
        but using the smaller VGG16 network to speed up training.

        Arguments:

        input_shape: tuple
            Shape of the input images.
        """

        super().__init__()

        self.network = tf.keras.applications.vgg16.VGG16(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )

        self.input_shape = input_shape
        self.output_shape = self.network.output_shape[1:]

        # Fold the image preprocessing into the model
        # The different pretrained models expect different inputs, so propagate that into here
        self.extractor = tf.keras.Sequential(
            [tf.keras.applications.vgg16.preprocess_input, self.network]
        )
