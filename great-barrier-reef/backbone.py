# This class will contain the backbone module
import tensorflow as tf
import tensorflow_hub as hub

# NOTE, from here on all variables denoted xx, yy, hh, ww refer to *feature space*
# and all variables denoted x, y, h, w denote *image space*


def instantiate(backbone_type, init_args):
    '''
    Create a subclass of Backbone with type backbone_type
    and constructor arguments **init_args.

    Arguments:

    backbone_type : str
        One of the defined backbones, 'InceptionResNet-V2', 'VGG16', 'ResNet50'
    init_args : dict
        Keywords to pass to the backbone constructor.
    '''

    backbone_type = backbone_type.lower()

    if backbone_type == 'inceptionresnet-v2':
        return Backbone_InceptionResNetV2(**init_args)
    elif backbone_type == 'vgg16':
        return Backbone_VGG16(**init_args)
    elif backbone_type == 'resnet50':
        return Backbone_ResNet50(**init_args)
    else:
        raise ValueError('Argument backbone=%s is not recognized.' % backbone_type)


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

        tf.keras.models.save_model(self.network, path)

    def load_backbone(self, path):
        """
        Load the tuned backbone from a path.

        path: str
            Load path for the (tuned) backbone model.
        """

        local_network = tf.keras.models.load_model(path)
        self.network.set_weights(local_network.get_weights())
        del local_network

    def feature_coords_to_image_coords(self, xx, yy):
        """
        Naively maps coordinates x,y in extracted feature space to
        coordinates in map space.

        Arguments:

        xx : int or numpy array
            Pixel coordinate in the feature map.
        yy : int or numpy array
            Pixel corrdinate in the feature map.
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
        """

        return (
            x * float(self.output_shape[1] / self.input_shape[1]),
            y * float(self.output_shape[0] / self.input_shape[0]),
        )

    def pretrain(
        self,
        training_data,
        validation_data=None,
        optimizer='adam',
        epochs=[2, 10],
        learning_rates=[1e-3, 1e-6],
        optimizer_kwargs={},
        fit_kwargs={},
        return_history=False,
    ):
        """
        Pretrain a backbone to do classification on starfish / not starfish thumbnails.
        Requires that self.extractor be an instantiated valid keras model and trains
        on the thumbnail classification task.

        Arguments:

        training_data : data_utils.DataLoaderThumbnail.get_training()
            Training dataset. Handles file I/O and batching.
        validation_data : data_utils.DataLoaderThumbnail.get_validation() or None
            Validation dataset. Pass in None to train on the full dataset for the production run.
        optimizer : string or tf.keras.optimizer.Optimizer subclass
            Either the name of an optimizer ('adam') or an optimizer known to TensorFlow.
            If the optimizer is already instantiated learning_rates[1] is ignored.
        epochs : tuple of int
            Number of epochs to fine tune. First int is the number of epochs to train only the
            classification layer the second int is for fine tuning the full backbone.
        learning_rates : tuple of float
            Two learning rates, the first to initially train the classification layers
            and the second to fine tune the whole backbone. The second is usually much smaller.
            I have found good results fine tuning Inception ResNetv2 with 1e-6.
        optimizer_kwargs : dict
            Set of keyword arguments to pass to the optimizer for tuning the backbone itself.
        fit_kwargs : dict
            Set of keyword arguments to pass to the fine tuning fit call.
        return_history : bool
            Return the training history
        """

        # Check to make sure the backbone has actually been instantiated
        assert isinstance(self.extractor, tf.keras.Model)

        # First, freeze the feature extractor and train the classification layers
        self.extractor.trainable = False

        # It's a plain stack of layers, so let's just use Sequential for readability
        # TODO do we want to put the dropout before or after the GAP2D? We originally
        # had dropout before the GAP but it makes more sense after in my head. Will revisit later.
        model = tf.keras.Sequential(
            [
                self.extractor,
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1),
            ]
        )

        # Compile the model with a fixed optimizer,
        # only pass in a learning rate and number of epochs
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rates[0]),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
            if validation_data is not None
            else None,
        )
        classify = model.fit(
            training_data,
            epochs=epochs[0],
            validation_data=validation_data,
        )

        # Now unfreeze extractor
        self.extractor.trainable = True

        # Define the fine-tuning optimizer, use their deserialization
        # methods instead of writing our own
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            tf.keras.optimizers.deserialize(
                {
                    'class_name': optimizer,
                    'config': {
                        'learning_rate': learning_rates[1],
                        **optimizer_kwargs,
                    },
                }
            )

        # Recompile with new optimizer
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
            if validation_data is not None
            else None,
        )

        # And fine-tune
        finetune = model.fit(
            training_data,
            epochs=epochs[1],
            validation_data=validation_data,
            **fit_kwargs
        )

        # Now freeze the weights again
        self.extractor.trainable = False

        # Return the training history if requested
        if return_history:
            return classify.history, finetune.history


class Backbone_InceptionResNetV2(Backbone):

    # Start by downloading pretrained weights from the Tensorflow hub
    def __init__(self, input_shape=(720, 1280, 3), weights='imagenet', **kwargs):
        """
        Initialize the network backbone. Downloads the Inception Resnet V2 pretrained on ImageNet.

        Arguments:

        input_shape: tuple
            Shape of the input images.
        weights : str or None
            Set of weights to use initially.
        """

        super().__init__()

        # Feature extractor,
        self.network = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False,
            weights=weights,
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )

        # The things connected to this model will need to know output geometry
        self.input_shape = input_shape
        self.output_shape = self.network.output_shape[1:]

        # Fold the image preprocessing into the model
        self.extractor = tf.keras.Sequential(
            [
                tf.keras.layers.Lambda(
                    tf.keras.applications.inception_resnet_v2.preprocess_input
                ),
                self.network,
            ]
        )


class Backbone_VGG16(Backbone):
    def __init__(self, input_shape=(720, 1280, 3), weights='imagenet', **kwargs):
        """
        Same arguments as Backbone_InceptionResNetV2,
        but using the smaller VGG16 network to speed up training.

        Arguments:

        input_shape: tuple
            Shape of the input images.
        weights : str or None
            Set of weights to use initially.
        """

        super().__init__()

        self.network = tf.keras.applications.vgg16.VGG16(
            include_top=False,
            weights=weights,
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )

        self.input_shape = input_shape
        self.output_shape = self.network.output_shape[1:]

        # Fold the image preprocessing into the model
        # The different pretrained models expect different inputs, so propagate that into here
        self.extractor = tf.keras.Sequential(
            [
                tf.keras.layers.Lambda(tf.keras.applications.vgg16.preprocess_input),
                self.network,
            ]
        )


class Backbone_ResNet50(Backbone):
    def __init__(self, input_shape=(720, 1280, 3), weights='imagenet', **kwargs):
        """
        Same arguments as Backbone_InceptionResNetV2,
        but using the base ResNet50 network. Trains and runs somewhat faster.

        Arguments:

        input_shape: tuple
            Shape of the input images.
        weights : str or None
            Set of weights to use initially.
        """

        super().__init__()

        self.network = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            weights=weights,
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
        )

        self.input_shape = input_shape
        self.output_shape = self.network.output_shape[1:]

        # Fold the image preprocessing into the model
        # The different pretrained models expect different inputs, so propagate that into here
        self.extractor = tf.keras.Sequential(
            [
                tf.keras.layers.Lambda(tf.keras.applications.resnet.preprocess_input),
                self.network,
            ]
        )
