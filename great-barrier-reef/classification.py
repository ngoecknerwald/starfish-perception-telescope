# Final classification network and training wrapper

import tensorflow as tf


class Classifier(tf.keras.Model):
    def __init__(
        self,
        n_proposals,
        input_feature_size=7,
        dense_layers=4096,
        n_classes=2,
        dropout=0.2,
    ):
        '''
        Class for the Faster R-CNN output layers.

        Arguments:

        n_proposals : int
            Number of RoIs passed from the RPN to the output network.
        input_feature_size : int
            Dimension of the layers after the RoI pooling operation.
        dense_layers : int or list of int
            Number of fully connected neurons in the dense layers before
            the classification and bounding box regression steps.
        n_classes : int
            Number of classes (including background). For this application
            this will always be 2.
        dropout : float or None
            Dropout parameter to use between the dense and classification
            and bounding box regression layers.
        '''

        super().__init__()

        # Record for posterity
        self.n_proposals = n_proposals
        self.input_feature_size = input_feature_size
        self.dense_layers = (
            [dense_layers, dense_layers]
            if isinstance(dense_layers, int)
            else dense_layers
        )
        self.n_classes = n_classes
        self.dropout = dropout

        # Instantiate network components
        self.dense1 = tf.keras.layers.Dense(dense_layers[0], activation='relu')
        self.dense2 = tf.keras.layers.Dense(dense_layers[1], activation='relu')
        self.cls = tf.keras.layers.Dense(
            n_proposals * n_classes,
        )
        self.bbox = tf.keras.layers.Dense(n_proposals * n_classes * 4)
        self.dropout1 = tf.keras.layers.Dropout(0.2)

    def call(x, training=False):
        x = dense1(x)
        x = dense2(x)
        if training:
            x = self.dropout1(x)
        cls = self.cls(x)
        bbox = self.bbox(x)
        return cls, bbox


class ClassificationWrapper:
    def __init__(self):
        pass
