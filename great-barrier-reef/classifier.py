# Final classification network and training wrapper

import tensorflow as tf


class Classifier(tf.keras.Model):
    def __init__(
        self,
        n_proposals,
        dense_layers=1024,
        n_classes=2,
        dropout=0.2,
    ):
        """
        Class for the Faster R-CNN output network.

        Arguments:

        n_proposals : int
            Number of RoIs passed from the RPN to the output network.
        dense_layers : int or list of int
            Number of fully connected neurons in the dense layers before
            the classification and bounding box regression steps.
        n_classes : int
            Number of classes (including background). For this application
            this will always be 2.
        dropout : float or None
            Dropout parameter to use between the dense and classification
            and bounding box regression layers.
        """

        super().__init__()

        # Record for posterity
        self.n_proposals = n_proposals
        self.dense_layers = (
            [dense_layers, dense_layers]
            if isinstance(dense_layers, int)
            else dense_layers
        )
        self.n_classes = n_classes
        self.dropout = dropout

        # Instantiate network components
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(self.dense_layers[0], activation="relu")
        self.dense2 = tf.keras.layers.Dense(self.dense_layers[1], activation="relu")
        self.cls = tf.keras.layers.Dense(
            n_proposals * n_classes,
        )
        self.bbox = tf.keras.layers.Dense(n_proposals * n_classes * 4)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout)

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        if training:
            x = self.dropout1(x)
        cls = self.cls(x)
        bbox = self.bbox(x)
        return cls, bbox


class ClassifierWrapper:
    def __init__(
        self,
        backbone,
        n_proposals,
        dense_layers=1024,
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2, decay_steps=1000, decay_rate=0.9
        ),
        class_dropout=0.2,
    ):
        """
        Wrapper class for the final classification model.

        Arguments :

        backbone : subclass of backbone.Backbone()
            Used for geometry methods when taking a training step.
        n_proposals : int
            Number of proposals handed in from the pooled RoI
        dense_layers : int
            Number of neurons in the dense network.
        learning_rate : float or tf.keras.optimizers.schedules.LearningRateSchedule
            Learning rate when tuning only the classification layers.
        class_dropout : float between 0 and 1
            Dropout parameter to use when training the classification layers.

        """

        # Record for posterity
        self.backbone = backbone
        self.n_proposals = n_proposals
        self.learning_rate = learning_rate
        self.dense_layers = dense_layers

        # Network and optimizer
        self.classifier = Classifier(
            n_proposals,
            dropout=class_dropout,
            dense_layers=dense_layers,
        )
        self.optimizer = tf.keras.optimizers.SGD(self.learning_rate, momentum=0.9)

    def save_classifier_state(self, filename):
        """
        Save the trained RPN state.

        Arguments:

        path: str
            Save path for the classifier.
        """

        tf.keras.models.save_model(self.classifier, filename)

    def load_classifier_state(self, filename):
        """
        Load the trained RPN state.

        path: str
            Load path for the classifier.
        """

        self.classifier = tf.keras.models.load_model(filename)

    def training_step(self, features, roi, label_x):

        """
        Take a training step with the classification network.

        Arguments:

        features : tf.tensor
            Image convolved by the backbone.
        roi : tf.tensor
            Slice of features output by the RoI pooling operation
        label : list of dict
            Decoded grond truth labels for the training minibatch.

        """

        print("Inside ClassWrapper.training_step()")
        print(features.shape)
        print(roi.shape)
        print(label_x)

        with tf.GradientTape() as tape:
            cls, bbox = self.classifier(features)

            print(cls.shape)
            print(bbox.shape)
            loss = tf.reduce_sum(cls) + tf.reduce_sum(bbox)

        # Compute gradients
        gradients = tape.gradient(loss, self.classifier.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(
            (grad, var)
            for grad, var in zip(gradients, self.classifier.trainable_variables)
            if grad is not None
        )
