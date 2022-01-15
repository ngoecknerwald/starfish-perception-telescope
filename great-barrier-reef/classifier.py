# Final classification network and training wrapper

import tensorflow as tf
import numpy as np
import geometry


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

        # Loss calculations
        self.class_loss = tf.keras.losses.CategoricalCrossentropy()

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

    def training_step(self, features, roi, label_x, update_backbone=False):

        """
        Take a training step with the classification network.

        Arguments:

        features : tf.tensor
            Image convolved by the backbone.
        roi : tf.tensor
            Slice of features output by the RoI pooling operation
        label_x : list of dict
            Decoded grond truth labels for the training minibatch.

        """

        # How are the interconnects in the dense layers created? In theory axis 0 should
        # be disconnected from the others, but does feature pixel (1, 4) in roi=3 talk to (2,3) in roi=7?
        # We should be careful about what the Flatten() on line 45 is actually doing.

        # Another option would be to pass individual images through the Classifier() instead of a minibatch
        # stack of 4, which would make the Flatten() treat the first axis
        # (i.e. RoI) as independent minibatch examples

        # The official faster R-CNN ties everything in an image together, perhaps to avoid duplicate proposals.
        # We can revisit this later.

        # For the record the inputs look like
        # print(features.shape)
        # (4, 10, 4, 4, 1536)
        # print(roi.shape)
        # (4, 10, 4)
        # print(label_x)
        # [[{'x': 442, 'y': 202, 'width': 31, 'height': 26}], [], [], []]

        # Preliminaries, assign region proposals to ground truth boxes
        ground_truth_match = -1 * np.ones((features.shape[0], features.shape[1]))

        # Work one image at a time, noting that this short circuits if there is no label
        for i_image, image_labels in enumerate(label_x):

            if len(image_labels) == 0:
                continue

            # Coordinates and area of the proposed region
            x, y = self.backbone.feature_coords_to_image_coords(
                roi[i_image, :, 1], roi[i_image, :, 0]
            )
            w, h = self.backbone.feature_coords_to_image_coords(
                roi[i_image, :, 3], roi[i_image, :, 2]
            )

            IoUs = np.stack(
                [
                    geometry.calculate_IoU(
                        np.asarray([x, y, w, h]),
                        np.asarray(
                            [
                                label["x"],
                                label["y"],
                                label["width"],
                                label["height"],
                            ]
                        ),
                    )
                    for i_label, label in enumerate(image_labels)
                ]
            )

            # Iterate over positive labels to match them
            for ilabel in range(len(image_labels)):

                # Grab the RoI with the largest IoU
                iroi = np.argmax(IoUs[ilabel, :])

                # If that's a match, then make the assignment and
                # ignore the RoI in subsequent matching
                if IoUs[ilabel, iroi] > 1e-2:
                    ground_truth_match[i_image, iroi] = ilabel
                    IoUs[:, iroi] = 0.0

        with tf.GradientTape() as tape:

            # Classification layer forward pass
            cls, bbox = self.classifier(features, training=True)

            # Binary loss term, encoded the same way as the RPN classification
            loss = self.class_loss(
                np.hstack([ground_truth_match > 0, ground_truth_match < 0]).astype(int),
                cls,
            )

            # TODO add the bounding box regression terms!
            # print(bbox.shape)
            # (4, 80)

        # Compute gradients
        gradients = tape.gradient(loss, self.classifier.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(
            (grad, var)
            for grad, var in zip(gradients, self.classifier.trainable_variables)
            if grad is not None
        )

    def predict_classes(self, image, roi, positive_thresh=0.5):
        """
        Return predictions for an image.
        """
