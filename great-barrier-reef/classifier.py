# Final classification network and training wrapper

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import geometry


class Classifier(tf.keras.Model):
    def __init__(
        self,
        n_proposals,
        dense_layers=512,
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
        self.bbox = tf.keras.layers.Dense(n_proposals * 4)
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
        dense_layers=512,
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9
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
        self.optimizer = tfa.optimizers.SGDW(
            self.learning_rate, weight_decay=1e-5, momentum=0.9
        )

        # Loss calculations
        self.class_loss = tf.keras.losses.CategoricalCrossentropy()
        self.bbox_reg_l1 = tf.keras.losses.Huber()
        self.bbox_reg_giou = tfa.losses.GIoULoss()

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

    def compute_loss(self, roi, label_x, cls, bbox):
        """
        Compute the loss term for the full network.

        Arguments:

        roi : tf.Tensor or np.ndarray
            Tensor of dimension [n_images, n_rois, xywh] encoding the position of the RoI.
        label_x : list of dict
            Decoded ground truth annotations.
        cls : tf.Tensor
            Classification score of dimension [n_images, n_rois * n_classes = 2]
            encoded the same was as the RPN.
        bbox : tf.Tensor
            Bounding box regression of dimension [n_images, n_rois * 4]
            encoded the same way as the RPN.

        """

        # Stop the training if we hit nan values
        if np.any(np.logical_not(np.isfinite(cls.numpy()))):
            raise ValueError("NaN detected in the classifier, aborting training.")

        # Preliminaries, assign region proposals to ground truth boxes
        ground_truth_match = -1 * np.ones((roi.shape[0], roi.shape[1]), dtype=int)

        # Coordinates and area of the proposed region
        x, y = self.backbone.feature_coords_to_image_coords(roi[:, :, 0], roi[:, :, 1])
        w, h = self.backbone.feature_coords_to_image_coords(roi[:, :, 2], roi[:, :, 3])

        # Work one image at a time, noting that this short circuits if there is no label
        for i_image, image_labels in enumerate(label_x):

            if len(image_labels) == 0:
                continue

            IoUs = np.stack(
                [
                    geometry.calculate_IoU(
                        np.asarray(
                            [x[i_image, :], y[i_image, :], w[i_image, :], h[i_image, :]]
                        ),
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

            del IoUs

        # First the regularization term
        loss = tf.nn.l2_loss(cls) / (10.0 * tf.size(cls, out_type=tf.float32))
        loss += tf.nn.l2_loss(bbox) / tf.size(bbox, out_type=tf.float32)

        # Binary loss term, encoded the same way as the RPN classification
        loss += self.class_loss(
            np.hstack([ground_truth_match < 0, ground_truth_match > 0]).astype(int),
            cls,
        )

        # Go through the region proposals and reject the ones not associated with a ground truth box
        for i_image in range(roi.shape[0]):
            for i_roi in range(roi.shape[1]):

                if ground_truth_match[i_image, i_roi] > -1:

                    # Bounding box coords and the ground truth
                    this_roi = label_x[i_image][ground_truth_match[i_image, iroi]]

                    # Huber loss, which AFAIK is the same as smooth L1
                    t_x_star = (x[i_image, i_roi] - this_roi["x"]) / (w[i_image, i_roi])
                    t_y_star = (y[i_image, i_roi] - this_roi["y"]) / (h[i_image, i_roi])
                    t_w_star = geometry.safe_log(
                        this_roi["width"] / (w[i_image, i_roi])
                    )
                    t_h_star = geometry.safe_log(
                        this_roi["height"] / (h[i_image, i_roi])
                    )

                    loss += self.bbox_reg_l1(
                        [t_x_star, t_y_star, t_w_star, t_h_star],
                        bbox[i_image, i_roi :: roi.shape[1]],
                    )

        return loss

    def training_step(
        self,
        features,
        roi,
        label_x,
    ):

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

        # Classification layer forward pass
        with tf.GradientTape() as tape:

            cls, bbox = self.classifier(features, training=True)
            loss = self.compute_loss(roi, label_x, cls, bbox)

        # Compute gradients
        gradients = tape.gradient(loss, self.classifier.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(
            (grad, var)
            for grad, var in zip(gradients, self.classifier.trainable_variables)
            if grad is not None
        )

    def predict_classes(self, features, roi):
        """
        Run the final prediction to map

        features : tf.tensor
            RoI pooled features for a minibatch.
        roi : tf.tensor
            Slice of features output by the RoI pooling operation
        label_x : list of dict
            Decoded grond truth labels for the training minibatch.

        """

        cls, bbox = self.classifier(features)

        # Same as rpnwrapper.propose_regions()
        objectness_l0 = cls[:, : features.shape[1]].numpy()
        objectness_l1 = cls[:, features.shape[1] :].numpy()

        # Need to unpack a bit and hit with softmax
        objectness_l0 = objectness_l0.reshape((objectness_l0.shape[0], -1))
        objectness_l1 = objectness_l1.reshape((objectness_l1.shape[0], -1))
        objectness = tf.nn.softmax(
            np.stack([objectness_l0, objectness_l1]), axis=0
        ).numpy()

        # Cut to the positive bit
        objectness = objectness[1, :, :]

        xx = roi[:, :, 0] - (bbox[:, : roi.shape[1]].numpy() * roi[:, :, 2])
        yy = roi[:, :, 1] - (
            bbox[:, roi.shape[1] : 2 * roi.shape[1]].numpy() * roi[:, :, 3]
        )
        ww = (
            roi[:, :, 2]
            * geometry.safe_exp(bbox[:, 2 * roi.shape[1] : 3 * roi.shape[1]]).numpy()
        )
        hh = (
            roi[:, :, 3]
            * geometry.safe_exp(bbox[:, 3 * roi.shape[1] : 4 * roi.shape[1]]).numpy()
        )

        x, y = self.backbone.feature_coords_to_image_coords(xx, yy)
        w, h = self.backbone.feature_coords_to_image_coords(ww, hh)

        return [
            [
                {
                    "x": x[i_image, i_roi],
                    "y": y[i_image, i_roi],
                    "width": w[i_image, i_roi],
                    "height": h[i_image, i_roi],
                    "score": objectness[i_image, i_roi],
                }
                for i_roi in range(roi.shape[1])
            ]
            for i_image in range(roi.shape[0])
        ]
