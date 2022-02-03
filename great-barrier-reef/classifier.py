# Final classification network and training wrapper

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import geometry


class Classifier(tf.keras.layers.Layer):
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
        dense_layers : int
            Number of channels in the convolutional layer feeding into the dense layers.
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
        self.dense_layers = dense_layers
        self.n_classes = n_classes
        self.dropout = dropout

        # Instantiate network components
        self.conv1 = tf.keras.layers.Conv2D(
            self.dense_layers, (1, 1), activation="relu"
        )
        self.flatten = tf.keras.layers.Flatten()
        self.cls = tf.keras.layers.Dense(
            n_proposals * n_classes,
        )
        self.bbox = tf.keras.layers.Dense(n_proposals * 4)
        if self.dropout is not None:
            self.dropout1 = tf.keras.layers.Dropout(self.dropout)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.flatten(x)
        if hasattr(self, "dropout1") and training:
            x = self.dropout1(x)
        cls = self.cls(x)
        bbox = self.bbox(x)
        return cls, bbox


class ClassifierModel(tf.keras.Model):
    def __init__(
        self,
        backbone,
        rpn,
        pool,
        label_decoder,
        n_proposals,
        dense_layers=512,
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

        super().__init__()

        # Record for posterity
        self.backbone = backbone
        self.rpn = rpn
        self.pool = pool
        self.label_decoder = label_decoder
        self.n_proposals = n_proposals
        self.dense_layers = dense_layers

        # Network and optimizer
        self.classifier = Classifier(
            n_proposals,
            dropout=class_dropout,
            dense_layers=dense_layers,
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

    @tf.function
    def _compute_loss(self, data):
        """
        Compute the loss term for the full network.
        Works on one image at a time.

        Arguments:

        data : (tf.tensor, tf.tensor, tf.tensor, tf.tensor)
            Packed classifier scores, bbox regressors, roi, and labels for this image.

        """

        # NO BATCH DIMENSION IN THIS FUNCTION
        # INTENDED TO BE CALLED WITH MAP_FN

        cls, bbox, roi, labels = data

        roi = tf.cast(roi, tf.float32)
        # Coordinates and area of the proposed region
        x, y = self.backbone.feature_coords_to_image_coords(roi[:, 0], roi[:, 1])
        w, h = self.backbone.feature_coords_to_image_coords(roi[:, 2], roi[:, 3])

        roi = tf.stack([x, y, w, h], axis=0)

        starfish = labels[tf.math.count_nonzero(labels, axis=1) > 0]

        def _calc_IoU(sf):
            return geometry.calculate_IoU(sf, roi)  # returns (nroi,) tensor

        IoUs = tf.map_fn(_calc_IoU, starfish)  # returns (nstarfish, nroi) tensor

        # for each starfish, grab the highest IoU roi
        match = tf.math.argmax(IoUs, axis=1)  # returns (nstarfish, ) tensor

        # First the regularization term
        loss = tf.nn.l2_loss(cls) / (10.0 * tf.size(cls, out_type=tf.float32))
        loss += tf.nn.l2_loss(bbox) / tf.size(bbox, out_type=tf.float32)

        def _bbox_loss(idx):
            k = tf.where(match == idx)
            if tf.size(k) == 0:
                return tf.constant(0.0, dtype=tf.float32)
            truth_box = starfish[k[0, 0]]
            t_x_star = (truth_box[0] - roi[0][idx]) / roi[2][idx]
            t_y_star = (truth_box[1] - roi[1][idx]) / roi[3][idx]
            t_w_star = geometry.safe_log(truth_box[2] / roi[2][idx])
            t_h_star = geometry.safe_log(truth_box[3] / roi[3][idx])
            return self.bbox_reg_l1(
                [t_x_star, t_y_star, t_w_star, t_h_star],
                bbox[idx :: self.n_proposals],
            )

        for i in tf.range(self.n_proposals, dtype=tf.int64):
            positive = tf.reduce_any(tf.math.equal(i, match))
            if positive:
                ground_truth = tf.constant([0.0, 0.1])
                loss += _bbox_loss(i)
            else:
                ground_truth = tf.constant([1.0, 0.0])
            cls_select = tf.nn.softmax(cls[i :: self.n_proposals])
            loss += self.class_loss(cls_select, ground_truth)

        return loss

    @tf.function
    def train_step(self, data):

        """
        Take a training step with the classification network.

        Arguments:

        data : (tf.tensor, labels)
            Packed images and labels.

        """

        # How are the interconnects in the dense layers created? In theory axis 0 should
        # be disconnected from the others, but does feature pixel (1, 4) in roi=3 talk to (2,3) in roi=7?
        # We should be careful about what the Flatten() on line 45 is actually doing.

        # Another option would be to pass individual images through the Classifier() instead of a minibatch
        # stack of 4, which would make the Flatten() treat the first axis
        # (i.e. RoI) as independent minibatch examples

        # The official faster R-CNN ties everything in an image together, perhaps to avoid duplicate proposals.
        # We can revisit this later.

        # Run the feature extractor
        features = self.backbone(data[0])

        # Accumulate RoI data
        labels = self.label_decoder(data[1])

        # Loop over images accumulating RoI proposals
        features, roi = self.pool((features, self.rpn.propose_regions(features)))

        # Classification layer forward pass
        with tf.GradientTape() as tape:

            cls, bbox = self.classifier(features, training=True)
            loss = tf.reduce_sum(
                tf.map_fn(
                    self._compute_loss,
                    [cls, bbox, roi, labels],
                    fn_output_signature=(tf.float32),
                )
            )

        # Compute gradients
        gradients = tape.gradient(loss, self.classifier.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(
            (grad, var)
            for grad, var in zip(gradients, self.classifier.trainable_variables)
            if grad is not None
        )

        return {"loss": loss}

    @tf.function
    def call(self, data):
        """
        Run the final prediction to map

        data : (tf.tensor, tf.tensor)
            Pooled features and  roi for this minibatch.

        """
        pooled_features, roi = data

        cls, bbox = self.classifier(pooled_features)

        # Same as rpnwrapper.propose_regions()
        objectness_l0 = cls[:, : self.n_proposals]
        objectness_l1 = cls[:, self.n_proposals :]

        # Need to unpack a bit and hit with softmax
        objectness = tf.nn.softmax(tf.stack([objectness_l0, objectness_l1]), axis=0)

        # Cut to the positive bit
        objectness = objectness[1]

        # Convert roi to image coordinates
        roi = tf.cast(roi, tf.float32)
        x, y = self.backbone.feature_coords_to_image_coords(roi[:, :, 0], roi[:, :, 1])
        w, h = self.backbone.feature_coords_to_image_coords(roi[:, :, 2], roi[:, :, 3])
        roi = tf.stack([x, y, w, h], axis=-1)

        bbox = tf.reshape(bbox, (-1, 4, self.n_proposals))
        bbox = tf.transpose(bbox, perm=[0, 2, 1])

        x = roi[:, :, 0] - (bbox[:, :, 0] * roi[:, :, 2])
        x = roi[:, :, 0] - (bbox[:, :, 1] * roi[:, :, 3])
        w = roi[:, :, 2] * geometry.safe_exp(bbox[:, :, 2])
        h = roi[:, :, 3] * geometry.safe_exp(bbox[:, :, 3])

        return tf.stack([x, y, w, h, objectness], axis=-1)

    def read_scores(data):
        """
        Convert network output to dicts of predicted locations.

        data : tensor of shape (None, self.n_proposals, 5)
        where the last axis containts (x,y,w,h,score).

        """

        x, y, w, h, score = tf.unstack(data, axis=-1)
        return [
            [
                {
                    "x": x[i_image, i_roi].numpy(),
                    "y": y[i_image, i_roi].numpy(),
                    "width": w[i_image, i_roi].numpy(),
                    "height": h[i_image, i_roi].numpy(),
                    "score": score[i_image, i_roi].numpy(),
                }
                for i_roi in range(self.n_proposals)
            ]
            for i_image in range(score.shape[0])
        ]
