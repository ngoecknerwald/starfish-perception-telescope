# Final classification network and training wrapper

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import geometry


class Classifier(tf.keras.layers.Layer):
    def __init__(
        self,
        n_proposals,
        dropout,
        dense_layers,
        n_classes=2,
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

        print("Python interpreter in Classifier.call()")

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
        rpnwrapper,
        roi_pool,
        label_decoder,
        n_proposals,
        augmentation_params,
        dense_layers=512,
        class_dropout=0.5,
        negative_weight=0.1, 
    ):
        """
        Wrapper class for the final classification model.

        Arguments :

        backbone : subclass of backbone.Backbone()
            Used for geometry methods when taking a training step.
        n_proposals : int
            Number of proposals handed in from the pooled RoI
        training_params : dict
            Augmentation parameters to use in *image space* before image preprocessing.
        dense_layers : int
            Number of neurons in the dense network.
        learning_rate : float or tf.keras.optimizers.schedules.LearningRateSchedule
            Learning rate when tuning only the classification layers.
        class_dropout : float between 0 and 1
            Dropout parameter to use when training the classification layers.
        negative weight : float
            Relative weight of negative RoIs. Multiplies the classification loss. 
        """

        super().__init__()

        # Record for posterity
        self.backbone = backbone
        self.rpnwrapper = rpnwrapper
        self.roi_pool = roi_pool
        self.label_decoder = label_decoder
        self.n_proposals = n_proposals
        self.dense_layers = dense_layers
        self.augmentation_params = augmentation_params
        self.negative_weight = negative_weight
        # Network and optimizer
        self.classifier = Classifier(
            n_proposals,
            class_dropout,
            dense_layers,
        )

        # Loss calculations
        self.class_loss = tf.keras.losses.CategoricalCrossentropy()
        self.bbox_reg_l1 = tf.keras.losses.Huber()

        self.augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomZoom(self.augmentation_params["zoom"]),
                tf.keras.layers.RandomRotation(self.augmentation_params["rotation"]),
                tf.keras.layers.GaussianNoise(self.augmentation_params["gaussian"]),
                tf.keras.layers.RandomContrast(self.augmentation_params["contrast"]),
            ]
        )

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
            Note that the RoI should be in *feature coordinates*, not image coordinates.

        """

        print("Python interpreter in classifier._compute_loss()")

        # No batch dimensions in this function, called with map_fn

        # Unpack and cast RoI to float for later IoU calculations
        cls, bbox, roi, labels = data

        # Conver to image coordinates
        roi = tf.cast(roi, tf.float32)
        x, y = self.backbone.feature_coords_to_image_coords(roi[:, 0], roi[:, 1])
        w, h = self.backbone.feature_coords_to_image_coords(roi[:, 2], roi[:, 3])
        roi = tf.stack([x, y, w, h], axis=0)

        # Figure out if there is a starfish or not
        starfish = labels[tf.math.count_nonzero(labels, axis=1) > 0]

        def _calc_IoU(sf):
            return geometry.calculate_IoU(sf, roi)  # returns (nroi,) tensor

        IoUs = tf.map_fn(_calc_IoU, starfish)  # returns (nstarfish, nroi) tensor

        # For each starfish, grab the highest IoU roi
        match = tf.math.argmax(IoUs, axis=1)  # returns (nstarfish, ) tensor

        # Check if the match is a real max or the first of all zeros
        check_match = tf.math.count_nonzero(IoUs, axis=1) > 0

        # Set index to -1 for false matches. This won't equal any index in range(proposals),
        # so -1 means "not in range" here rather than "last index"
        match = tf.where(check_match, match, -1)

        # First the regularization term, turned down to match what's in the RPN
        # This regularization is on the outputs of the classifier network, not weights
        # which is done implicitly by the SGDW optimizer
        loss = tf.nn.l2_loss(cls) / (100.0 * tf.size(cls, out_type=tf.float32))
        loss += tf.nn.l2_loss(bbox) / (10.0 * tf.size(bbox, out_type=tf.float32))

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
            cls_select = tf.nn.softmax(cls[i :: self.n_proposals])
            if positive:
                ground_truth = tf.constant([0.0, 1.0])
                loss += _bbox_loss(i)
                loss += self.class_loss(cls_select, ground_truth)
            else:
                ground_truth = tf.constant([1.0, 0.0])
                loss += self.negative_weight * self.class_loss(cls_select, ground_truth)

        return loss

    @tf.function
    def train_step(self, data):

        """
        Take a training step with the classification network.

        Arguments:

        data : (tf.tensor, labels)
            Packed images and labels.

        """

        # Run the data augmentation
        data_aug = self.augmentation(data[0])

        # Run the feature extractor
        features = self.backbone(data_aug)

        # Accumulate RoI data
        labels = self.label_decoder(data[1])

        # Loop over images accumulating RoI proposals
        features, roi = self.roi_pool(
            (
                features,
                self.rpnwrapper.propose_regions(
                    features, input_images=False, output_images=False
                ),
            )
        )

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

        self.compiled_metrics.update_state(data[1], self.call((features, roi)))
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

    @tf.function
    def test_step(self, data):

        """
        Take a test step with the classification network.

        Arguments:

        data : (tf.tensor, labels)
            Packed images and labels.

        """

        # Run the feature extractor
        features = self.backbone(data[0])

        # Accumulate RoI data
        labels = self.label_decoder(data[1])

        # Loop over images accumulating RoI proposals
        features, roi = self.roi_pool(
            (
                features,
                self.rpnwrapper.propose_regions(
                    features, input_images=False, output_images=False
                ),
            )
        )

        # Classification layer forward pass
        cls, bbox = self.classifier(features, training=False)
        loss = tf.reduce_sum(
            tf.map_fn(
                self._compute_loss,
                [cls, bbox, roi, labels],
                fn_output_signature=(tf.float32),
            )
        )

        self.compiled_metrics.update_state(data[1], self.call((features, roi)))
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

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
