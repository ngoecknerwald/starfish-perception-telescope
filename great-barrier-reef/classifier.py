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
        n_proposals,
        dense_layers=512,
        learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[
                10000,
            ],
            values=[1e-3, 1e-4],
        ),
        weight_decay=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[
                10000,
            ],
            values=[1e-4, 1e-5],
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

        super().__init__()

        # Record for posterity
        self.backbone = backbone
        self.n_proposals = n_proposals
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dense_layers = dense_layers

        # Network and optimizer
        self.classifier = Classifier(
            n_proposals,
            dropout=class_dropout,
            dense_layers=dense_layers,
        )
        self.optimizer = tfa.optimizers.SGDW(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0.9,
            clipvalue=1e2,
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

        #NO BATCH DIMENSION IN THIS FUNCTION
        #INTENDED TO BE CALLED WITH MAP_FN

        cls, bbox, roi, labels = data

        roi = tf.cast(roi, tf.float32)
        # Coordinates and area of the proposed region
        x, y = self.backbone.feature_coords_to_image_coords(roi[:, 0], roi[:, 1])
        w, h = self.backbone.feature_coords_to_image_coords(roi[:, 2], roi[:, 3])

        roi = tf.stack([x,y,w,h], axis=-1)

        starfish = labels[tf.math.count_nonzero(labels, axis = 1) > 0]

        def _calc_IoU(sf):
            return geometry.calculate_IoU(sf, roi) # returns (nroi,) tensor

        IoUs = tf.map_fn(_calc_IoU, starfish) # returns (nstarfish, nroi) tensor

        # for each starfish, grab the highest IoU roi
        match = tf.math.argmax(IoUs, axis=1) # returns (nstarfish, ) tensor

        # First the regularization term
        loss = tf.nn.l2_loss(cls) / (10.0 * tf.size(cls, out_type=tf.float32))
        loss += tf.nn.l2_loss(bbox) / tf.size(bbox, out_type=tf.float32)

        def _bbox_loss(idx):

            truth_box = starfish[tf.squeeze(tf.where(match == idx))]
            t_x_star = (truth_box[0] - roi[idx][0]) / roi[idx][0]
            t_y_star = (truth_box[1] - roi[idx][1]) / roi[idx][1]
            t_w_star = geometry.safe_log(truth_box[2] / roi[idx][2])
            t_h_star = geometry.safe_log(truth_box[3] / roi[idx][3])
            return self.bbox_reg_l1(
                        [t_x_star, t_y_star, t_w_star, t_h_star],
                        bbox[idx :: self.n_proposals],
                    )

        for i in tf.range(self.n_proposals, dtype=tf.int64):
            positive = tf.reduce_any(tf.math.equal(i, match))
            ground_truth = tf.cond(positive, lambda: tf.constant([0.0, 0.1]), lambda: tf.constant([1.0, 0.0]))            
            cls_select = tf.nn.softmax(cls[i :: self.n_proposals])
            loss += self.class_loss(cls_select , ground_truth)
            loss += tf.cond(positive, lambda: _bbox_loss(i), lambda: tf.constant(0.0))
       
        return {"loss": loss}

    def train_step(
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

    def call(self, features, roi):
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
