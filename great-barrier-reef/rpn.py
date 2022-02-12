# RPN model and wrapper class

# NOTE, from here on all variables denoted xx, yy, hh, ww refer to *feature space*
# and all variables denoted x, y, h, w denote *image space*

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import geometry, evaluation, callback


class RPNLayer(tf.keras.layers.Layer):
    def __init__(self, k, kernel_size, anchor_stride, filters, dropout):
        """
        Class for the RPN consisting of a convolutional layer and two fully connected layers
        for "objectness" and bounding box regression.

        k : int
            Number of bounding box sizes, usually 9.
        kernel_size : int
            Kernel size for the first convolutional layer.
        anchor_stride : int
            Stride of the anchor in image space.
        filters: int
            Number of filters between the convolutional layer
            and the fully connected layers.
        dropout : float or None
            Add dropout to the RPN with this fraction. Pass None to skip.

        """

        super().__init__()
        self.k = k
        self.kernel_size = kernel_size
        self.anchor_stride = anchor_stride
        self.filters = filters
        self.dropout = dropout
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=kernel_size,
            strides=(anchor_stride, anchor_stride),
            activation="relu",
            padding="same",
        )
        if self.dropout is not None:
            self.dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.cls = tf.keras.layers.Conv2D(
            filters=2 * self.k,
            kernel_size=1,
            strides=(1, 1),
        )
        self.bbox = tf.keras.layers.Conv2D(
            filters=4 * self.k, kernel_size=1, strides=(1, 1)
        )
        self.outputs = None

    def call(self, x, training=False):

        x = self.conv1(x)
        if hasattr(self, "dropout1") and training:
            x = self.dropout1(x)
        cls = self.cls(x)
        bbox = self.bbox(x)
        return (cls, bbox)

    def get_config(self):
        return {
            "k": self.k,
            "kernel_size": self.kernel_size,
            "anchor_stride": self.anchor_stride,
            "filters": self.filters,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RPNModel(tf.keras.Model):
    def __init__(
        self,
        backbone,
        label_decoder,
        kernel_size,
        anchor_stride,
        window_sizes,
        filters,
        roi_minibatch_per_image,
        n_roi_output,
        IoU_neg_threshold,
        rpn_dropout,
        training_params,
    ):
        """
        Initialize the RPN model for pretraining.

        backbone : Backbone
            Knows input image and feature map sizes but is not directly trained here.
        label_decoder : tf.function
            Method to decode the training dataset labels.
        kernel_size : int
            Kernel size for the first convolutional layer in the RPN.
        anchor_stride : int
            Stride of the anchor in image space in the RPN.
        window_sizes : list of ints
            Width of the proposals to select, *in extracted feature space*. Bypasses
            aspect ratios with np.meshgrid().
        filters: int
            Number of filters between the convolutional layer
            and the fully connected layers in the RPN.
        roi_minibatch_per_image : int
            Number of RoIs to use ( = positive + negative) per image per gradient step in the RPN.
        n_roi_output : int
            Number of regions to propose in forward pass. Pass -1 to return all.
        IoU_neg_threshold : float
            IoU to declare that a region proposal is a negative example.
        rpn_dropout : float or None
            Add dropout to the RPN with this fraction. Pass None to skip.
        training_params : dict
            Augmentation parameters to use in *image space* before image preprocessing.
        """

        super().__init__()

        # Store the image size
        self.backbone = backbone
        self.label_decoder = label_decoder
        self.kernel_size = kernel_size
        self.anchor_stride = anchor_stride
        self.window_sizes = window_sizes
        self.filters = filters
        self.roi_minibatch_per_image = roi_minibatch_per_image
        self.n_roi_output = n_roi_output
        self.IoU_neg_threshold = IoU_neg_threshold
        self.rpn_dropout = rpn_dropout
        self.training_params = training_params

        # Anchor box sizes
        self._build_anchor_boxes()

        # Build the model
        self.rpn = RPNLayer(
            self.k,
            self.kernel_size,
            self.anchor_stride,
            self.filters,
            self.rpn_dropout,
        )

        # Classification loss
        self.objectness = tf.keras.losses.CategoricalCrossentropy()

        # Regression loss terms
        self.bbox_reg_l1 = tf.keras.losses.Huber()

        # Data augmentation
        self.augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomZoom(training_params["zoom"]),
                tf.keras.layers.RandomRotation(training_params["rotation"]),
                tf.keras.layers.GaussianNoise(training_params["gaussian"]),
                tf.keras.layers.RandomContrast(training_params["contrast"]),
            ]
        )

        # Same thing as the classifier, accumulate a running average of the number of
        # examples that are + and use that to veto the negative examples.
        # Assume we'll use O(10) minibatch samples per image, this will quickly adjust
        # from the initial setting
        self._positive = tf.Variable(100.0, trainable=False)
        self._negative = tf.Variable(100.0, trainable=False)

    @tf.function
    def test_step(self, data):
        """
        Shallow wrapper around the loss calculation
        to get a loss and metrics from the validation
        set during training.

        data : tuple
            tf.Tensor() containing images and integer labels.

        """

        # Run the feature extractor
        features = self.backbone(data[0])

        # Accumulate RoI data
        labels = self.label_decoder(data[1])
        rois = tf.map_fn(self._accumulate_roi, labels)

        # Call the RPN
        cls, bbox = self.rpn(features, training=False)

        # Compute the loss using the classification scores and bounding boxes
        loss = tf.reduce_sum(
            tf.map_fn(
                self._compute_loss,
                [cls, bbox, rois],
                fn_output_signature=(tf.float32),
            )
        )

        self.compiled_metrics.update_state(
            data[1], self.call(features, input_images=False, output_images=True)
        )
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

    # Two methods required by the tf.keras.Model interface,
    # the training step and the forward pass
    @tf.function
    def train_step(self, data):

        """
        Take a convolved feature map, compute RoI, and
        update the RPN comparing to the ground truth.

        Arguments:

        data : (tf.tensor, tf.ragged.constant)
            Packed images and labels for this minibatch.

        """

        # Run the data augmentation
        data_aug = self.augmentation(data[0])

        # Run the feature extractor
        features = self.backbone(data_aug)

        # Accumulate RoI data
        labels = self.label_decoder(data[1])

        # Loop over images accumulating RoI proposals
        rois = tf.map_fn(self._accumulate_roi, labels)

        # Compute loss
        with tf.GradientTape() as tape:

            # Call the RPN
            cls, bbox = self.rpn(features, training=True)

            # Compute the loss using the classification scores and bounding boxes
            loss = tf.reduce_sum(
                tf.map_fn(
                    self._compute_loss,
                    [cls, bbox, rois],
                    fn_output_signature=(tf.float32),
                )
            )

        gradients = tape.gradient(loss, self.rpn.trainable_variables)

        # Apply gradients in the optimizer. Note that TF will throw spurious warnings if gradients
        # don't exist for the bbox regression if there were no + examples in the minibatch
        # This isn't actually an error, so list comprehension around it

        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradients, self.rpn.trainable_variables)
            if grad is not None
        )

        self.compiled_metrics.update_state(
            data[1], self.call(features, input_images=False, output_images=True)
        )
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

    @tf.function
    def call(
        self,
        data,
        input_images=True,
        output_images=True,
    ):
        """
        Run the RPN in forward mode on a minibatch of images.
        This method is used to train the final classification network
        and to evaluate at test time.

        Arguments:

        data : tf.tensor dataset minibatch
            Set of image(s) or feature(s) to run through the network.
        input_images : bool
            Set to true to run the input through the backbone.
        output_images : bool
            Return regions in image coordinates, otherwise work entirely in feature space.

        Returns:

        if not output_images:
            Tensor of shape (batch_size, top, 4) with feature space
            coordinates (xx,yy,ww,hh)
        else:
            Tensor of shape (batch_size, top, 4) with image space
            coordinates (x,y,w,h)
        """

        # Run through the extractor if images
        if input_images:
            features = self.backbone(data)
        else:
            features = data

        # Run through the RPN
        cls, bbox = self.rpn(features)

        # Dimension is image, iyy, ixx, ik were ik is
        # [neg_k=0, neg_k=1, ... pos_k=0, pos_k=1...]
        objectness_l0 = cls[:, :, :, : self.k]
        objectness_l1 = cls[:, :, :, self.k :]

        # Need to unpack a bit and hit with softmax
        flatten = tf.keras.layers.Flatten()
        objectness_l0 = flatten(objectness_l0)
        objectness_l1 = flatten(objectness_l1)
        objectness = tf.nn.softmax(tf.stack([objectness_l0, objectness_l1]), axis=0)

        # Cut to the one-hot bit
        objectness = objectness[1, :, :]

        # Remove the invalid bounding boxes
        objectness = tf.math.multiply(
            objectness,
            tf.reshape(tf.cast(self.valid_mask, dtype=tf.float32), (-1,))[
                tf.newaxis, :
            ],
        )

        # Dimension for bbox is same as cls but ik follows
        # [t_x_k=0, t_x_k=1, ..., t_y_k=0, t_y_k=1,
        # ..., t_w_k=0, t_w_k=1, ..., t_h_k=0, t_h_k=1, ...]
        xx = self.anchor_xx[tf.newaxis, :, :, tf.newaxis] - (
            bbox[:, :, :, : self.k] * self.ww[tf.newaxis, tf.newaxis, tf.newaxis, :]
        )
        yy = self.anchor_yy[tf.newaxis, :, :, tf.newaxis] - (
            bbox[:, :, :, self.k : 2 * self.k]
            * self.hh[tf.newaxis, tf.newaxis, tf.newaxis, :]
        )
        ww = self.ww[tf.newaxis, tf.newaxis, tf.newaxis, :] * geometry.safe_exp(
            bbox[:, :, :, 2 * self.k : 3 * self.k]
        )
        hh = self.hh[tf.newaxis, tf.newaxis, tf.newaxis, :] * geometry.safe_exp(
            bbox[:, :, :, 3 * self.k : 4 * self.k]
        )

        # Reshape to flatten along proposal dimension within an image
        argsort = tf.argsort(objectness, axis=-1, direction="DESCENDING")

        # Sort things by objectness
        xx = geometry.batch_sort(flatten(xx), argsort, self.n_roi_output)
        yy = geometry.batch_sort(flatten(yy), argsort, self.n_roi_output)
        ww = geometry.batch_sort(flatten(ww), argsort, self.n_roi_output)
        hh = geometry.batch_sort(flatten(hh), argsort, self.n_roi_output)

        # Return in feature space
        if not output_images:
            output = tf.stack([xx, yy, ww, hh], axis=-1)
        else:
            # Convert to image plane
            x, y = self.backbone.feature_coords_to_image_coords(xx, yy)
            w, h = self.backbone.feature_coords_to_image_coords(ww, hh)
            output = tf.stack([x, y, w, h], axis=-1)

        return output

    @tf.function
    def _accumulate_roi(self, label):
        """
        Make a list of RoIs of positive and negative examples
        and their corresponding ground truth annotations.

        Note that RoI are not guaranteed to be valid. This is checked
        in the loss calculation step.

        Arguments:

        label : tf.tensor
            Tensor slice containing decoded labels for a single image

        Returns:

        tf.tensor of shape (self.roi_minibatch_per_image, 8)
            The first 4 coordinates are (ixx,iyy,ik,valid) for the RoI.
            The next  4 coordinates are (x,y,w,h) for the ground truth box,
            or (0.,0.,0.,0.) if the RoI is not associated with any ground truth.
        """

        rois = []

        # Helper to deal with broadcasting, compute the ground truth
        # IoU for the first roi_minibatch_per_image entries in the label
        def _anchor_IoU(ik):
            return self._ground_truth_IoU(
                label[: self.roi_minibatch_per_image, :],
                self.anchor_xx,
                self.anchor_yy,
                tf.cast(self.ww[ik], tf.float32) * tf.ones(self.anchor_xx.shape),
                tf.cast(self.hh[ik], tf.float32) * tf.ones(self.anchor_yy.shape),
            )

        # No clue why this works but tf.map_fn() doesn't
        # (ik, i_starfish, iyy, ixx)
        ground_truth_IoU = tf.stack(
            [_anchor_IoU(ik) for ik in tf.unstack(tf.range(self.k, dtype=tf.int32))]
        )

        # Work one starfish at a time
        for i_starfish in range(self.roi_minibatch_per_image):

            # No starfish, so pick something at random
            if tf.math.greater_equal(
                tf.constant(self.IoU_neg_threshold),
                tf.reduce_max(ground_truth_IoU[:, i_starfish, :, :]),
            ):

                rxx = tf.random.uniform(
                    shape=[], dtype=tf.int32, maxval=tf.shape(self.anchor_xx)[1]
                )
                ryy = tf.random.uniform(
                    shape=[], dtype=tf.int32, maxval=tf.shape(self.anchor_xx)[0]
                )
                rk = tf.random.uniform(shape=[], dtype=tf.int32, maxval=self.k)
                ground_truth = tf.constant([0.0, 0.0, 0.0, 0.0])

            else:

                pos_slice = tf.where(
                    ground_truth_IoU[:, i_starfish, :, :]
                    == tf.reduce_max(ground_truth_IoU[:, i_starfish, :, :])
                )

                rk = tf.cast(pos_slice[0, 0], tf.int32)
                rxx = tf.cast(pos_slice[0, 2], tf.int32)
                ryy = tf.cast(pos_slice[0, 1], tf.int32)
                ground_truth = label[i_starfish, :]

            rois.append(
                [
                    tf.cast(rxx, tf.float32),
                    tf.cast(ryy, tf.float32),
                    tf.cast(rk, tf.float32),
                    tf.cast(self.valid_mask[ryy, rxx, rk], tf.float32),
                    *tf.unstack(ground_truth),
                ]
            )

        return tf.convert_to_tensor(rois)

    @tf.function
    def _compute_loss(self, data):

        """
        Compute the loss function for a set of classification
        scores cls and bounding boxes bbox on the set of regions
        of interest RoIs.

        Arguments:

        cls : tensor, shape (iyy, ixx, 2 * ik)
            Slice of the output from the RPN corresponding to the
            classification (object-ness) field. From here on out
            the k ordering is [neg_k=0, neg_k=1, ... pos_k=0, pos_k=1...].
        bbox : tensor, shape (iyy, ixx, 4 * ik)
            Slice of the output from the RPN. The k dimension is ordered
            following a similar convention as cls:
            [t_x_k=0, t_x_k=1, ..., t_y_k=0, t_y_k=1,
            ..., t_w_k=0, t_w_k=1, ..., t_h_k=0, t_h_k=1, ...]
        rois : list or RoIs, [ixx,iyy,ik,valid,(x, y, w, h)]
            Output of accumulate_roi(), see training_step for use case.
        """

        cls, bbox, rois = data

        # Sanity check
        tf.debugging.assert_all_finite(cls, "NaN encountered in RPN training.")

        # No L2 regularization loss for now
        loss = 0.0

        # Count how many positive valid boxes we have
        n_positive = 0.0
        n_negative = 0.0

        # Work one RoI at a time
        for i in range(self.roi_minibatch_per_image):

            valid = tf.cast(rois[i, 3], tf.bool)

            if valid:

                # Decode and cast to ints to use as indices
                ixx = tf.cast(rois[i, 0], tf.int32)
                iyy = tf.cast(rois[i, 1], tf.int32)
                ik = tf.cast(rois[i, 2], tf.int32)
                roi_gt = rois[i, 4:]

                positive = tf.math.greater_equal(tf.reduce_sum(roi_gt), 0.001)

                # First, compute the categorical cross entropy objectness loss
                cls_select = tf.nn.softmax(cls[iyy, ixx, ik :: self.k])

                # Compare anchor to ground truth
                if positive:

                    ground_truth = tf.constant([0.0, 1.0])
                    loss += 2.0 * self.objectness(ground_truth, cls_select)

                    # Refer the corners of the bounding box back to image space
                    # Note that this assumes said mapping is linear.
                    x, y = self.backbone.feature_coords_to_image_coords(
                        self.anchor_xx[iyy, ixx],
                        self.anchor_yy[iyy, ixx],
                    )
                    w, h = self.backbone.feature_coords_to_image_coords(
                        self.ww[ik],
                        self.hh[ik],
                    )

                    t_x_star = (x - roi_gt[0]) / (w)
                    t_y_star = (y - roi_gt[1]) / (h)
                    t_w_star = geometry.safe_log(roi_gt[2] / (w))
                    t_h_star = geometry.safe_log(roi_gt[3] / (h))

                    # Huber loss, which AFAIK is the same as smooth L1
                    loss += self.bbox_reg_l1(
                        [t_x_star, t_y_star, t_w_star, t_h_star],
                        bbox[iyy, ixx, ik :: self.k],
                    )

                    n_positive += 1.0

                elif tf.math.greater(
                    self._positive, self._negative
                ):  # take a negative because we have enough positives
                    ground_truth = tf.constant([1.0, 0.0])
                    loss += self.objectness(ground_truth, cls_select)
                    n_negative += 1.0
                else:  # too many negatives, ignore this RoI
                    pass

        # Exponential moving average update
        self._positive.assign(0.99 * self._positive + n_positive)
        self._negative.assign(0.99 * self._negative + n_negative)

        return loss

    def _build_anchor_boxes(self):
        """
        Build the anchor box sizes in the feature space.

        """

        # Make the list of window sizes
        hh, ww = np.meshgrid(self.window_sizes, self.window_sizes)
        self.hh = tf.constant(hh.reshape(-1), dtype="float32")
        self.ww = tf.constant(ww.reshape(-1), dtype="float32")
        self.k = len(self.window_sizes) ** 2

        # Need to refer the anchor points back to the feature space
        anchor_xx, anchor_yy = tf.meshgrid(
            range(
                0,
                tf.cast(self.backbone._output_shape[1], dtype="int32"),
                self.anchor_stride,
            ),
            range(
                0,
                tf.cast(self.backbone._output_shape[0], dtype="int32"),
                self.anchor_stride,
            ),
        )
        self.anchor_xx = tf.cast(anchor_xx, dtype="float32")
        self.anchor_yy = tf.cast(anchor_yy, dtype="float32")

        # Mask off invalid RoI that cross the image boundary
        self.valid_mask = tf.constant(
            tf.math.logical_and(
                tf.math.logical_and(
                    self.anchor_xx[:, :, tf.newaxis] >= 0,
                    self.anchor_yy[:, :, tf.newaxis] >= 0,
                ),
                tf.math.logical_and(
                    self.anchor_xx[:, :, tf.newaxis]
                    + self.ww[tf.newaxis, tf.newaxis, :]
                    < self.backbone._output_shape[1],
                    self.anchor_yy[:, :, tf.newaxis]
                    + self.hh[tf.newaxis, tf.newaxis, :]
                    < self.backbone._output_shape[0],
                ),
            ),
            dtype="bool",
        )

    @tf.function
    def _ground_truth_IoU(self, annotations, xx, yy, ww, hh):
        """
        Compute the ground truth IoU for a set of boxes defined in feature space.
        Note that xx,yy refers to the bottom left corner of the box, not the middle.

        Arguments:

        annotations : slice of tf.ragged.constant()
            Tensor of shape (None, 4,) encoded [xywh] in image coordinates.
        xx : tf.constant
            Pixel coordinates in the feature map.
        yy : tf.constant
            Pixel coordinates in the feature map.
        ww : tf.constant
            Pixel coordinates in the feature map.
        hh : tf.constant
            Pixel coordinates in the feature map.

        Returns:

        IoU: list of numpy array
            Ground truth IoU for each annotation.

        """

        # Coordinates and area of the proposed region
        x, y = self.backbone.feature_coords_to_image_coords(xx, yy)
        w, h = self.backbone.feature_coords_to_image_coords(ww, hh)
        proposal_box = tf.stack([x, y, w, h])

        def get_IoU(annotation):
            return geometry.calculate_IoU(proposal_box, annotation)

        return tf.map_fn(get_IoU, annotations)


# Just like the old wrapper class, doesn't extend anything from keras
class RPNWrapper:
    def __init__(
        self,
        backbone,
        label_decoder,
        kernel_size=3,
        anchor_stride=1,
        window_sizes=[2.0, 4.0],
        filters=1024,
        roi_minibatch_per_image=16,  # Might as well catch everything, because we are ignoring RoI to balance
        n_roi_output=128,
        IoU_neg_threshold=0.01,
        rpn_dropout=0.5,
        learning_rate={
            "epochs": [1, 4, 7],
            "values": [1e-3, 1e-4, 1e-5],
        },
        weight_decay={
            "epochs": [1, 4, 7],
            "values": [1e-4, 1e-5, 1e-6],
        },
        momentum=0.9,
        clipvalue=1e3,
        top_n_recall=32,
        training_params={
            "zoom": 0.01,
            "rotation": 0.01,
            "gaussian": 5.0,
            "contrast": 0.25,
        },
    ):

        """
        Wrapper class for the RPN model

        backbone : Backbone
            Knows input image and feature map sizes but is not directly trained here.
        label_decoder : tf.function
            Method to decode the training dataset labels.
        kernel_size : int
            Kernel size for the first convolutional layer in the RPN.
        anchor_stride : int
            Stride of the anchor in image space in the RPN.
        window_sizes : list of ints
            Width of the proposals to select, *in extracted feature space*. Bypasses
            aspect ratios with np.meshgrid().
        filters: int
            Number of filters between the convolutional layer
            and the fully connected layers in the RPN.
        roi_minibatch_per_image : int
            Number of RoIs to use ( = positive + negative) per image per gradient step in the RPN.
        n_roi_output : int
            Number of regions to propose in forward pass. Pass -1 to return all.
        IoU_neg_threshold : float
            IoU to declare that a region proposal is a negative example. Something safely above
            floating point error.
        rpn_dropout : float or None
            Add dropout to the RPN with this fraction. Pass None to skip.
        learning_rate : dict
            {'epoch' : list of epoch number, 'rate' : list of learning rates}
        weight_decay : dict
            {'epoch' : list of epoch number, 'rate' : list of decay rates}
        momentum : float
            Momentum parameter for the SGDW optimizer.
        clipvalue : float
            Maximum allowable gradient for the SGDW optimizer.
        top_n_recall : int
            Number of regions to consider when computing the recall score for the training / validation set.
        training_params : dict
            Parameters to pass to the augmentation segment when training. The Gaussian noise augmentation
            and contrast are copied over from the backbone fine tuning. The translation and rotation
            should be small enough to not meaningfully break the matching of RoI to the ground truth boxes.
        """

        # RPN model itself
        self.rpnmodel = RPNModel(
            backbone,
            label_decoder,
            kernel_size,
            anchor_stride,
            window_sizes,
            filters,
            roi_minibatch_per_image,
            n_roi_output,
            IoU_neg_threshold,
            rpn_dropout,
            training_params,
        )

        # Optimizer parameters
        self.momentum = momentum
        self.clipvalue = clipvalue

        # Optimizer
        self.optimizer = tfa.optimizers.SGDW(
            learning_rate=learning_rate["values"][0],
            weight_decay=weight_decay["values"][0],
            momentum=self.momentum,
            clipvalue=self.clipvalue,
        )

        self.callbacks = [callback.LearningRateCallback(learning_rate, weight_decay)]

        # Recall metric for the RPN
        self.validation_recall = evaluation.TopNRegionsRecall(
            top_n_recall, label_decoder, name="top%d_recall" % top_n_recall
        )

    def train_rpn(self, train_dataset, valid_dataset=None, epochs=9, kwargs={}):
        """
        Main training loop iterating over a dataset.

        Arguments:

        train_dataset: tensorflow dataset
            Dataset of input images. Minibatch size and validation
            split are determined when this is created.
        epochs : int
            Number of epochs to run training for.

        """

        self.rpnmodel.compile(
            optimizer=self.optimizer,
            metrics=[
                self.validation_recall,
            ],
        )
        self.rpnmodel.fit(
            train_dataset,
            epochs=epochs,
            validation_data=valid_dataset,
            callbacks=self.callbacks,
            **kwargs
        )

    def save_rpn_state(self, filename):
        """
        Save the trained RPN state.

        Arguments:

        path: str
            Save path for the RPN model.
        """

        tf.keras.models.save_model(self.rpnmodel.rpn, filename)

    def load_rpn_state(self, filename):
        """
        Load the trained RPN state.

        path: str
            Load path for the RPN model.
        """

        localmodel = tf.keras.models.load_model(filename)
        self.rpnmodel.rpn.set_weights(localmodel.get_weights())
        del localmodel

    def propose_regions(self, images, **kwargs):
        """
        Run the RPN in forward mode.

        Arguments:

        images : tf.tensor
            Minibatch of images
        **kwargs
            Arguments to be handed down to RPNModel.call()
        """

        return self.rpnmodel(images, **kwargs)
