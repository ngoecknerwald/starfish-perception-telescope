# RPN model and wrapper class

# NOTE, from here on all variables denoted xx, yy, hh, ww refer to *feature space*
# and all variables denoted x, y, h, w denote *image space*

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import warnings
import geometry

# Number of tries to find valid RoIs. Note that if this fails it is still
# possible to successfully build a minibatch
CUTOFF = 100


class RPN(tf.keras.Model):
    def __init__(self, k, kernel_size, anchor_stride, filters, dropout=0.2):
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
        self.cls_bbox = tf.keras.layers.Conv2D(
            filters=6 * self.k, kernel_size=1, strides=(1, 1)
        )

    # This needs to return a single variable to hand to the loss
    # when we compile the model, so just concatenate cls and bbox
    def call(self, x, training=False):
        x = self.conv1(x)
        if hasattr(self, "dropout1") and training:
            x = self.dropout1(x)
        return self.cls_bbox(x)


class RPNWrapper:
    def __init__(
        self,
        backbone,
        kernel_size=3,
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9
        ),
        anchor_stride=1,
        window_sizes=[2, 4],  # these must be divisible by 2
        filters=512,
        rpn_minibatch=16,
        IoU_neg_threshold=0.1,
        IoU_pos_threshold=0.7,
        rpn_dropout=0.2,
        n_roi=100,
    ):
        """
        Initialize the RPN model for pretraining.

        backbone : Backbone
            Knows input image and feature map sizes but is not directly trained here.
        kernel_size : int
            Kernel size for the first convolutional layer in the RPN.
        learning_rate : float or tf.keras.optimizers.schedules.LearningRateSchedule
            Learning rate for the SGD optimizer.
        anchor_stride : int
            Stride of the anchor in image space in the RPN.
        window_sizes : list of ints
            Width of the proposals to select, *in extracted feature space*. Bypasses
            aspect ratios with np.meshgrid().
        filters: int
            Number of filters between the convolutional layer
            and the fully connected layers in the RPN.
        rpn_minibatch : int
            Number of RoIs to use ( = positive + negative) per gradient step in the RPN.
        IoU_neg_threshold : float
            IoU to declare that a region proposal is a negative example.
        IoU_pos_threshold : float
            IoU to declare that a region proposal is a positive example.
        rpn_dropout : float or None
            Add dropout to the RPN with this fraction. Pass None to skip.
        n_roi : int
            Number of regions to propose in forward pass. Pass -1 to return all.
        """

        # Store the image size
        self.backbone = backbone
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.anchor_stride = anchor_stride
        self.window_sizes = window_sizes
        self.filters = filters
        self.rpn_minibatch = rpn_minibatch
        self.IoU_neg_threshold = IoU_neg_threshold
        self.IoU_pos_threshold = IoU_pos_threshold
        self.rpn_dropout = rpn_dropout
        self.n_roi = n_roi

        # Anchor box sizes
        self._build_anchor_boxes()

        # Build the model
        self.rpn = RPN(
            self.k,
            self.kernel_size,
            self.anchor_stride,
            self.filters,
            dropout=self.rpn_dropout,
        )

        # Optimizer
        self.optimizer = tfa.optimizers.SGDW(
            learning_rate=self.learning_rate, weight_decay=1e-4, momentum=0.9
        )

        # Classification loss
        self.objectness = tf.keras.losses.CategoricalCrossentropy()

        # Regression loss terms
        self.bbox_reg_l1 = tf.keras.losses.Huber()

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
        anchor_xx, anchor_yy = np.meshgrid(
            range(
                0,
                tf.cast(self.backbone.output_shape[1], dtype="int32"),
                self.anchor_stride,
            ),
            range(
                0,
                tf.cast(self.backbone.output_shape[0], dtype="int32"),
                self.anchor_stride,
            ),
        )
        self.anchor_xx = tf.constant(anchor_xx, dtype="float32")
        self.anchor_yy = tf.constant(anchor_yy, dtype="float32")

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
                    < self.backbone.output_shape[1],
                    self.anchor_yy[:, :, tf.newaxis]
                    + self.hh[tf.newaxis, tf.newaxis, :]
                    < self.backbone.output_shape[0],
                ),
            ),
            dtype="bool",
        )

    @tf.function
    def ground_truth_IoU(self, annotations, xx, yy, ww, hh):
        """
        Compute the ground truth IoU for a set of boxes defined in feature space.
        Note that xx,yy refers to the bottom left corner of the box, not the middle.

        Arguments:

        annotations : slice of tf.ragged.constant()
            Tensor of shape (None, 4,) encoded [xywh] in image coordinates.
        xx : tf.constant
            Pixel coordinates in the feature map.
        yy : tf.constant
            Pixel corrdinates in the feature map.
        ww : tf.constant
            Pixel corrdinates in the feature map.
        hh : tf.constant
            Pixel coordinates in the feature map.

        Returns:

        IoU: list of numpy array
            Ground truth IoU for each annotation.

        """

        print("Python interpreter in rpnwrapper.ground_truth_IoU")

        # Coordinates and area of the proposed region
        x, y = self.backbone.feature_coords_to_image_coords(xx, yy)
        w, h = self.backbone.feature_coords_to_image_coords(ww, hh)
        proposal_box = tf.stack([x, y, w, h])

        # Return stacked tensor
        return tf.stack(
            [
                geometry.calculate_IoU(proposal_box, annotations[i, :])
                for i in range(annotations.shape[0])
            ]
        )

    # Method that will eventually belong to the RPNWrapper class
    def accumulate_roi(self, label_decode, image_minibatch):
        """
        Make a list of RoIs of positive and negative examples
        and their corresponding ground truth annotations.

        Arguments:

        label_decode : list of dicts
            Decoded labels for this minibatch
        image_minibatch : int
            Number of images in a minibatch.

        Returns:

        list of [i, iyy, ixx, ik, {<label or empty>}] where variables
        denote indices into the RPN output

        """

        rois = []

        # Now iterate over images in the minibatch
        for i, this_label in enumerate(label_decode):
            """
            Fill the list of ROIs with both positive and negative examples

            Return (rpn_minibatch/images) samples (image number, xx, yy, hh, ww, {})
            corresponding to negative examples no matter what. This ensures that we
            have enough examples to fill out the RPN minibatch

            For each ground truth positive in label_decode append
            a) The best IoU as a positive
            b) Any region proposal with IoU > self.IoU_threshold

            Note that we are doing this with the box definitions without
            tuning to keep the RPN from running away from the original definitions.
            """

            # Fill in negative examples by random sampling
            count = 0
            for _ignore in range(CUTOFF):

                if count >= self.rpn_minibatch / image_minibatch:
                    break

                # Pick one at random
                ixx = np.random.randint(self.anchor_xx.shape[1])
                iyy = np.random.randint(self.anchor_xx.shape[0])
                ik = np.random.randint(self.k)

                # Check if this is a valid negative RoI
                if self.valid_mask[iyy, iyy, ik] and (
                    len(this_label) == 0
                    or all(
                        [
                            IoU < self.IoU_neg_threshold
                            for IoU in self.ground_truth_IoU(
                                this_label,
                                self.anchor_xx[iyy, ixx],
                                self.anchor_yy[iyy, ixx],
                                self.hh[ik],
                                self.ww[ik],
                            )
                        ]
                    )
                ):
                    rois.append([i, iyy, ixx, ik, {}])
                    count += 1

            # Short circuit if there are no starfish
            if len(this_label) == 0:
                continue

            # If there are positive examples return the example with the highest IoU per example
            # and any with IoU > threshold. First do the giant IoU calculation k times per annotation
            # Axis dimensions are labels, k, yy, xx
            ground_truth_IoU = np.stack(
                [
                    self.ground_truth_IoU(
                        this_label,
                        self.anchor_xx,
                        self.anchor_yy,
                        self.hh[ik],
                        self.ww[ik],
                    )
                    for ik in range(self.k)
                ]
            ).swapaxes(0, 1)

            # Acquire anything with IoU > self.IoU_pos_threshold
            for ilabel in range(ground_truth_IoU.shape[0]):
                pos_slice = np.argwhere(
                    np.logical_or(
                        ground_truth_IoU[ilabel, :, :, :] > self.IoU_pos_threshold,
                        ground_truth_IoU[ilabel, :, :, :]
                        == np.max(ground_truth_IoU[ilabel, :, :, :]),
                    )
                )
                for j in range(pos_slice.shape[0]):
                    ik, iyy, ixx = pos_slice[j, :]
                    xx = self.anchor_xx[iyy, ixx]
                    yy = self.anchor_yy[iyy, ixx]
                    hh = self.hh[ik]
                    ww = self.ww[ik]
                    rois.append([i, iyy, ixx, ik, this_label[ilabel]])

        # Something has gone horribly wrong with collecting RoIs, so skip this training step
        if len(rois) < self.rpn_minibatch:
            warnings.warn(
                "Something has gone wrong with collecting minibatch RoIs, skip training step."
            )
            return None

        # Finally, cut the list of RoIs down to something usable.
        # Do this by sorting on the existence of a ground truth
        # box + a small perturbation to randomly sample
        rois = sorted(
            rois,
            key=lambda roi: 1.0 * float("x" not in roi[-1].keys())
            + 0.001 * np.random.random(),
        )
        rois = (
            rois[: int(self.rpn_minibatch / 2)]
            + rois[-1 * int(self.rpn_minibatch / 2) :]
        )

        return rois

    # Required signature is this, so let's
    # work backwards. The rois needs to be a RaggedTensor
    # custom_loss(y_actual,y_pred)
    @tf.function
    def compute_loss(self, cls, bbox, rois):

        """
        Compute the loss function for a set of classification
        scores cls and bounding boxes bbox on the set of regions
        of interest RoIs.

        Arguments:

        cls : tensor, shape (i_image, iyy, ixx, ik)
            Slice of the output from the RPN corresponding to the
            classification (object-ness) field. From here on out
            the k ordering is [neg_k=0, neg_k=1, ... pos_k=0, pos_k=1...].
        bbox : tensor, shape (i_image, iyy, ixx, ik)
            Slice of the output from the RPN. The k dimension is ordered
            following a similar convention as cls:
            [t_x_k=0, t_x_k=1, ..., t_y_k=0, t_y_k=1,
            ..., t_w_k=0, t_w_k=1, ..., t_h_k=0, t_h_k=1, ...]
        rois : list or RoIs, [i, iyy, ixx, ik, {<label or empty>}]
            Output of accumulate_roi(), see training_step for use case.
        """

        # First, compute the categorical cross entropy objectness loss
        cls_select = tf.nn.softmax(
            [cls[roi[0], roi[1], roi[2], roi[3] :: self.k] for roi in rois]
        )
        ground_truth = [[1, 0] if "x" not in roi[4].keys() else [0, 1] for roi in rois]

        # Stop the training if we hit nan values
        if np.any(np.logical_not(np.isfinite(cls_select.numpy()))):
            raise ValueError("NaN detected in the RPN, aborting training.")

        # Start with an L2 regularization
        loss = tf.nn.l2_loss(cls) / (10.0 * tf.size(cls, out_type=tf.float32))
        loss += tf.nn.l2_loss(bbox) / tf.size(bbox, out_type=tf.float32)
        loss += self.objectness(ground_truth, cls_select)

        # Now add the bounding box term
        for roi in rois:

            # Compare anchor to ground truth
            if "x" in roi[4].keys():

                # Refer the corners of the bounding box back to image space
                # Note that this assumes said mapping is linear.
                x, y = self.backbone.feature_coords_to_image_coords(
                    self.anchor_xx[roi[1], roi[2]],
                    self.anchor_yy[roi[1], roi[2]],
                )
                w, h = self.backbone.feature_coords_to_image_coords(
                    self.ww[roi[3]],
                    self.hh[roi[3]],
                )

                t_x_star = (x - roi[4]["x"]) / (w)
                t_y_star = (y - roi[4]["y"]) / (h)
                t_w_star = geometry.safe_log(roi[4]["width"] / (w))
                t_h_star = geometry.safe_log(roi[4]["height"] / (h))

                # Huber loss, which AFAIK is the same as smooth L1
                loss += self.bbox_reg_l1(
                    [t_x_star, t_y_star, t_w_star, t_h_star],
                    bbox[roi[0], roi[1], roi[2], roi[3] :: self.k],
                )

        return loss

    def training_step(self, features, label_decode):

        """
        Take a convolved feature map, compute RoI, and
        update the RPN comparing to the ground truth.

        Arguments:

        features : tensor
            Minibatch of input images after running through the backbone
        label_decode : list of dicts
            Decoded labels for this minibatch

        """

        rois = self.accumulate_roi(label_decode, features.shape[0])

        # Something went wrong with making a list of RoIs, return
        if rois is None:
            return

        with tf.GradientTape() as tape:

            # Call the RPN
            cls, bbox = self.rpn(features, training=True)

            # Compute the loss using the classification scores and bounding boxes
            loss = self.compute_loss(cls, bbox, rois)

        # Apply gradients in the optimizer. Note that TF will throw spurious warnings if gradients
        # don't exist for the bbox regression if there were no + examples in the minibatch
        # This isn't actually an error, so list comprehension around it
        gradients = tape.gradient(loss, self.rpn.trainable_variables)
        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradients, self.rpn.trainable_variables)
            if grad is not None
        )

    def train_rpn(self, train_dataset, valid_dataset=None, epochs=5):
        """
        Main training loop iterating over a dataset.

        Arguments:

        train_dataset: tensorflow dataset
            Dataset of input images. Minibatch size and validation
            split are determined when this is created.
        epochs : int
            Number of epochs to run training for.

        """

        # TODO do we need or want a validation here? That might be slow-ish.
        valid_dataset
        self.rpn.fit(train_dataset, epochs=epochs, validation_data=valid_dataset)

        # Training loop, all complexity is in training_step()
        # for epoch in range(epochs):

        #   print("RPN training epoch %d" % epoch, end="")

        #   for i, (train_x, label_x) in enumerate(train_dataset):

        # The dots were getting out of hand
        #       if i % 100 == 0:
        #           print(".", end="")

        # Note that we could in theory fine-tune the method by moving the feature
        # convolution by the backbone into the self.training_step() tf.GradientTape()
        # block, but we are not doing that here.
        #        features = self.backbone.extractor(train_x)

        # Run gradient step with these features and the ground truth labels
        #        self.training_step(features, [labelfunc(label) for label in label_x])

        #    print("")

    @tf.function
    def propose_regions(
        self,
        input_image_or_features,
        image_coords=False,
        ignore_bbox=False,
        input_is_images=False,
    ):
        """
        Run the RPN in forward mode on a minibatch of images.
        This method is used to train the final classification network
        and to evaluate at test time.

        Arguments:

        input_image_or_features : dataset minibatch
            Set of image(s) to run through the network. Either features or images.
        top : int
            Return this number of region proposals with the highest classification
            scores. If <= 0 then return everything.
        image_coords : bool
            If True, returns objectness and coordinates in image space as numpy arrays.
            Otherwise, returns output as a tensor in feature space coordinates for
            feedforward to the rest of the network.
        ignore_bbox : bool
            If True, ignore the bounding box regression and return unmodified proposal
            regions based on the classification score. Useful for debugging.
        input_is_images : bool
            Set to true to run the input through the backbone, otherwise assumes this
            has already been done.

        Returns:
        [image_coords = False]
            Tensor of shape (batch_size, top, 4) with feature space
            coordinates (xx,yy,ww,hh)

        [image_coords = True]
            Tensor of shape (batch_size, top, 4) with image space
            coordinates (x,y,w,h)
        """

        # Run the feature extractor and the RPN in forward mode, adding an additional
        # image dimension if necessary

        if input_is_images:
            features = self.backbone.extractor(input_image_or_features)
        else:
            features = input_image_or_features

        # Run through the RPN
        cls, bbox = self.rpn(features)

        # Zero out the bounding box regression if requested
        if ignore_bbox:
            bbox *= 0.0

        # Dimension is image, iyy, ixx, ik were ik is
        # [neg_k=0, neg_k=1, ... pos_k=0, pos_k=1...]
        objectness_l0 = cls[:, :, :, : self.k]
        objectness_l1 = cls[:, :, :, self.k :]

        # Need to unpack a bit and hit with softmax
        objectness_l0 = tf.reshape(objectness_l0, (objectness_l0.shape[0], -1))
        objectness_l1 = tf.reshape(objectness_l1, (objectness_l1.shape[0], -1))
        objectness = tf.nn.softmax(tf.stack([objectness_l0, objectness_l1]), axis=0)

        # Cut to the one-hot bit
        objectness = objectness[1, :, :]

        # Remove the invalid bounding boxes
        objectness = tf.math.multiply(
            objectness, self.valid_mask.reshape(-1)[np.newaxis, :]
        )

        # Dimension for bbox is same as cls but ik follows
        # [t_x_k=0, t_x_k=1, ..., t_y_k=0, t_y_k=1,
        # ..., t_w_k=0, t_w_k=1, ..., t_h_k=0, t_h_k=1, ...]
        # Now cue the infinite magic numpy indexing
        xx = self.anchor_xx[np.newaxis, :, :, np.newaxis] - (
            bbox[:, :, :, : self.k] * self.ww[np.newaxis, np.newaxis, np.newaxis, :]
        )
        yy = self.anchor_yy[np.newaxis, :, :, np.newaxis] - (
            bbox[:, :, :, self.k : 2 * self.k]
            * self.hh[np.newaxis, np.newaxis, np.newaxis, :]
        )
        ww = self.ww[np.newaxis, np.newaxis, np.newaxis, :] * geometry.safe_exp(
            bbox[:, :, :, 2 * self.k : 3 * self.k]
        )
        hh = self.hh[np.newaxis, np.newaxis, np.newaxis, :] * geometry.safe_exp(
            bbox[:, :, :, 3 * self.k : 4 * self.k]
        )

        # Reshape to flatten along proposal dimension within an image
        argsort = tf.argsort(objectness, axis=-1, direction="DESCENDING")

        # If no limit requested, just return everything
        if self.n_roi < 1:
            self.n_roi = objectness.shape[1]

        def batch_sort(arr, inds, n):
            return tf.gather(tf.reshape(arr, (arr.shape[0], -1)), inds, batch_dims=1)[
                :, :n
            ]

        # Sort things by objectness
        xx = batch_sort(xx, argsort, self.n_roi)
        yy = batch_sort(yy, argsort, self.n_roi)
        ww = batch_sort(ww, argsort, self.n_roi)
        hh = batch_sort(hh, argsort, self.n_roi)

        if not image_coords:
            output = tf.stack([xx, yy, ww, hh], axis=-1)
            return output

        # Convert to image plane
        x, y = self.backbone.feature_coords_to_image_coords(xx, yy)
        w, h = self.backbone.feature_coords_to_image_coords(ww, hh)
        return tf.stack([x, y, w, h], axis=-1)

    def save_rpn_state(self, filename):
        """
        Save the trained RPN state.

        Arguments:

        path: str
            Save path for the RPN model.
        """

        tf.keras.models.save_model(self.rpn, filename)

    def load_rpn_state(self, filename):
        """
        Load the trained RPN state.

        path: str
            Load path for the RPN model.
        """

        self.rpn = tf.keras.models.load_model(filename)
