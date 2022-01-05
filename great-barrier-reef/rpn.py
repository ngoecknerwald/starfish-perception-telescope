# NOTE, from here on all variables denoted xx, yy, hh, ww refer to *feature space*
# and all variables denoted x, y, h, w denote *image space*

import tensorflow as tf
import numpy as np
import warnings

# Number of tries to find a four valid RoIs. Note that if this fails it is still
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
        self.cls = tf.keras.layers.Conv2D(
            filters=2 * self.k, kernel_size=1, strides=(1, 1), activation="softmax"
        )
        self.bbox = tf.keras.layers.Conv2D(
            filters=4 * self.k,
            kernel_size=1,
            strides=(1, 1),
        )
        if self.dropout is not None:
            self.dropout1 = tf.keras.layers.Dropout(self.dropout)

    # Also TODO figure out if we want use generalized IoU instead of the L1 loss
    def call(self, x, training=False):
        x = self.conv1(x)
        if hasattr(self, 'dropout1') and training:
            x = self.dropout1(x)
        cls = self.cls(x)
        bbox = self.bbox(x)
        return cls, bbox


class RPNWrapper:
    def __init__(
        self,
        backbone,
        kernel_size=3,
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2, decay_steps=1000, decay_rate=0.9
        ),
        anchor_stride=1,
        window_sizes=[2, 4, 6],  # TODO these must be divisible by 2
        filters=512,
        rpn_minibatch=256,
        IoU_neg_threshold=0.1,
        IoU_pos_threshold=0.7,
        rpn_dropout=0.2,
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

        # Anchor box sizes
        self.build_anchor_boxes()

        # Build the valid mask
        self.build_valid_mask()

        # Build the model
        self.rpn = RPN(
            self.k,
            self.kernel_size,
            self.anchor_stride,
            self.filters,
            dropout=self.rpn_dropout,
        )

        # Optimizer
        self.optimizer = tf.keras.optimizers.SGD(self.learning_rate, momentum=0.9)

        # Classification loss
        self.objectness = tf.keras.losses.CategoricalCrossentropy()

        # Regression loss
        self.bbox_reg = tf.keras.losses.Huber()

    def build_anchor_boxes(self):
        """
        Build the anchor box sizes in the feature space.

        """

        # Make the list of window sizes
        hh, ww = np.meshgrid(self.window_sizes, self.window_sizes)
        self.hh = hh.reshape(-1)
        self.ww = ww.reshape(-1)
        self.k = len(self.window_sizes) ** 2

        # Need to refer the anchor points back to the feature space
        self.anchor_xx, self.anchor_yy = np.meshgrid(
            range(0, self.backbone.output_shape[1], self.anchor_stride),
            range(0, self.backbone.output_shape[0], self.anchor_stride),
        )

    def validate_anchor_box(self, xx, yy, ww, hh):
        """
        Validate whether or not an anchor box in the extracted feature space
        defined by xx, yy, ww, hh is valid.

        Arguments:

        xx : int
            Pixel coordinate in the feature map.
        yy : int
            Pixel coordinate in the feature map.
        ww : int
            Pixel coordinate in the feature map.
        hh : int
            Pixel coordinate in the feature map.

        Returns:

        bool, whether or not this box is valid

        """

        xxmin = int(xx - ww / 2)
        xxmax = int(xx + ww / 2)
        yymin = int(yy - hh / 2)
        yymax = int(yy + hh / 2)

        return (xxmin >= 0 and yymin >= 0) and (
            xxmax < self.backbone.output_shape[1]
            and yymax < self.backbone.output_shape[0]
        )

    def build_valid_mask(self):
        """
        Build a mask the same shape as the output of the RPN
        indicating whether or not a proposal box crosses the image
        boundary.

        """

        self.valid_mask = np.zeros(
            (
                len(range(0, self.backbone.output_shape[0], self.anchor_stride)),
                len(range(0, self.backbone.output_shape[1], self.anchor_stride)),
                self.k,
            ),
            dtype=bool,
        )

        for ixx, xx in enumerate(
            range(0, self.backbone.output_shape[1], self.anchor_stride)
        ):
            for iyy, yy in enumerate(
                range(0, self.backbone.output_shape[0], self.anchor_stride)
            ):
                for ik, (hh, ww) in enumerate(zip(self.hh, self.ww)):
                    self.valid_mask[iyy, ixx, ik] = self.validate_anchor_box(
                        xx, yy, ww, hh
                    )

    def ground_truth_IoU(self, annotations, xx, yy, hh, ww):
        """
        Compute the ground truth IoU for a set of boxes defined in feature space.

        Arguments:

        annotations : list
            List of annotations in the input format, i.e.
            {'x': 406, 'y': 591, 'width': 57, 'height': 58}.
        xx : numpy array
            Pixel coordinate in the feature map.
        yy : numpy array
            Pixel corrdinate in the feature map.
        ww : numpy array
            Pixel corrdinate in the feature map.
        hh : numpy array
            Pixel coordinate in the feature map.

        Returns:

        IoU: list of numpy array
            Ground truth IoU for each annotation.

        """

        # Coordinates and area of the proposed region
        xmin, ymin = self.backbone.feature_coords_to_image_coords(
            xx - ww / 2, yy - hh / 2
        )
        xmax, ymax = self.backbone.feature_coords_to_image_coords(
            xx + ww / 2, yy + hh / 2
        )

        # Cast to numpy array to vectorize IoU over many proposal regions
        xmin = xmin * np.ones_like(xx)
        ymin = ymin * np.ones_like(yy)
        xmax = xmax * np.ones_like(xx)
        xmax = xmax * np.ones_like(yy)

        # Final output
        IoU = []

        # Cycle through the annotations and compute IoU
        for annotation in annotations:

            intersect = np.maximum(
                0,
                1
                + (
                    np.minimum(xmax, annotation["x"] + annotation["width"] / 2)
                    - np.maximum(xmin, annotation["x"] - annotation["width"] / 2)
                ),
            ) * np.maximum(
                0,
                1
                + (
                    np.minimum(ymax, annotation["y"] + annotation["height"] / 2)
                    - np.maximum(ymin, annotation["y"] - annotation["height"] / 2)
                ),
            )
            IoU.append(
                (intersect)
                / (
                    (xmax - xmin) * (ymax - ymin)
                    + annotation["width"] * annotation["height"]
                    - intersect
                )
            )

        return IoU

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

                # Get coords for the guess
                xx = self.anchor_xx[iyy, ixx]
                yy = self.anchor_yy[iyy, ixx]
                hh = self.hh[ik]
                ww = self.ww[ik]

                # Check if this is a valid negative RoI
                if self.valid_mask[iyy, iyy, ik] and (
                    len(this_label) == 0
                    or all(
                        [
                            IoU < self.IoU_neg_threshold
                            for IoU in self.ground_truth_IoU(this_label, xx, yy, hh, ww)
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

    def compute_loss(self, cls, bbox, rois, giou_frac=0.5):

        '''
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
        giou_frac : float
            Fraction of the cost function computed as generalized IoU
            using https://www.tensorflow.org/addons/api_docs/python/tfa/losses/GIoULoss
            Setting this to 0. uses only the smooth L1 loss on the bounding box regression.

        '''

        # First, compute the categorical cross entropy objectness loss
        cls_select = [cls[roi[0], roi[1], roi[2], roi[3] :: self.k] for roi in rois]
        ground_truth = [[1, 0] if 'x' not in roi[4].keys() else [0, 1] for roi in rois]
        loss = self.objectness(ground_truth, cls_select)
        # Now add the bounding box term
        for roi in rois:

            # Short circuit if there is no ground truth
            if 'x' not in roi[4].keys():
                continue

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

            # Compare anchor to ground truth
            t_x_star = (x - roi[4]['x']) / (w)
            t_y_star = (y - roi[4]['y']) / (h)
            t_w_star = np.log(roi[4]['width'] / (w))
            t_h_star = np.log(roi[4]['height'] / (h))

            # Huber loss, which AFAIK is the same as smooth L1
            loss += self.bbox_reg(
                [t_x_star, t_y_star, t_w_star, t_h_star],
                bbox[roi[0], roi[1], roi[2], roi[3] :: self.k],
            )
        return loss

    def training_step(self, train_x, label_decode, update_backbone=False):

        '''
        Take a convolved feature map, compute RoI, and
        update the RPN comparing to the ground truth.

        Arguments:

        features : tensor
            Minibatch of input images (before running through the backbone)
        label_decode : list of dicts
            Decoded labels for this minibatch

        '''

        # Note that we could in theory fine-tune the method by moving the feature
        # convolution by the backbone into the tf.GradientTape() block, but we are not doing that here.
        features = self.backbone.extractor(train_x)

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

    def train_rpn(self, train_dataset, labelfunc, epochs=10, update_backbone=False):
        '''
        Main training loop iterating over a dataset.

        Arguments:

        train_dataset: tensorflow dataset
            Dataset of input images. Minibatch size and validation
            split are determined when this is created.
        labelfunc : function
            Dictionary for translating the labels in to the dataset into
            ground truth annotations, usually DataLoaderFull.decode_label
        epochs : int
            Number of epochs to run training for.

        '''

        # Training loop, all complexity is in training_step()
        for epoch in range(epochs):

            print('RPN training epoch %d' % epoch, end='')

            for train_x, label_x in train_dataset:

                print('.', end='')

                # Run gradient step with these features and the ground truth labels
                self.training_step(train_x, [labelfunc(label) for label in label_x])

            print('')

    def propose_regions(self, minibatch, top=-1, image_coords=False):
        '''
        Run the RPN in forward mode on a minibatch of images.
        This method is used to train the final classification network
        and to evaluate at test time.

        Arguments:

        minibatch : dataset minibatch
            Set of image(s) to run through the network and extract features from.
        top : int
            Return this number of region proposals with the highest classification
            scores. If <= 0 then return everything.
        image_coords : bool
            If True, returns objectness and coordinates in image space as numpy arrays.
            Otherwise, returns output as a tensor in feature space coordinates for 
            feedforward to the rest of the network. 

        Returns:
        [image_coords = False]
            Tensor of shape (batch_size, top, 4) with feature space 
            coordinates (xx,yy,ww,hh)

        [image_coords = True]
        objectness, x, y, w, h
            Numpy arrays with dimension [image, region proposals]
            sorted by likelihood of being a starfish according to the RPN
        '''

        # Run the feature extractor and the RPN in forward mode, adding an additional
        # image dimension if necessary


        try:
            features = self.backbone.extractor(minibatch)
        except:
            features = self.backbone.extractor(minibatch[None, :, :, :])
        cls, bbox = self.rpn(features)

        # Dimension is image, iyy, ixx, ik were ik is
        # [neg_k=0, neg_k=1, ... pos_k=0, pos_k=1...]
        objectness = cls[:, :, :, self.k :].numpy()

        # Dimension for bbox is same as cls but ik follows
        # [t_x_k=0, t_x_k=1, ..., t_y_k=0, t_y_k=1,
        # ..., t_w_k=0, t_w_k=1, ..., t_h_k=0, t_h_k=1, ...]
        # Now cue the infinite magic numpy indexing
        output = {}
        xx = self.anchor_xx[np.newaxis, :, :, np.newaxis] + (
            bbox[:, :, :, : self.k].numpy()
            * self.ww[np.newaxis, np.newaxis, np.newaxis, :]
        )
        yy = self.anchor_yy[np.newaxis, :, :, np.newaxis] + (
            bbox[:, :, :, self.k : 2 * self.k].numpy()
            * self.hh[np.newaxis, np.newaxis, np.newaxis, :]
        )
        ww = self.ww[np.newaxis, np.newaxis, np.newaxis, :] * np.exp(
            bbox[:, :, :, 2 * self.k : 3 * self.k].numpy()
        )
        hh = self.hh[np.newaxis, np.newaxis, np.newaxis, :] * np.exp(
            bbox[:, :, :, 3 * self.k : 4 * self.k].numpy()
        )

        # Reshape to flatten along proposal dimension within an image
        objectness = objectness.reshape((objectness.shape[0], -1))
        argsort = np.argsort(objectness, axis=-1)
        argsort = argsort[:, ::-1]

        # If no limit requested, just return everything
        if top < 1:
            top = objectness.shape[1]

        def batch_sort(arr, inds, top):
            return np.take_along_axis(arr.reshape(arr.shape[0], -1), inds, axis=-1)[:, :top]

        # Sort things by objectness
        objectness = batch_sort(objectness, argsort, top)
        xx = batch_sort(xx, argsort, top)
        yy = batch_sort(yy, argsort, top)
        ww = batch_sort(ww, argsort, top)
        hh = batch_sort(hh, argsort, top)

        if not image_coords: 
            output = tf.stack([xx,yy,ww,hh], axis=-1)
            return output

        output = {}

        # Convert to image plane
        (
            output['x'],
            output['y'],
        ) = self.backbone.feature_coords_to_image_coords(xx, yy)
        (
            output['w'],
            output['h'],
        ) = self.backbone.feature_coords_to_image_coords(ww, hh)

        return (
            objectness,
            output['x'],
            output['y'],
            output['w'],
            output['h'],
        )
