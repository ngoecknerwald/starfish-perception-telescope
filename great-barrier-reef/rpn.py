# This will contain the RPN class, here a totally empty stub

# NOTE, from here on all variables denoted xx, yy, hh, ww refer to *feature space*
# and all variables denoted x, y, h, w denote *image space*

import tensorflow as tf
import numpy as np
import warnings

# Number of tries to find a four valid RoIs. Note that if this fails it is still
# possible to successfully build a minibatch
CUTOFF = 100


class RPN(tf.keras.Model):
    def __init__(self, k, kernel_size, anchor_stride, filters):
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

        """

        super().__init__()
        self.k = k
        self.kernel_size = kernel_size
        self.anchor_stride = anchor_stride
        self.filters = filters
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

    def call(self, x):
        x = self.conv1(x)
        cls = self.cls(x)
        bbox = self.bbox(x)
        return cls, bbox


class RPNWrapper:
    def __init__(
        self,
        backbone,
        kernel_size=3,
        learning_rate=1e-4,
        anchor_stride=1,
        window_sizes=[2, 4, 6],  # TODO these must be divisible by 2
        filters=256,
        rpn_minibatch=256,
        IoU_neg_threshold=0.1,
        IoU_pos_threshold=0.7,
    ):
        """
        Initialize the RPN model for pretraining.

        backbone : Backbone
            Knows input image and feature map sizes but is not directly trained here.
        kernel_size : int
            Kernel size for the first convolutional layer in the RPN.
        learning_rate : float
            Learning rate for the ADAM optimizer.
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

        # Anchor box sizes
        self.build_anchor_boxes()

        # Build the valid mask
        self.build_valid_mask()

        # Build the model
        self.rpn = RPN(self.k, self.kernel_size, self.anchor_stride, self.filters)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

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
            Pixel corrdinate in the feature map.
        ww : int
            Pixel corrdinate in the feature map.
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
    def accumulate_roi(self, features, label_decode):
        """
        Make a list of RoIs of positive and negative examples
        and their corresponding ground truth annotations.

        Arguments:

        features : tensor
            Minibatch of input images after running through the backbone.
        label_decode : list of dicts
            Decoded labels for this minibatch

        Returns:

        list of [xx, yy, hh, ww, {<label or empty>}]

        """

        # Sanity checking
        assert features.shape[0] == len(label_decode)
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
            for i in range(CUTOFF):

                if count >= self.rpn_minibatch / features.shape[0]:
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
                    rois.append([xx, yy, hh, ww, {}])
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
                for i in range(pos_slice.shape[0]):
                    ik, iyy, ixx = pos_slice[i, :]
                    xx = self.anchor_xx[iyy, ixx]
                    yy = self.anchor_yy[iyy, ixx]
                    hh = self.hh[ik]
                    ww = self.ww[ik]
                    rois.append([xx, yy, hh, ww, this_label[ilabel]])

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

    def train_step(self):

        pass
