# This will contain the RPN class, here a totally empty stub

# NOTE, from here on all variables denoted xx, yy, hh, ww refer to *feature space*
# and all variables denoted x, y, h, w denote *image space*

import tensorflow as tf
import numpy as np


class RPN(tf.keras.Model):
    def __init__(self, k, kernel_size, anchor_stride, filters):
        '''
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

        '''

        super().__init__()
        self.k = k
        self.kernel_size = kernel_size
        self.anchor_stride = anchor_stride
        self.filters = filters
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=kernel_size,
            strides=(anchor_stride, anchor_stride),
            activation='relu',
            padding='same',
        )
        self.cls = tf.keras.layers.Conv2D(
            filters=2 * self.k, kernel_size=1, strides=(1, 1), activation='softmax'
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
    ):
        '''
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

        '''

        # Store the image size
        self.backbone = backbone
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.anchor_stride = anchor_stride
        self.window_sizes = window_sizes
        self.filters = filters

        # Anchor box sizes
        self.build_anchor_boxes()

        # Build the valid mask
        self.build_valid_mask()

        # Build the model
        self.rpn = RPN(self.k, self.kernel_size, self.anchor_stride, self.filters)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def feature_coords_to_image_coords(self, xx, yy):
        '''
        Naively maps coordinates x,y in extracted feature space to
        coordinates in map space.

        Arguments:

        xx : int or numpy array
            Pixel coordinate in the feature map.
        yy : int or numpy array
            Pixel corrdinate in the feature map.

        TODO this probably isn't actually right, because of edge effect.
        Come back to this if the boxes are all systematically offset.
        '''

        return (
            xx * float(self.backbone.input_shape[1] / self.backbone.output_shape[1]),
            yy * float(self.backbone.input_shape[0] / self.backbone.output_shape[0]),
        )

    def image_coords_to_feature_coords(self, x, y):
        '''
        Naively map coordinates in image space to feature space.

        Arguments:

        x : int or numpy array
            Pixel coordinate in the image map.
        y : int or numpy array
            Pixel corrdinate in the image map.

        TODO this probably isn't actually right, because of edge effect.
        Come back to this if the boxes are all systematically offset.
        '''

        return (
            x * float(self.backbone.output_shape[1] / self.backbone.input_shape[1]),
            y * float(self.backbone.output_shape[0] / self.backbone.input_shape[0]),
        )

    def build_anchor_boxes(self):
        '''
        Build the anchor box sizes in the feature space.

        '''

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
        '''
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

        '''

        xxmin = int(xx - ww / 2)
        xxmax = int(xx + ww / 2)
        yymin = int(yy - hh / 2)
        yymax = int(yy + hh / 2)

        return (xxmin >= 0 and yymin >= 0) and (
            xxmax < self.backbone.output_shape[1]
            and yymax < self.backbone.output_shape[0]
        )

    def build_valid_mask(self):
        '''
        Build a mask the same shape as the output of the RPN
        indicating whether or not a proposal box crosses the image
        boundary.

        '''

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
        '''
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

        '''

        # Coordinates and area of the proposed region
        xmin, ymin = self.feature_coords_to_image_coords(xx - ww / 2, yy - hh / 2)
        xmax, ymax = self.feature_coords_to_image_coords(xx + ww / 2, yy + hh / 2)

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
                    np.minimum(xmax, annotation['x'] + annotation['width'] / 2)
                    - np.maximum(xmin, annotation['x'] - annotation['width'] / 2)
                ),
            ) * np.maximum(
                0,
                1
                + (
                    np.minimum(ymax, annotation['y'] + annotation['height'] / 2)
                    - np.maximum(ymin, annotation['y'] - annotation['height'] / 2)
                ),
            )
            IoU.append(
                (intersect)
                / (
                    (xmax - xmin) * (ymax - ymin)
                    + annotation['width'] * annotation['height']
                    - intersect
                )
            )

        return IoU

    def train_step(self):

        pass
