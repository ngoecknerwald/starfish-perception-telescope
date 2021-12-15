# This will contain the RPN class, here a totally empty stub

# NOTE, from here on all variables denoted xx, yy, hh, ww refer to *feature space*
# and all variables denoted x, y, h, w denote *image space*

import tensorflow as tf


class RPN(tf.keras.Model):
    def __init__(self, k, kernel_size=3):
        super(MyModel, self).__init__()
        self.conv1 = (
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=kernel_size, strides=(1, 1), activation='relu'
            ),
        )
        self.cls = (
            tf.keras.layers.Conv2D(
                filters=2 * self.k, kernel_size=1, strides=(1, 1), activation='softmax'
            ),
        )
        self.bbox = (
            tf.keras.layers.Conv2D(
                filters=4 * self.k,
                kernel_size=1,
                strides=(1, 1),
            ),
        )

    def call(self, x, training=None, **kwargs):
        x = self.conv1(x)
        cls = self.cls(x)
        bbox = self.bbox(x)
        return cls, bbox


class RPNWrapper:
    def __init__(self, backbone, **kwargs):
        '''
        Initialize the RPN model for pretraining.

        '''

        # Store the image size
        self.backbone = backbone

        # Now initialize networks
        self.build_anchor_boxes(**kwargs)
        self.build_anchor_points(**kwargs)

        # Build the model
        self.rpn = RPN(self.k)

    def feature_coords_to_image_coords(self, xx, yy):
        '''
        Naively maps coordinates x,y in extracted feature space to
        coordinates in map space.

        Arguments:

        xx : int
            Pixel coordinate in the feature map.
        yy : int
            Pixel corrdinate in the feature map.

        TODO this probably isn't actually right, because of edge effect.
        Come back to this if the boxes are all systematically offset.
        '''

        return (
            int(
                xx * float(self.backbone.input_shape[1] / self.backbone.output_shape[1])
            ),
            int(
                yy * float(self.backbone.input_shape[0] / self.backbone.output_shape[0])
            ),
        )

    def image_coords_to_feature_coords(self, x, y):
        '''
        Naively map coordinates in image space to feature space.

        Arguments:

        x : int
            Pixel coordinate in the image map.
        y : int
            Pixel corrdinate in the image map.

        TODO this probably isn't actually right, because of edge effect.
        Come back to this if the boxes are all systematically offset.
        '''

        return (
            int(
                x * float(self.backbone.output_shape[1] / self.backbone.input_shape[1])
            ),
            int(
                y * float(self.backbone.output_shape[0] / self.backbone.input_shape[0])
            ),
        )

    def build_anchor_boxes(
        self,
        window_sizes=[2, 4, 8],
        aspect_ratios=[0.5, 1, 2.0],
    ):
        '''
        Build the anchor box sizes in the image space.

        window_sizes : list of ints
            Width of the proposals to select, *in extracted feature space*.
        aspect_ratios : list of floats
            Set of aspect ratios (height / width) to select.

        '''

        # Make the list of window sizes
        hh, ww = np.meshgrid(window_sizes, window_sizes)
        hh *= aspect_ratios[:, np.newaxis]

        # Flatten for simplicity later
        self.hh = hh.reshape[-1]
        self.ww = ww.reshape[-1]
        self.k = len(window_sizes) * len(aspect_ratios)

    def build_anchor_points(
        self,
        stride=1,
        boundary='clip',
    ):

        '''
        Build the anchor points.

        Arguments:

        stride : int
            Place an anchor every <stride> pixels in the *extracted feature space*.
        '''
        # For each point, define an anchor box
        xx, yy = np.meshgrid(
            range(0, self.backbone.output_shape[1], stride),
            range(0, self.backbone.output_shape[0], stride),
        )
        self.xx = xx.reshape(-1)
        self.yy = yy.reshape(-1)

    def validate_anchor_box(xx, yy, ww, hh):
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

        return (lxmin > 0 and lymin > 0) and (
            lxmax < self.image_shape[1] and lymax < self.image_shape[0]
        )

    def train_step(self):

        pass
