# Class for converting from the RPN outputs to the tail network inputs
# Adapted from https://medium.com/xplore-ai/implementing-attention-in-tensorflow-keras-using-roi-pooling-992508b6592b

import tensorflow as tf
import geometry

# Extend the Keras layer class
class RoIPooling(tf.keras.layers.Layer):
    def __init__(self, feature_size, n_regions, pool_size=(3, 3), IoU_threshold=0.4):
        """
        Instantiate the RoI pooling layer.

        Arguments :

        n_regions : int
            Number of region proposals to return after clipping and pooling.
        pool_size : tuple of int
            Size of the pooled images in feature space. Equivalent to the
            input spatial dimension of the final classification layer.
        IoU_threshold : float > 0 and < 1
            Threshold above which two returned regions of interest
            are deemed the same underlying object.
        feature_size : tuple of int
            Shape of the feature map or the input image.
            Assumed to be the same convention (image vs feature dim) as roi_pruned.

        """

        super().__init__()
        self.n_regions = n_regions
        self.pool_size = pool_size
        self.IoU_threshold = IoU_threshold
        self.feature_size = feature_size

        # sanity checking that the pool size isn't > the feature size
        assert (
            self.feature_size[0] > self.pool_size[0]
            and self.feature_size[1] > self.pool_size[1]
        )

    # The proper thing for model serialization is to
    # extend call() and not __call__(), apparently
    @tf.function
    def call(self, data):
        """
        Perform IoU suppression and clipping on the RoI and then pool
        the features map output by the backbone.

        Arguments:

        data: tuple containing (features, roi)

        features : tf.Tensor
            Feature map returned by the backbone network.
            Shape [image, xx, yy, channels.]
        roi : tf.Tensor
            RoI bounds returned by the RPN before any
            postprocessing to clip boundaries and remove duplicate proposals.
            Shape [image, regions, xywh]

        Returns:

        features_pool : tf.tensor
            Feature tensor after RoI pooling.
        roi_clipped : np.ndarray
            Processed RoI to remove duplicates, impose minimum feature
            dimensions, and trim to image boundaries.

        """

        features, roi = data

        # Deduplicate and clip the input RoI
        roi_clipped = self._clip_RoI(self._IoU_supression(roi))

        # Use map_fn to iterate over images
        pooled_areas = tf.map_fn(
            self._pool_rois, (features, roi_clipped), dtype=tf.float32
        )

        return pooled_areas, roi_clipped

    # TODO need to excise the numpy from this function
    @tf.function
    def _IoU_supression(self, roi):

        """
        Remove duplicate RoIs from the stack produced
        by the RPN or after clipping.

        Arguments:

        roi : tf.Tensor or np.ndarray
            Regions of interest tensor as output by the RPN.
            This tensor must be sorted by descending objectness.
            Shape is [image number, RoI number, (x,y,w,h)].

        Outputs :

        roi_pruned : tensor
            Pruned RoI tensor. Shape is the same as the input roi
            tensor, [image number, RoI number, (x,y,w,h)].

        """

        # Sanity checking
        assert roi.shape[2] == 4  # x,y,w,h

        # Indices to output
        index_tensor = np.empty((roi.shape[0], self.n_regions, 4), int)

        # loop over batch dim
        for i in range(roi.shape[0]):

            # double loop over roi in batch
            # fix a pivot starting at 0
            # for each pivot, discard all elements with IoU > threshold
            # next pivot is the next element that has been kept

            discard = []

            for pivot in range(roi.shape[1] - 1):

                if pivot in discard:
                    continue

                for j in range(pivot + 1, roi.shape[1]):

                    # Logic should short circuit here and not evaluate IoU
                    # if it's already in discard
                    if (
                        j not in discard
                        and geometry.calculate_IoU(
                            (roi[i, pivot, :]),
                            (roi[i, j, :]),
                        )
                        > self.IoU_threshold
                    ):
                        discard.append(j)

            # Pick out what's remaining
            keep = np.setdiff1d(np.arange(roi.shape[1]), discard, assume_unique=True)

            # Insufficient regions
            if keep.shape[0] < self.n_regions:
                arr = np.array(
                    [
                        roi.shape[1] - 1,
                    ]
                    * (self.n_regions - keep.shape[0])
                )
                keep = np.concatenate((keep, arr))

            # Fill out the index tensor
            index_tensor[i, :, :] = keep[: self.n_regions, np.newaxis]

        return tf.experimental.numpy.take_along_axis(roi, index_tensor.astype("int32"), axis=1)

    # TODO need to excise the numpy from this function
    @tf.function
    def _clip_RoI(self, roi):

        """
        Take the IoU before or after IoU supression and clip to the image boundaries.
        Agnostic to feature dimensions or image dimensions.

        Arguments:

        roi : np.ndarray or tf.Tensor
            Float tensor of the same size as returned by
            IoU_supression. Shape is [image number, RoI number, (x,y,w,h).

        Returns:

        roi_clipped : np.ndarray
            RoIs clipped to the image bondaries and the minimum sizes.
        """

        #  temporarily convert from x,y,w,h to x,x+w,y,y+h
        x_min = tf.cast(tf.math.maximum(0., roi[:, :, 0]), "int32")
        y_min = tf.cast(tf.math.maximum(0., roi[:, :, 1]), "int32")
        x_max = tf.cast(tf.math.minimum(
            tf.cast(self.feature_size[1], "float32"), (roi[:, :, 0] + roi[:, :, 2])
        ), "int32")
        y_max = tf.cast(tf.math.minimum(
            tf.cast(self.feature_size[0], "float32"), (roi[:, :, 1] + roi[:, :, 3])
        ), "int32")

        roi_clipped = tf.stack([x_min, x_max, y_min, y_max], axis=-1)

        # Padding:
        # 0. Leave box alone if big enough.
        # 1. If too small, attempt to pad symetrically left and right
        #    If pool_size is odd, always add the extra pixel to the right (top)
        # 2. Elif not enough space on left (bottom), fix box to left (bottom) boundary
        # 3. Elif not enough space on right (top), fix box to right (top) boundary

        clipped = []  # same order as roi_clipped
        for mini, maxi, si in zip([0, 2], [1, 3], [1, 0]):

            pad = self.pool_size[si] - (
                roi_clipped[:, :, maxi] - roi_clipped[:, :, mini]
            )
            fix_min = roi_clipped[:, :, mini] < pad // 2
            fix_max = (self.feature_size[si] - roi_clipped[:, :, maxi]) < (1 + pad) // 2

            symmetric = tf.math.logical_and(pad > 0, ~tf.math.logical_or(fix_min, fix_max))

            omin = tf.where(symmetric, roi_clipped[:,:,mini] - pad//2, roi_clipped[:,:,mini])
            omax = tf.where(symmetric, roi_clipped[:,:,maxi] + (1 + pad)//2, roi_clipped[:,:,maxi])

            omin = tf.where(tf.math.logical_and(pad > 0, fix_min), 0, omin)
            omax = tf.where(tf.math.logical_and(pad > 0, fix_min), self.pool_size[si], omax)

            omin = tf.where(tf.math.logical_and(pad > 0, fix_max), self.feature_size[si] - self.pool_size[si], omin)
            omax = tf.where(tf.math.logical_and(pad > 0, fix_max), self.feature_size[si], omax)

            clipped.extend([omin, omax])
        
        roi_clipped = tf.stack([clipped[0], clipped[2], clipped[1] - clipped[0], clipped[3] - clipped[2]], axis=-1)

        return roi_clipped

    @tf.function
    def _pool_rois(self, x):
        """
        Internal helper method.

        Apply ROI pooling for a single image and a list of RoI.
        """

        feature_map, rois = x

        # Use map_fn to iterate over RoI
        return tf.map_fn(
            partial(self._pool_roi, feature_map=feature_map), rois, dtype=tf.float32
        )

    @tf.function
    def _pool_roi(self, roi, feature_map=None):
        """
        Internal helper method.

        Apply RoI pooling for a single image and a single RoI.
        """
        print(roi)
        region = feature_map[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2], :]
        print(region.shape)
        # Divide the region into non overlapping areas
        h_step = tf.cast(region.shape[0] / self.pool_size[0], "int32")
        w_step = tf.cast(region.shape[1] / self.pool_size[1], "int32")

        areas = [
            [
                (
                    i * h_step,
                    j * w_step,
                    (i + 1) * h_step if i + 1 < self.pool_size[0] else region.shape[0],
                    (j + 1) * w_step if j + 1 < self.pool_size[1] else region.shape[1],
                )
                for j in range(self.pool_size[1])
            ]
            for i in range(self.pool_size[0])
        ]

        return tf.stack(
            [
                [
                    tf.math.reduce_max(region[x[0] : x[2], x[1] : x[3], :], axis=[0, 1])
                    for x in row
                ]
                for row in areas
            ]
        )
