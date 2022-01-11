# Class for converting from the RPN outputs to the tail network inputs
import tensorflow as tf
import geometry
import numpy as np


def clip_RoI(roi, feature_size, pool_size):

    """
    Take the IoU before or after IoU supression and clip to the image boundaries.
    Agnostic to feature dimensions or image dimensions.

    Arguments:

    roi : np.ndarray or tf.Tensor
        Float tensor of the same size as returned by
        IoU_supression. Shape is [image number, RoI number, (x,y,w,h).
    feature_size : tuple of int
        Shape of the feature map or the input image.
        Assumed to be the same convention (image vs feature dim) as roi_pruned.
    pool_size : size of the pooled images. Imposes a floor on w and h.
        Assumed to be the same convention (image vs feature dim) as roi_pruned.

    Returns:

    roi_clipped : np.ndarray
        RoIs clipped to the image bondaries and the minimum sizes.
    """

    if isinstance(roi, tf.Tensor):
        roi = roi.numpy()

    # sanity checking that the pool size isn't > the feature size
    assert feature_size[0] > pool_size[0] and feature_size[1] > pool_size[1]

    roi_clipped = np.zeros(roi.shape, dtype=int)
    #  temporarily convert from x,y,w,h to x,y,x+w,y+h
    roi_clipped[:, :, 0] = np.maximum(0, roi[:, :, 0].astype(int))
    roi_clipped[:, :, 1] = np.maximum(0, roi[:, :, 1].astype(int))
    roi_clipped[:, :, 2] = np.minimum(
        feature_size[1], (roi[:, :, 0] + roi[:, :, 2]).astype(int)
    )
    roi_clipped[:, :, 3] = np.minimum(
        feature_size[0], (roi[:, :, 1] + roi[:, :, 3]).astype(int)
    )

    # Padding:
    # 0. Leave box alone if big enough.
    # 1. If too small, attempt to pad symetrically left and right
    #    If pool_size is odd, always add the extra pixel to the right (top)
    # 2. Elif not enough space on left (bottom), fix box to left (bottom) boundary
    # 3. Elif not enough space on right (top), fix box to right (top) boundary

    for mini, maxi, si in zip([0, 1], [2, 3], [1, 0]):

        pad = pool_size[si] - (roi_clipped[:, :, maxi] - roi_clipped[:, :, mini])
        fix_min = roi_clipped[:, :, mini] < pad // 2
        fix_max = (feature_size[si] - roi_clipped[:, :, maxi]) < (1 + pad) // 2

        symmetric = np.logical_and(pad > 0, ~np.logical_or(fix_min, fix_max))
        roi_clipped[:, :, mini][symmetric] -= pad[symmetric] // 2
        roi_clipped[:, :, maxi][symmetric] += (1 + pad[symmetric]) // 2

        roi_clipped[:, :, mini][np.logical_and(pad > 0, fix_min)] = 0
        roi_clipped[:, :, maxi][np.logical_and(pad > 0, fix_min)] = pool_size[si]

        roi_clipped[:, :, mini][np.logical_and(pad > 0, fix_max)] = (
            feature_size[si] - pool_size[si]
        )
        roi_clipped[:, :, maxi][np.logical_and(pad > 0, fix_max)] = feature_size[si]

    #  convert back from x,y,x+w,y+h to x,y,w,h
    roi_clipped[:, :, 2] -= roi_clipped[:, :, 0]
    roi_clipped[:, :, 3] -= roi_clipped[:, :, 1]

    return roi_clipped


def IoU_supression(roi, IoU_threshold=0.4, n_regions=10):

    """
    Remove duplicate RoIs from the stack produced
    by the RPN or after clipping. Note this is agnostic to
    feature versus image coordinates.

    Arguments:

    roi : tf.Tensor or np.ndarray
        Regions of interest tensor as output by the RPN.
        This tensor must be sorted by descending objectness.
        Shape is [image number, RoI number, (x,y,w,h)].

    IoU_threshold : float
        Threshold above which two returned regions of interest
        are deemed the same underlying object.

    n_regions : int
        Number of regions to return. Note that roi.shape[1] should
        be >> n_regions to avoid an underflow once regions are pruned.

    Outputs :

    roi_pruned : tensor
        Pruned RoI tensor. Shape is the same as the input roi
        tensor, [image number, RoI number, (x,y,w,h)].

    """

    if isinstance(roi, tf.Tensor):
        roi = roi.numpy()

    # Sanity checking
    assert roi.shape[2] == 4  # x,y,w,h

    # Indices to output
    index_tensor = np.empty((roi.shape[0], n_regions, 4), int)

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
                    > IoU_threshold
                ):
                    discard.append(j)

        # Pick out what's remaining
        keep = np.setdiff1d(np.arange(roi.shape[1]), discard, assume_unique=True)

        # Insufficient regions
        if keep.shape[0] < n_regions:
            arr = np.array(
                [
                    roi.shape[1] - 1,
                ]
                * (n_regions - keep.shape[0])
            )
            keep = np.concatenate((keep, arr))

        # Fill out the index tensor
        index_tensor[i, :, :] = keep[:n_regions, np.newaxis]

    return np.take_along_axis(roi, index_tensor, axis=1)


class ROIPooling(tf.keras.layers.Layer):

    # based on https://medium.com/xplore-ai/implementing-attention-in-tensorflow-keras-using-roi-pooling-992508b6592b
    # added cropping and minimum size padding

    def __init__(self, pool_size, n_regions, IoU_threshold=0.4, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.n_regions = n_regions
        self.IoU_threshold = IoU_threshold

    def call(self, x):
        # x[0] = feature tensor
        # x[1] = output from rpn.propose_regions

        # TODO: rewrite this class to use vectorization instead of tf.map_fn
        # though map_fn works nicely here because it broadcasts over the first dimension
        # of both features and roi

        features, roi = x
        roi_pruned = IoU_supression(roi, self.IoU_threshold, self.n_regions)
        roi_clipped = clip_RoI(roi_pruned, features.shape[1:3], self.pool_size)
        x = (features, roi_clipped)

        def curried_pool_rois(x):
            return ROIPooling._pool_rois(x[0], x[1], self.pool_size)

        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_rois(feature_map, rois, pool_size):
        """Applies ROI pooling for a single image and various ROIs"""

        def curried_pool_roi(roi):
            return ROIPooling._pool_roi(feature_map, roi, pool_size)

        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_roi(feature_map, roi, pool_size):

        region = feature_map[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[1], :]
        # Divide the region into non overlapping areas
        h_step = tf.cast(region.shape[0] / pool_size[0], "int32")
        w_step = tf.cast(region.shape[1] / pool_size[1], "int32")

        areas = [
            [
                (
                    i * h_step,
                    j * w_step,
                    (i + 1) * h_step if i + 1 < pool_size[0] else region.shape[0],
                    (j + 1) * w_step if j + 1 < pool_size[1] else region.shape[1],
                )
                for j in range(pool_size[1])
            ]
            for i in range(pool_size[0])
        ]

        def pool_area(x):
            return tf.math.reduce_max(region[x[0] : x[2], x[1] : x[3], :], axis=[0, 1])

        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
        return pooled_features
