# Class for converting from the RPN outputs to the tail network inputs
import tensorflow as tf
import geometry
import numpy as np


def clip_RoI(roi, feature_size, pool_size):

    '''
    Take the IoU before or after IoU supression and clip to the image boundaries.
    Agnostic to feature dimensions or image dimensions.

    Arguments:

    roi : np.ndarray or tf.Tensor
        Float tensor of the same size as returned by
        IoU_supression. Shape is [image number, RoI number, (x,y,w,h)].
    feature_size : tuple of int
        Shape of the feature map or the input image.
        Assumed to be the same convention (image vs feature dim) as roi_pruned.
    pool_size : size of the pooled images. Imposes a floor on w and h.
        Assumed to be the same convention (image vs feature dim) as roi_pruned.

    Returns:

    roi_clipped : np.ndarray
        RoIs clipped to the image bondaries and the minimum sizes.
    '''

    # Note that the roi is ordered x,y where as numpy indexing is y,x
    # meaning it looks like the dimensions are jumbled. I'm fairly certain it's right.

    if isinstance(roi, tf.Tensor):
        roi = roi.numpy()

    # sanity checking that the pool size isn't > the feature size
    assert feature_size[0] > pool_size[0] and feature_size[1] > pool_size[1]

    roi_clipped = np.zeros(roi.shape, dtype=int)

    # Now impose the size minima
    roi_clipped[:, :, 2] = np.maximum(roi[:, :, 2].astype(int), pool_size[1])
    roi_clipped[:, :, 3] = np.maximum(roi[:, :, 3].astype(int), pool_size[0])

    # Now put the boxes where they should go with np.rint()
    roi_clipped[:, :, :1] = np.rint(roi[:, :, :1])

    # Finally shift boxes toward the middle if they are clipped at the edge
    roi_clipped[:, :, 0] = np.maximum(roi[:, :, 0], int(pool_size[1] / 2))
    roi_clipped[:, :, 1] = np.maximum(roi[:, :, 1], int(pool_size[0] / 2))
    roi_clipped[:, :, 0] = np.minimum(
        roi[:, :, 0], feature_size[1] - int(pool_size[1] / 2)
    )
    roi_clipped[:, :, 1] = np.minimum(
        roi[:, :, 1], feature_size[0] - int(pool_size[0] / 2)
    )

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

    def __init__(self, pooled_height, pooled_width, **kwargs):
        super().__init__(**kwargs)
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

    def call(self, x):
        # x[0] = feature tensor
        # x[1] = output from rpn.propose_regions

        def curried_pool_rois(x):
            return ROIPooling._pool_rois(
                x[0], x[1], self.pooled_height, self.pooled_width
            )

        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """Applies ROI pooling for a single image and various ROIs"""

        def curried_pool_roi(roi):
            return ROIPooling._pool_roi(feature_map, roi, pooled_height, pooled_width)

        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):

        xx = roi[0]
        yy = roi[1]
        ww = roi[2]
        hh = roi[3]

        # Crop RoI to image boundaries
        h_start = tf.math.maximum(tf.cast(yy - hh / 2, "int32"), 0)
        h_end = tf.math.minimum(tf.cast(yy + hh / 2, "int32"), feature_map.shape[0])
        w_start = tf.math.maximum(tf.cast(xx - ww / 2, "int32"), 0)
        w_end = tf.math.minimum(tf.cast(xx + ww / 2, "int32"), feature_map.shape[1])

        # Enlarge RoI as needed:
        # Assume the image can always be extended in some direction.
        # Find the necessary padding size, and extend the image either symetrically
        # in both directions; or, go to the edge on one side and make up
        # the difference on the other side.
        hpad = pooled_height - (h_end - h_start)
        if hpad > 0:
            hpad = tf.cast(tf.math.ceil(hpad / 2), "int32")
            top = feature_map.shape[0] - h_end
            bottom = h_start
            if top < hpad:
                h_end += top
                h_start -= 2 * hpad - top
            elif bottom < hpad:
                h_end += 2 * hpad - bottom
                h_start -= bottom
            else:
                h_start -= hpad
                h_end += hpad

        wpad = pooled_width - (w_end - w_start)
        if wpad > 0:
            wpad = tf.cast(tf.math.ceil(wpad / 2), "int32")
            right = feature_map.shape[1] - w_end
            left = w_start
            if right < wpad:
                w_end += right
                w_start -= 2 * wpad - right
            elif left < wpad:
                w_end += 2 * wpad - left
                w_start -= left
            else:
                w_start -= wpad
                w_end += wpad

        region = feature_map[h_start:h_end, w_start:w_end, :]

        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width = w_end - w_start
        h_step = tf.cast(region_height / pooled_height, "int32")
        w_step = tf.cast(region_width / pooled_width, "int32")

        areas = [
            [
                (
                    i * h_step,
                    j * w_step,
                    (i + 1) * h_step if i + 1 < pooled_height else region_height,
                    (j + 1) * w_step if j + 1 < pooled_width else region_width,
                )
                for j in range(pooled_width)
            ]
            for i in range(pooled_height)
        ]

        def pool_area(x):
            return tf.math.reduce_max(region[x[0] : x[2], x[1] : x[3], :], axis=[0, 1])

        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
        return pooled_features
