# Class for converting from the RPN outputs to the tail network inputs
import tensorflow as tf


class IoU_supression:
    def __init__(self, IoU_threshold=0.7):

        '''
        Instantiate an IoU supression call. Designed
        to remove duplicate RoIs from the stack produced
        by the RPN.

        Arguments:

        IoU_threshold : float
            Threshold above which two returned regions of interest
            are deemed the same underlying object.

        '''

        self.IoU_threshold = IoU_threshold

    def call(roi_list_sort):

        '''
        Reduce an RoI list by removing any entry with an IoU greater
        than a higher ranked RoI.

        Arguments

        roi_list_sort : list
            List of RoIs sorted by likelihood of being a ground truth starfish.

        '''

        pass


class ROIPooling(tf.keras.layers.Layer):

    # mostly taken from https://medium.com/xplore-ai/implementing-attention-in-tensorflow-keras-using-roi-pooling-992508b6592b

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
        h_start = tf.math.maximum(tf.cast(yy - hh / 2, 'int32'), 0)
        h_end = tf.math.minimum(tf.cast(yy + hh / 2, 'int32'), feature_map.shape[0])
        w_start = tf.math.maximum(tf.cast(xx - ww / 2, 'int32'), 0)
        w_end = tf.math.minimum(tf.cast(xx + ww / 2, 'int32'), feature_map.shape[1])

        # Enlarge RoI as needed
        hpad = pooled_height - (h_end - h_start)
        if hpad > 0:
            hpad = tf.cast(tf.math.ceil(hpad/2), 'int32')
            top = feature_map.shape[0] - h_end
            bottom = h_start
            if top < hpad:
                h_end += top
                h_start -= (2*hpad - top)
            elif bottom < hpad:
                h_end += (2*hpad - bottom)
                h_start -= bottom
            else:
                h_start -= hpad
                h_end += hpad

        wpad = pooled_width - (w_end - w_start)
        if wpad > 0:
            wpad = tf.cast(tf.math.ceil(wpad/2), 'int32')
            right = feature_map.shape[1] - w_end
            left = w_start
            if right < wpad:
                w_end += right 
                w_start -= (2*wpad - right)
            elif left < wpad:
                w_end += (2*wpad - left)
                w_start -= left 
            else:
                w_start -= wpad
                w_end += wpad

        region = feature_map[h_start:h_end, w_start:w_end, :]

        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width = w_end - w_start
        h_step = tf.cast(region_height / pooled_height, 'int32')
        w_step = tf.cast(region_width / pooled_width, 'int32')

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
