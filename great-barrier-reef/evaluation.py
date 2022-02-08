# Python utilities for evaluating the recall score for a set of region proposals given a ground truth annotation

import tensorflow as tf
import geometry


class TopNRegionsRecall(tf.keras.metrics.Metric):

    """
    Class wrapper around the recall score metric called when training the RPN.

    Declare that the top N regions are positive and compute a recall score
    against the ground truth. Used as a training / validation metric
    when fitting the RPN model.

    Assumes that all of the first N proposals handed to it are positive.
    This is useful when we always pass the same size list to the RoI pooling operation.


    """

    def __init__(self, N, label_decoder, name="recall" ** kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.recalls = self.add_weight(name=name, initializer="zeros")
        self.label_decoder = label_decoder

    def update_state(self, y_true, y_pred, sample_weight=None):

        print("Python interpreter in TopNRegionsRecall.update_state()")

        ignore = tf.convert_to_tensor(
            [tf.constant(False)] * self.N
            + [
                tf.constant(True),
            ]
            * (y_pred.shape[1] - self.N),
            dtype=tf.bool,
        )

        recalls = compute_recall_scores(
            y_pred, self.label_decoder(y_true), ignore, unique_ignore=False
        )

        self.recalls.assign_add(tf.reduce_mean(recalls))

    def result(self):
        return self.recalls


class ThresholdRecall(tf.keras.metrics.Metric):
    """
    Class wrapper for evaluation with a fixed classification
    score threshold.

    Declare that any region with score above threshold
    counts as a positive hit, then compute the resulting recall score.

    """

    def __init__(self, thresh, label_decoder, name="recall", **kwargs):
        super().__init__(**kwargs)
        self.thresh = thresh
        self.recalls = self.add_weight(name=name, initializer="zeros")
        self.label_decoder = label_decoder

    def update_state(self, y_true, y_pred, sample_weight=None):

        print("Python interpreter in ThresholdRecall.update_state()")

        # Check that classification scores are returned
        tf.debugging.assert_equal(y_pred.shape[2], 5)

        ignore = tf.where(
            y_pred[:, :, 4] > self.thresh,
            tf.constant(False),
            tf.constant(True),
        )

        recalls = compute_recall_scores(
            y_pred[:, :, :4], self.label_decoder(y_true), ignore, unique_ignore=True
        )

        self.recalls.assign_add(tf.reduce_mean(recalls))

    def result(self):
        return self.recalls


@tf.function
def compute_recall_scores(
    proposals, labels, ignore, IoU_thresholds=[0.3, 0.5, 0.8], unique_ignore=True
):

    """
    Compute the recall score for a batch of images.

    Arguments:

    proposals : tf.tensor(None, None, 4)
        Coordinates of the proposed regions. Shaped (image, region, xywh) in order
        of descending objectness.
    labels : tf.tensor(None, None, 4)
         Coordinates of the ground truth labels. Axes and coordinate systems should match
         the proposals argument.
    ignore : tf.tensor(None, None, 1)
        Binary tensor indicating which RoI should be ignored when
    IoU_thresholds : list of float
        Minimum IoU to consider a proposal and a label a match. The recall
        score is a reduce_mean() over the set of thresholds provided.
    unique_ignore : bool
        Use a unique ignore per image otherwise move it outside the map_fn call.
    """

    print("Python interpreter in evaluation.compute_recall_scores()")

    if unique_ignore:

        # Make an analogue to functools.partial()
        def _compute_recall_unique(data):
            return compute_recall_score(*data, thresholds=IoU_thresholds)

        return tf.map_fn(
            _compute_recall_unique,
            (proposals, labels, ignore),
            fn_output_signature=tf.float32,
        )

    else:

        # Make an analogue to functools.partial()
        def _compute_recall(data):
            return compute_recall_score(*data, ignore, thresholds=IoU_thresholds)

        return tf.map_fn(
            _compute_recall, (proposals, labels), fn_output_signature=tf.float32
        )


def compute_recall_score(proposal, label, ignore, thresholds):

    """

    Compute precision for a single image.

    Arguments:

    proposals : tf.tensor(None, 4)
        Coordinates of the proposed regions. Shaped (image, region, xywh) in order
        of descending objectness.
    labels : tf.tensor(None, 4)
         Coordinates of the ground truth labels. Axes and coordinate systems should match
         the proposals argument.
    ignore : tf.tensor(None, 1)
        Binary tensor indicating which RoI should be ignored when
    IoU_thresholds : list of float
        Minimum IoU to consider a proposal and a label a match. The recall score is
        a reduce_mean() over the set of thresholds provided.
    """

    print("Python interpreter in evaluation.compute_recall_score")

    true_pos = [
        0.0,
    ] * len(thresholds)
    false_neg = [
        1.0,
    ] * len(thresholds)

    for ithreshold, threshold in enumerate(thresholds):

        for ilabel in range(label.shape[0]):

            # An actual annotation
            if tf.reduce_sum(label[ilabel, :]) > 0.0:

                thislabel = (label[ilabel, :])[:, tf.newaxis]

                imatch = tf.where(
                    tf.logical_and(
                        geometry.calculate_IoU(
                            thislabel, tf.transpose(proposal, [1, 0])
                        )
                        > threshold,
                        tf.logical_not(ignore),
                    ),
                    1.0,
                    0.0,
                )

                if tf.reduce_sum(imatch) > 0.0:
                    true_pos[ithreshold] += 1.0
                else:
                    false_neg[ithreshold] += 1.0

    # Covert to tensors
    true_pos = tf.convert_to_tensor(true_pos)
    false_neg = tf.convert_to_tensor(false_neg)

    outval = tf.reduce_mean(true_pos / (true_pos + false_neg))

    # Corner case where there is nothing, so return precision=1 by definition
    if tf.math.is_finite(outval):
        return outval
    else:
        return 1.0
