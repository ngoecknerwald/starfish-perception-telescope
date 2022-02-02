# Python utilities for evaluating the F2 score for a set of region proposals given a ground truth annotation

import tensorflow as tf
import geometry


class TopNRegionsF2(tf.keras.metrics.Metric):

    """
    Class wrapper around the F2 score metric called when training the RPN.

    Declare that the top N regions are positive and compute an F2 score
    against the ground truth. Used as a training / validation metric
    when fitting the RPN model.

    Assumes that all of the first N proposals handed to it are positive.
    This is useful for the RPN where all we *really* care about is recall
    and always pass the same size list to the RoI pooling operation.


    """

    def __init__(self, N, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.f2s = self.add_weight(name="f2", initializer="zeros")

    def update_state(self, y_true, y_pred):

        print("Python interpreter in TopNRegionsF2.update_state")
        print("y_true")
        print(y_true)
        print("y_pred")
        print(y_pred)

        self.f2s.assign_add(
            tf.convert_to_tensor(
                [
                    0.0,
                ]
            )
        )

    def result(self):
        return self.f2s


class ThresholdF2(tf.keras.metrics.Metric):
    """
    Class wrapper for evaluation with a fixed classification
    score threshold.

    Declare that any region with score above threshold
    counts as a positive hit, then compute the resulting F2 score.

    """

    def __init__(self, thresh, **kwargs):
        super().__init__(**kwargs)
        self.thresh = thresh
        self.f2s = self.add_weight(name="f2", initializer="zeros")

    def update_state(self, y_true, y_pred):

        print("Python interpreter in TopNRegionsF2.update_state")
        print("y_true")
        print(y_true)
        print("y_pred")
        print(y_pred)

        self.f2s.assign_add(
            tf.convert_to_tensor(
                [
                    0.0,
                ]
            )
        )

    def result(self):
        return self.f2s


@tf.function
def compute_F2_scores(
    proposals,
    labels,
    ignore,
    IoU_thresholds=[0.3, 0.5, 0.8],
):

    """
    Compute the F2 score analogous to what's reported in the test set scoring.

    Arguments:

    proposals : tf.tensor(None, None, 4)
        Coordinates of the proposed regions. Shaped (image, region, xywh) in order
        of descending objectness. Assumes that some "minimum objectness" cut has been
        applied.
    labels : tf.tensor(None, None, 4)
         Coordinates of the ground truth labels. Axes and coordinate systems should match
         the proposals argument.
    ignore : tf.tensor(None, None, 1)
        Binary tensor indicating which RoI should be ignored when
    IoU_thresholds : list of float
        Minimum IoU to consider a proposal and a label a match. The F2 score is
        a reduce_mean() over the set of thresholds provided.
    """

    print("Python interpreter in evaluation.compute_F2_scores()")

    # Make an analogue to functools.partial()
    def _compute_F2(data):
        return compute_FBeta_score(*data, thresholds=IoU_thresholds, beta=2.0)

    return tf.map_fn(
        _compute_F2, (proposals, labels, ignore), fn_output_signature=tf.float32
    )


def compute_FBeta_score(proposal, label, ignore, thresholds, beta):

    """
    Note that this is different from the usual F2 score in that if two duplicate
    proposals match a ground truth object they are *both* scored as true positives
    and not only one of them. This is more applicable to testing the RPN on the validation
    set because we expect some duplicate proposals for IoU suppression.
    """

    print("Python interpreter in evaluation.compute_FBeta_score")

    true_pos = [
        0.0,
    ] * len(thresholds)
    false_pos = [
        0.0,
    ] * len(thresholds)
    false_neg = [
        1.0,
    ] * len(thresholds)

    for ithreshold, threshold in enumerate(thresholds):

        # Track if a proposal has been matched
        matched = tf.zeros(
            [
                proposal.shape[0],
            ],
            dtype=tf.float32,
        )

        for ilabel in range(label.shape[0]):

            # An actual annotation
            if tf.reduce_sum(label[ilabel, :]) > 0.0:

                thislabel = (label[ilabel, :])[:, tf.newaxis]

                imatch = tf.where(
                    tf.logical_and(
                        tf.logical_and(
                            geometry.calculate_IoU(
                                thislabel, tf.transpose(proposal, [1, 0])
                            )
                            > threshold,
                            matched == 0.0,
                        ),
                        tf.logical_not(ignore),
                    ),
                    1.0,
                    0.0,
                )

                if tf.reduce_sum(imatch) > 0.5:
                    true_pos[ithreshold] += tf.reduce_sum(imatch)
                    # Hacky bitwise or with floats
                    matched += imatch
                    matched = tf.minimum(matched, 1.0)
                else:
                    false_neg[ithreshold] += 1.0

        # False positive
        false_pos[ithreshold] = tf.reduce_sum(
            [(1.0 - tf.convert_to_tensor(matched)) * tf.cast(ignore, tf.float32)]
        )

    # Covert to tensors
    true_pos = tf.convert_to_tensor(true_pos)
    false_pos = tf.convert_to_tensor(false_pos)
    false_neg = tf.convert_to_tensor(false_neg)

    outval = tf.reduce_mean(
        ((1.0 + beta ** 2) * true_pos)
        / (((1.0 + beta ** 2) * true_pos) + (beta ** 2 * false_neg) + (false_pos))
    )

    # Corner case where there is nothing, and we return no hits means F_Beta = 1 by definition
    if tf.math.is_finite(outval):
        return outval
    else:
        return 1.0
