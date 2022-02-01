# Python utilities for evaluating the F2 score for a set of region proposals given a ground truth annotation

import tensorflow as tf
import geometry


@tf.function
def compute_F2_scores(
    proposals,
    labels,
    thresholds=[0.3, 0.5, 0.8],
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
    thresholds : list of float
        Minimum IoU to consider a proposal and a label a match. The F2 score is
        a reduce_mean() over the set of thresholds provided.
    """

    print("Python interpreter in evaluation.compute_F2_scores()")

    # Make an analogue to functools.partial()
    def _compute_F2(data):
        return compute_FBeta_score(*data, thresholds=thresholds, beta=2.0)

    return tf.map_fn(_compute_F2, (proposals, labels), fn_output_signature=tf.float32)


def compute_FBeta_score(proposal, label, thresholds, beta):

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
                        geometry.calculate_IoU(
                            thislabel, tf.transpose(proposal, [1, 0])
                        )
                        > threshold,
                        matched == 0.0,
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
        false_pos[ithreshold] = tf.reduce_sum([1.0 - tf.convert_to_tensor(matched)])

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
