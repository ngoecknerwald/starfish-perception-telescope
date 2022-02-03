# Methods for general geometry methods (i.e. IoU calculations)
# that don't belong in any specific class

import tensorflow as tf


@tf.function
def calculate_IoU(a, b):
    """
    Calculate the intersection over union for two boxes
    or arrays of boxes.

    Arguments:

    a : length-4 array-like, or tf.tensor of dimension (4,N,M,...)
        given as (xa, ya, wa, ha)
    b : length-4 array-like, or tf.tensor of dimension (4,N,M,...)
        given as (xb, yb, wb, hb)

    Returns:

    IoU : scalar or array of dimension (N, M,...)

    """

    print("Python interpreter in geometry.calculate_Iou()")

    intersect = tf.math.maximum(
        0.0,
        tf.math.minimum(a[0] + a[2], b[0] + b[2]) - tf.math.maximum(a[0], b[0]),
    ) * tf.math.maximum(
        0.0,
        tf.math.minimum(a[1] + a[3], b[1] + b[3]) - tf.math.maximum(a[1], b[1]),
    )
    overlap = (a[2] * a[3]) + (b[2] * b[3]) - intersect

    return intersect / overlap


@tf.function
def safe_exp(x):

    """
    Exponential stitched to a linear function.
    Implemented to make the bounding box regresion
    wear a helmet when riding a bike. Returns:

    e^x : x < 0
    x + 1 : x >= 0

    Arguments:

    x : tf.tensor
        Input variables.

    """

    print("Python interpreter in geometry.safe_exp()")

    return tf.nn.elu(x) + 1.0


@tf.function
def safe_log(x):

    """
    Inverse of the safe_exp() function. Returns:

    log(x) : x < 1
    x - 1. x >= 1

    Arguments:
    x : tf.tensor of shape 0
        Input variable

    """

    print("Python interpreter in geometry.safe_log()")

    if x < 1.0:
        return tf.math.log(x)
    return x - 1.0


@tf.function
def batch_sort(arr, inds, n):
    """
    Sort a flattened tensor arr by indices inds returning the first n.

    Note that this rebuilds a computation graph when n changes.
    """

    print("Python interpreter in geometry.batch_sort()")
    assert len(tf.shape(arr)) == 2

    return tf.gather(arr, inds, batch_dims=1)[:, :n]
