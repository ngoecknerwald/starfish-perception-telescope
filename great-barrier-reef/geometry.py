# Methods for general geometry methods (i.e. IoU calculations)
# that don't belong in any specific class

import numpy as np


def calculate_IoU(a, b):
    """
    Calculate the intersection over union for two boxes
    or arrays of boxes.

    Arguments:

    a : length-4 array-like, or numpy array of dimension (4,N,M,...)
        given as (xa, ya, wa, ha)
    b : length-4 array-like, or numpy array of dimension (4,N,M,...)
        given as (xb, yb, wb, hb)

    Returns:

    IoU : scalar or array of dimension (N, M,...)

    """
    xmin_a, xmax_a = a[0] - a[2] / 2, a[0] + a[2] / 2
    ymin_a, ymax_a = a[1] - a[3] / 2, a[1] + a[3] / 2
    xmin_b, xmax_b = b[0] - b[2] / 2, b[0] + b[2] / 2
    ymin_b, ymax_b = b[1] - b[3] / 2, b[1] + b[3] / 2

    intersect = np.maximum(
        0,
        1 + (np.minimum(xmax_a, xmax_b) - np.maximum(xmin_a, xmin_b)),
    ) * np.maximum(
        0,
        1 + (np.minimum(ymax_a, ymax_b) - np.maximum(ymin_a, ymin_b)),
    )
    overlap = (
        (xmax_a - xmin_a) * (ymax_a - ymin_a)
        + (xmax_b - xmin_b) * (ymax_b - ymin_b)
        - intersect
    )

    return intersect / overlap
