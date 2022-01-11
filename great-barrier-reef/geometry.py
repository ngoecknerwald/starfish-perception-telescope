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
    intersect = np.maximum(
        0,
        np.minimum(a[0] + a[2], b[0] + b[2]) - np.maximum(a[0], b[0]),
    ) * np.maximum(
        0,
        np.minimum(a[1] + a[3], b[1] + b[3]) - np.maximum(a[1], b[1]),
    )
    overlap = (a[2] * a[3]) + (b[2] * b[3]) - intersect

    return intersect / overlap
