# Methods for general geometry methods (i.e. IoU calculations)
# that don't belong in any specific class

import numpy as np

def center_to_boundary_coordinates(x,y,w,h):
    """
    Convert center x, y, width, height to 
    boundary xmin, xmax, ymin, ymax.
    """
    xmin = x - w / 2
    xmax = x + w / 2
    ymin = y - h / 2
    ymax = y + h / 2
    return (xmin,xmax,ymin,ymax)

def boundary_to_center_coordinates(xmin,xmax,ymin,ymax):
    """
    Convert boundary xmin, xmax, ymin, ymax
    to center x,y,width,height.
    """
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = (xmax - xmin)
    h = (ymax - ymin)
    return (x,y,w,h)

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
    xmin_a, xmax_a, ymin_a, ymax_a = boundary_to_center_coordinates(*a)
    xmin_b, xmax_b, ymin_b, ymax_b = boundary_to_center_coordinates(*b)

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
