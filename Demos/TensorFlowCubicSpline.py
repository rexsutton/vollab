#!/usr/bin/env python
"""
    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2004-2017.

    Demo cubic spline fitting using tensor flow.

    Beware plays fast and loose with dimension checks.

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def tridiag(lower, diag, upper):
    """
    Make matrix from tri-diagonal representation.

    Args:
        lower: Lower diagonal.
        diag: The diagonal.
        upper: The upper diagonal.

    Returns:
        Tri-diagonal matrix.

    """
    return np.diag(lower, -1) + np.diag(diag, 0) + np.diag(upper, 1)


def cubic_spline_xdeps(x_axis):
    """
    Calculate the dependencies on the x-axis as far as we can go on the CPU.

    Args:
        x_axis: The independent variable.

    Returns:
        A tuple, The A matrix, the differences between consecutive axis values, three constants.
    """
    # pylint: disable=invalid-name

    dx = np.diff(x_axis)
    n = x_axis.shape[0]

    A = np.zeros((3, n), x_axis.dtype)  # This is a banded matrix representation.

    A[0, 2:] = dx[:-1]                   # The upper diagonal
    A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
    A[2, :-2] = dx[1:]                   # The lower diagonal

    A[1, 0] = dx[1]
    A[0, 1] = x_axis[2] - x_axis[0]

    A[1, -1] = dx[-2]
    A[2, -2] = x_axis[-1] - x_axis[-3]

    A_upper = A[0, 1:]
    A_diag = A[1,]
    A_lower = A[2, :-1]

    A = tridiag(A_lower, A_diag, A_upper)
    return A, dx, x_axis[2] - x_axis[0], x_axis[-1] - x_axis[-3]
    # pylint: enable=invalid-name


def cubic_spline_coefficients(x, y_tensor): # pylint: disable=too-many-locals
    """
    Compute the cubic spline co-efficients given the dependent variable and
        the independent variable.

     NOTE probably fails for len(x) <4 .

    Args:
        x: The independent variable in numpy array.
        y_tensor: The dependent variable .

    Returns:
        Cubic spline co-efficients in tensors.

    """
    # pylint: disable=invalid-name
    n = x.shape[0]

    A, dx, d1, d2 = cubic_spline_xdeps(x)

    slope_tensor = y_tensor[1:] - y_tensor[:-1]

    b_beg_tensor = ((dx[0] + 2*d1) * dx[1] * slope_tensor[0:1] + dx[0]**2 * slope_tensor[1:2]) / d1
    b_mid_tensor = 3.0 * (dx[1:] * slope_tensor[:-1] + dx[:-1] * slope_tensor[1:])
    b_end_tensor = ((dx[-1]**2*slope_tensor[n-3:n-2]
                     + (2*d2 + dx[-1])*dx[-2]*slope_tensor[n-2:n-1]) / d2)

    B_tensor = tf.concat([b_beg_tensor, tf.concat([b_mid_tensor, b_end_tensor], 0)], 0)
    B_tensor = tf.reshape(B_tensor, [n, 1])

    A_tensor = tf.constant(A)
    # No tridiag solve in Tensorflow.
    s_tensor = tf.matrix_solve(A_tensor, B_tensor)
    flat_s_tensor = tf.reshape(s_tensor, [n])

    t_tensor = (flat_s_tensor[:-1] + flat_s_tensor[1:] - 2.0 * slope_tensor) / dx

    c0_tensor = t_tensor / dx
    c1_tensor = (slope_tensor - flat_s_tensor[:-1]) / dx - t_tensor
    c2_tensor = flat_s_tensor[:-1]
    c3_tensor = y_tensor[:-1]

    return c0_tensor, c1_tensor, c2_tensor, c3_tensor
    # pylint: enable=invalid-name


def tile_columns(vec, num_cols):
    """
    Build a matrix tensor of num_cols copies of a vector.
    Args:
        vec: The vector.
        num_cols: The number of columns.

    Returns:
        A matrix tensor of num_cols copies of a vector.

    """
    num_rows = vec.get_shape()[0]
    temp1 = tf.tile(vec, [num_cols])
    temp2 = tf.reshape(temp1, np.array([num_cols, num_rows]))
    return tf.transpose(temp2)


def compute_range_selection(axis, x): # pylint: disable=too-many-locals
    """

    Args:
        axis: A strictly increasing vector of floats, describing intervals.
        x: The items that are binned into the intervals.

    Returns:
        A matrix, masking the items to be be selected.

    """
    # pylint: disable=invalid-name
    axis_lower = axis[:-1]
    axis_upper = axis[1:]

    n = len(axis_lower)
    nx = int(x.get_shape()[0])

    al = tf.constant(axis_lower)
    alx = tile_columns(al, nx)

    au = tf.constant(axis_upper)
    aux = tile_columns(au, nx)

    xxx = tf.reshape(tf.tile(x, [n]), [n, nx])

    return tf.transpose(tf.logical_and(tf.less(xxx, aux), tf.greater_equal(xxx, alx)))
    # pylint: enable=invalid-name


def select(condition, items): # pylint: disable=too-many-locals
    """

    Args:
        condition: Bool matrix indicating the item intervals.
        items: The items to select from based on the interval, lenth len(axis)-1

    Returns:
        A vector of length x, of the items in q corresponding to the condition.

    """
    # pylint: disable=invalid-name
    num_rows = int(condition.get_shape()[0])
    num_cols = int(condition.get_shape()[1])
    return tf.boolean_mask(tf.reshape(tf.tile(items, [num_rows]), [num_rows, num_cols]), condition)
    # pylint: enable=invalid-name


def cubic(c0, c1, c2, c3, x): # pylint: disable=invalid-name
    """
        Evaluate the cubic polynomial given by co-efficients.
    Args:
        c0: First co-efficient.
        c1: Second co-efficient.
        c2: Third co-efficient.
        c3: Fourth co-efficient.
        x: The argument to the spline.

    Returns:
        The value of the cubic polynomial.

    """
    # pylint: disable=invalid-name
    xx = x*x
    xxx = x*xx
    return c0 * xxx + c1 * xx + c2 * x + c3
    # pylint: enable=invalid-name


def cubic_spline(independent_variable, dependent_variable, points, np_dtype=np.float32):
    """

    Args:
        independent_variable: The independent variable.
        dependent_variable: The dependent variable.
        points: The independent variables at which to evaluate the spline.
        np_dtype: The dtype.

    Returns:
        A tensor containing the values of the cubic spline at the points.

    """

    if independent_variable.dtype != np_dtype:
        raise ValueError("Axis(independent_variable)"
                         " and values(dependent_variable) must be compatible dtypes.")

    if points.dtype != dependent_variable.dtype:
        raise ValueError("Values and interpolation points must be compatible dtypes.")

    coefficients = cubic_spline_coefficients(independent_variable, dependent_variable)
    mask = compute_range_selection(independent_variable, points)
    c0_coeffs_tensor = select(mask, coefficients[0])
    c1_coeffs_tensor = select(mask, coefficients[1])
    c2_coeffs_tensor = select(mask, coefficients[2])
    c3_coeffs_tensor = select(mask, coefficients[3])

    lower_bounds = tf.constant(independent_variable[:-1])
    deltas = points - select(mask, lower_bounds)

    return cubic(c0_coeffs_tensor,
                 c1_coeffs_tensor,
                 c2_coeffs_tensor,
                 c3_coeffs_tensor,
                 deltas)


def main():
    """
    Plot an example fitting through sinusoidal, with 80,000 data points,
     and compute gradients of the interpolated points with respect to the dependent variable.

    """
    # pylint: disable=invalid-name
    x = np.arange(10.0)
    x = x.astype(np.float32, copy=False)
    y = np.sin(x)
    y = y.astype(np.float32, copy=False)
    xs = np.arange(0.5, 8.5, 0.0001)
    xs = xs.astype(np.float32, copy=False)

    y_tensor = tf.constant(y)
    xs_tensor = tf.constant(xs)
    ys_tensor = cubic_spline(x, y_tensor, xs_tensor)

    gradients_tensor = tf.gradients(ys_tensor, y_tensor)

    session = tf.Session()
    res = session.run([ys_tensor, gradients_tensor])
    ys = res[0]

    print "Num data points:", len(xs) # 80000 data points
    print "Gradients:"
    print res[1]

    plt.figure(figsize=(6.5, 4))
    plt.plot(x, y, 'o', label='data')
    plt.plot(xs, np.sin(xs), label='bench')
    plt.plot(xs, ys, 'x', label='spline')

    plt.xlim(-0.5, 9.5)
    plt.legend(loc='lower right', ncol=2)
    plt.show()
    # pylint: enable=invalid-name


if __name__ == "__main__":
    main()

