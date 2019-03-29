"""
    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2016.

    Small re-usable functions.
"""

import datetime


def logstring():
    """ The prefix for a log-entry.
    Returns:
        (string): The log-string
    """
    return '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())


def print_log(message):
    """Print date and times followed by message.
    Args:
        message (str): The message.
    """
    print(logstring(), message)


def print_matrix_shape(message, matrix):
    """Print message followed by the shape of matrix.
    Args:
        message (str): The message.
        matrix (matrix): The matrix.
    """
    print(message, len(matrix), len(matrix[0]))


def print_vector_shape(message, vector):
    """Print message followed by the length of vector.
    Args:
        message (str): The first parameter.
        vector (vector): The vector.
    """
    print(message, len(vector))
