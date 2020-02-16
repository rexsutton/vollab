"""
    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2016.

    Small re-usable functions.
"""
import datetime
import numpy as np


def compute_sim_call_price(strike, simulated_process):
    """
    Compute call price from simulated values.
    Args:
        strike: The strike.
        simulated_process: Vector of simulated values by time.

    Returns: The call prices.

    """
    temp = simulated_process - strike
    return np.mean(np.where(np.less(temp, 0.0), 0.0, temp))


def compute_sim_call_prices(strikes, simulated_process):
    """
    Compute call prices from simulated values.
    Args:
        strikes: The strikes.
        simulated_process: Matrix of simulated values by time.

    Returns: A matrix of call prices.

    """
    sim_call_prices = np.empty([len(simulated_process), len(strikes)])
    for idx_sim, simulations in enumerate(simulated_process):
        for idx_strike, strike in enumerate(strikes):
            sim_call_prices[idx_sim, idx_strike] = compute_sim_call_price(strike, simulations)

    return sim_call_prices


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
