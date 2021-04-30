"""
    Authors: The Vollab Developers 2004-2021
    License: BSD 3 clause

    Calculate the local volatility surface for given characteristic function in three steps:
        1. For a given characteristic function use Fast Fourier Transform to get call price surface.
        2. Uses the Lets Be Rational function for fast calculation of implied volatility.
        3. Construct local volatility using Dupire's formula and cubic spline interpolation.

"""

import numpy as np

from scipy.interpolate import CubicSpline

from .ImpliedVolatilitySurface import *

from .SplineSurface import SplineSurface
from .FFTEuropeanCallPrice import compute_call_prices_matrix


def change_variables(market_params, maturity_time, strikes, smile):
    """
        Change variables to log-strike and variance.

    Args:
        market_params: The market parameters.
        maturity_time: The maturity times.
        strikes: The strikes of the smile.
        smile: The implied volatility at the strikes.

    Returns:
        A pair of numpy arrays of equal size, log-strikes, variance.

    """

    forward = market_params.forward(maturity_time)

    rel_log_strikes = np.array(strikes)
    rel_log_strikes /= forward
    rel_log_strikes = np.log(rel_log_strikes)

    variance = np.square(smile)
    variance *= maturity_time
    return rel_log_strikes, variance


def compute_derivatives(log_strikes, variances):
    """

    Args:
        log_strikes: An array of log strikes.
        variances: An array of variances.

    Returns:
        A tuple, a cubic spline through the data,
         array of the first derivatives,
         array of the second derivatives.

    """
    spline = CubicSpline(log_strikes, variances)
    deriv_1 = spline(variances, 1)
    deriv_2 = spline(variances, 2)
    return deriv_1, deriv_2


def compute_denominator(log_strike, variance, deriv_1, deriv_2):
    """

    Compute denominator in local vol equation.

    Args:
        log_strike: The log-strike.
        variance: The variance.
        deriv_1: The first derivative of variance wrt log strike.
        deriv_2: The second derivative of variance wrt log strike.

    Returns:
        The denominator in local vol equation.

    """
    return 1.0 - (log_strike * deriv_1 / variance) \
           + 0.25 * (-0.25 - (1.0 / variance)
                     + (log_strike * log_strike / variance * variance)) \
           * (deriv_1 * deriv_1) \
           + 0.5 * deriv_2


def compute_denominator_row(market_params, strikes, maturity_time, smile):
    """
    Compute the denominator in local vol equation for a given maturity time.

    Args:
        market_params:The market parameters.
        strikes: The strikes.
        maturity_time: The tenor.
        smile: The smile.

    Returns:

    """
    row = []
    log_strikes, variances = change_variables(market_params, maturity_time, strikes, smile)
    derivs_1, derivs_2 = compute_derivatives(log_strikes, variances)
    for log_strike, variance, deriv_1, deriv_2 in zip(log_strikes, variances, derivs_1, derivs_2):
        row.append(compute_denominator(log_strike, variance, deriv_1, deriv_2))
    return row


def compute_denominator_matrix(market_params, strikes, maturity_times, implied_vol_surface):
    """

    Args:
        market_params: The market params.
        strikes: The strikes.
        maturity_times: The maturity times.
        implied_vol_surface: The implied vol surface matrix.

    Returns:

    """
    matrix = []
    for tenor, smile in zip(maturity_times, implied_vol_surface):
        matrix.append(compute_denominator_row(market_params, strikes, tenor, smile))
    return matrix


def compute_dvariance_dmaturity(market_params, strikes, tenors, implied_vol_surface):
    """
    Calculate the matrix of derivatives in variance by maturity.

    Args:
        characteristic_function: The characteristic function.
        market_params:The market parameters.
        strike_selector: Predicate function for selecting strikes.
        tenors: The maturity times of interest.

    Returns:
        A matrix of local volatility for the given strikes and tenors.

    """
    smile_splines = []
    for tenor, smile in zip(tenors, implied_vol_surface):
        dummy, variances = change_variables(market_params, tenor, strikes, smile)
        smile_splines.append(CubicSpline(strikes, variances))

    tenor_splines = []
    for strike in strikes:
        variances = []
        for spline in smile_splines:
            variances.append(spline(strike))
        tenor_splines.append(CubicSpline(tenors, variances))

    dvariance_dmaturity = []
    for tenor in tenors:
        row = []
        for idx_strike, strike in enumerate(strikes):
            row.append(tenor_splines[idx_strike](tenor, 1))
        dvariance_dmaturity.append(row)

    return dvariance_dmaturity


def compute_local_vol_matrix(characteristic_function,
                             market_params,
                             strike_selector,
                             maturity_times):
    """
    Calculate a matrix of local volatility at the given strikes and maturity_times.

    Args:
        characteristic_function: The characteristic function.
        market_params:The market parameters.
        strike_selector: Predicate function for selecting strikes.
        maturity_times: The maturity_times of interest.

    Returns:
        A matrix of local volatility at the given strikes and maturity_times.

    """
    strikes, maturity_times, implied_vol_surface = \
        compute_implied_vol_surface(characteristic_function,
                                    market_params,
                                    strike_selector,
                                    maturity_times)

    denom_matrix = compute_denominator_matrix(market_params,
                                              strikes,
                                              maturity_times,
                                              implied_vol_surface)

    dvariance_dtenor_matrix = compute_dvariance_dmaturity(market_params,
                                                          strikes,
                                                          maturity_times,
                                                          implied_vol_surface)

    local_vol_surface = np.empty([len(maturity_times), len(strikes)])
    for idx_tenor in range(len(maturity_times)):
        for idx_strike in range(len(strikes)):
            dvariance_dtenor = dvariance_dtenor_matrix[idx_tenor][idx_strike]
            denom = denom_matrix[idx_tenor][idx_strike]
            local_vol_surface[idx_tenor][idx_strike] = np.sqrt(dvariance_dtenor / denom)

    return strikes, maturity_times, local_vol_surface


def compute_local_vol_spline_surface(characteristic_function,
                                     market_params,
                                     strike_selector,
                                     maturity_times,
                                     maturity_time_floor=0.25):
    """
    Compute the local volatility surface.

    Args:
        characteristic_function: The characteristic function.
        market_params:The market parameters.
        strike_selector: Predicate function for selecting strikes.
        maturity_times: The maturity times of the surface.
        maturity_time_floor: Times smaller than this will be extrapolated flat.

    Returns: A tuple of: strikes, maturities, call prices, local vol as a spline surface

    """
    # calculate call prices
    selected_strikes, maturity_times, call_prices_by_fft = \
        compute_call_prices_matrix(characteristic_function,
                                   market_params,
                                   strike_selector,
                                   maturity_times)
    # calculate the local vol surface
    floored_maturity_times = np.array([t for t in maturity_times if t > maturity_time_floor])
    local_vol_matrix_results = compute_local_vol_matrix(characteristic_function,
                                                        market_params,
                                                        strike_selector,
                                                        floored_maturity_times)
    # make the spline surface
    local_vol_spline_surface = SplineSurface(selected_strikes,
                                             floored_maturity_times,
                                             local_vol_matrix_results[2],
                                             maturity_times)

    return selected_strikes, maturity_times, call_prices_by_fft, local_vol_spline_surface

