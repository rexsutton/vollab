"""
    Authors: The Vollab Developers 2004-2021
    License: BSD 3 clause

    Calculate the implied volatility surface in two steps:
        1. For a given characteristic function use Fast Fourier Transform to get call price surface.
        2. Uses the Lets Be Rational for fast calculation of Black-Scholes implied volatility.

    Uses:-
    Jäckel, Peter. "Let's be rational." Wilmott 2015.75 (2015): 40-53.
"""

import sys
import numpy as np
import lets_be_rational as lb

from .FFTEuropeanCallPrice import *


def compute_smile(call_price_calculator,
                  characteristic_function,
                  market_params,
                  maturity_time,
                  strike_selector):
    """
        Calculate the smile for a given maturity time for selected strikes.
    Args:
        call_price_calculator: The call price calculator.
        characteristic_function: The characteristic function.
        market_params: Market parameters.
        maturity_time: Measured in years relative to today, today being zero.
        strike_selector: Predicate for selecting strikes.

    Returns:
        The list of implied volatilities in order of increasing strike.

    """
    if maturity_time < 0.01:
        raise IndexError("Tenor below 0.01")

    # calculate the forward
    forward = market_params.forward(maturity_time)
    # calculate inverse discount factor
    inverse_discount_factor = np.exp(market_params.short_rate * maturity_time)
    # calculate the call prices
    call_prices = call_price_calculator.compute_call_prices(market_params,
                                                            maturity_time,
                                                            characteristic_function)
    # calculate the surface values
    smile = []
    last_good_vol = 0.0
    # for selected strikes
    for strike, call_price in zip(call_price_calculator.strike_axis, call_prices):
        if strike_selector(strike):
            # calculate the implied vol
            implied_volatility = lb.implied_volatility_from_a_transformed_rational_guess(
                call_price * inverse_discount_factor,
                forward,
                strike,
                maturity_time,
                1.0)
            # VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_BELOW_INTRINSIC = -DBL_MAX
            # VOLATILITY_VALUE_TO_SIGNAL_PRICE_IS_ABOVE_MAXIMUM = DBL_MAX
            if implied_volatility == sys.float_info.min \
                    or implied_volatility == -sys.float_info.max:
                implied_volatility = last_good_vol
            else:
                last_good_vol = implied_volatility
            smile.append(implied_volatility)
            # append
    return smile


def compute_implied_vol_surface(characteristic_function,
                                market_params,
                                strike_selector,
                                maturity_times):
    """
    Calculate a matrix of implied volatility.

    Args:
        characteristic_function: The characteristic function.
        market_params:The market parameters.
        strike_selector: Predicate function for selecting strikes.
        maturity_times: The maturity times of interest.

    Returns:
        A matrix of call prices at the selected strikes and maturity times.

    """
    # create the call price calculator
    call_price_calculator = CallPriceCalculator(num_points=4096, lmbda=0.005, alpha=1.0)
    # select the range of strikes to plot
    selected_strikes = [strike for strike in call_price_calculator.strike_axis
                        if strike_selector(strike)]
    # generate a surface for the following maturity times in years
    surface = []
    for tenor in maturity_times:
        # append smile to the surface
        surface.append(compute_smile(call_price_calculator,
                                     characteristic_function,
                                     market_params,
                                     tenor,
                                     strike_selector))
    return selected_strikes, maturity_times, surface
