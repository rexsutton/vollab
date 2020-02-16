#!/usr/bin/env python
"""

    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2004-2017.

    Compare plain vanilla option prices obtained via FFT with local volatility Monte Carlo.

"""
import argparse
import functools
import json

import matplotlib.pyplot as plt
import numpy as np

import env
import vollab as vl


def setup_local_vol(model, params):
    """

    Args:
        model: A string identifying the model's characteristic function.
        params: A dictionary of parameter values used to override defaults.

    Returns: The local volatility surface and matrix of call prices used to construct it.

    """
    # create market parameters
    market_params = vl.MarketParams()
    market_params.__dict__.update(params)
    # create the characteristic function
    characteristic_function = vl.create_characteristic_function(model)
    characteristic_function.__dict__.update(params)
    # select the range of strikes to work with
    strike_selector = functools.partial(vl.select_strike,
                                        0.7 * market_params.spot,
                                        1.3 * market_params.spot)
    # generate a surface out to five years
    tenors = np.linspace(0.0, 50 * 0.1, 1 + 50)
    # calculate call prices
    selected_strikes, tenors, call_prices_by_fft = \
        vl.compute_call_prices_matrix(characteristic_function,
                                      market_params,
                                      strike_selector,
                                      tenors)
    # calculate the local vol surface
    floored_tenors = np.array([tenor for tenor in tenors if tenor > 0.25])
    local_vol_matrix_results \
        = vl.compute_local_vol_matrix(characteristic_function,
                                      market_params,
                                      strike_selector,
                                      floored_tenors)
    # make the spline surface
    local_vol_spline_surface = vl.SplineSurface(selected_strikes,
                                                floored_tenors,
                                                local_vol_matrix_results[2],
                                                tenors)

    # print local_vol_spline_surface(market_params.spot, 0.0)
    return selected_strikes, tenors, call_prices_by_fft, local_vol_spline_surface


def compare_fft_with_local_vol_mc(model, params):
    """
    For the specified model compare plain vanilla option prices obtained via FFT,
     with local volatility Monte Carlo.

    Args:
        model: A string identifying the model's characteristic function.
        params: A dictionary of parameter values used to override defaults.
    """
    selected_strikes, tenors, call_prices_by_fft, local_vol_spline_surface \
        = setup_local_vol(model, params)
    # create the Monte Carlo
    monte_carlo = vl.LocalVolMonteCarlo(local_vol_spline_surface)
    monte_carlo.__dict__.update(params)
    simulated_stock = monte_carlo.sample()[0]
    vl.plot_paths(simulated_stock)
    # plot call prices and absolute error
    sim_call_prices = vl.compute_sim_call_prices(selected_strikes, simulated_stock)
    fig = plt.figure()
    vl.add_sub_surface_plot(fig, 1, 3, 1, selected_strikes, tenors, sim_call_prices.T,
                            "Monte Carlo.",
                            "Strike", "Maturity", "Price")
    vl.plot_heat_map(selected_strikes, tenors, np.transpose(sim_call_prices - call_prices_by_fft),
                     "Differences.",
                     "Strike", "Maturity")
    vl.add_sub_surface_plot(fig, 1, 3, 2, selected_strikes, tenors, np.transpose(call_prices_by_fft),
                            "Fast Fourier Transform.",
                            "Strike", "Maturity", "Price")

    print("Close the plot window to continue...")
    plt.show()


def main():
    """ The main entry point function.
    """
    parser = argparse.ArgumentParser()
    # parameters are a dictionary.
    parser.add_argument("-m", "--model",
                        help="The stochastic model.",
                        choices=vl.characteristic_function_names(),
                        default="Heston")
    parser.add_argument("-p", "--params",
                        help="The parameter dictionary.",
                        default="{}",
                        type=json.loads)
    # parse args
    args = parser.parse_args()
    compare_fft_with_local_vol_mc(args.model, args.params)


if __name__ == "__main__":
    main()