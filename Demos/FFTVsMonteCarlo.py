"""
    Authors: The Vollab Developers 2004-2021
    License: BSD 3 clause

    Compare plain vanilla option prices obtained via FFT and Monte Carlo for the Heston model.

"""
import argparse
import functools
import inspect
import json

import matplotlib.pyplot as plt
import numpy as np

import env
import vollab as vl


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def _sample_black_scholes_monte_carlo(params):
    defaulted_params = get_default_args(vl.GBMMonteCarlo.create_with_constant_time_step)
    defaulted_params.update(params)
    monte_carlo = vl.GBMMonteCarlo.create_with_constant_time_step(**defaulted_params)
    return monte_carlo.time_axis, monte_carlo.sample()


def _sample_heston_monte_carlo(params):
    defaulted_params = get_default_args(vl.HestonMonteCarlo.create_with_constant_time_step)
    defaulted_params.update(params)
    monte_carlo = vl.HestonMonteCarlo.create_with_constant_time_step(**defaulted_params)
    return monte_carlo.time_axis, monte_carlo.sample()[0]


def _create_factory():
    return {"BlackScholes": _sample_black_scholes_monte_carlo,
            "Heston": _sample_heston_monte_carlo}


def compare_fft_with_mc(model, params):
    """
    For the specified model compare plain vanilla option prices obtained via FFT and Monte Carlo.
    Args:
        model: A string identifying the model's characteristic function.
        params: A dictionary of parameter values used to override defaults.
    """
    # create market parameters
    market_params = vl.MarketParams()
    market_params.__dict__.update(params)
    # sample the monte carlo
    tenors, samples = _create_factory()[model](params)
    # create the characteristic function
    characteristic_function = vl.create_characteristic_function(model)
    characteristic_function.__dict__.update(params)
    # select the range of strikes to plot
    strike_selector = functools.partial(vl.select_strike,
                                        0.7 * market_params.spot,
                                        1.3 * market_params.spot)
    # generate a surface of call prices using FFT
    selected_strikes, tenors, call_prices_by_fft \
        = vl.compute_call_prices_matrix(characteristic_function,
                                        market_params,
                                        strike_selector,
                                        tenors)
    # plot simulation
    vl.plot_paths(samples)
    sim_call_prices = vl.compute_sim_call_prices(selected_strikes, samples)
    # plot surfaces
    fig = plt.figure()
    vl.add_sub_surface_plot(fig, 1, 3, 1, selected_strikes, tenors, sim_call_prices.T,
                            "Monte Carlo",
                            "Strike", "Maturity", "Price")
    vl.plot_heat_map(selected_strikes, tenors, np.transpose(sim_call_prices - call_prices_by_fft),
                     "Differences.",
                     "Strike", "Maturity")
    vl.add_sub_surface_plot(fig, 1, 3, 2, selected_strikes, tenors, np.transpose(call_prices_by_fft),
                            "Fast Fourier Transform",
                            "Strike", "Maturity", "Price")

    print("Close the plot window to continue...")
    plt.show()


def main():
    """ The main entry point function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="The stochastic model.",
                        choices=[k for k in _create_factory().keys()],
                        default="Heston")
    # parameters are a dictionary.
    parser.add_argument("-p", "--params",
                        help="The parameter dictionary.",
                        default="{}",
                        type=json.loads)
    # parse args
    args = parser.parse_args()
    compare_fft_with_mc(args.model, args.params)


if __name__ == "__main__":
    main()
