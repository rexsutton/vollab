"""
    Authors: The Vollab Developers 2004-2021
    License: BSD 3 clause

    Plot call prices calculated by Fast Fourier Transform.
"""

import argparse
import functools
import json

import matplotlib.pyplot as plt
import numpy as np

import env
import vollab as vl


def plot_call_prices(characteristic_function_name, params):
    """
        Display plot of European call prices.
    Args:
        characteristic_function_name: The name of the model.
        params: Dictionary of parameter values.

    """
    # create market parameters
    market_params = vl.MarketParams()
    market_params.__dict__.update(params)
    # create the characteristic function
    characteristic_function = vl.create_characteristic_function(characteristic_function_name)
    characteristic_function.__dict__.update(params)
    # select the range of strikes to plot
    strike_selector = functools.partial(vl.select_strike,
                                        0.7 * market_params.spot,
                                        1.3 * market_params.spot)
    # generate a surface at the following maturity times in years
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    selected_strikes, tenors, surface = vl.compute_call_prices_matrix(characteristic_function,
                                                                      market_params,
                                                                      strike_selector,
                                                                      tenors)
    vl.plot_surface(selected_strikes, tenors, np.transpose(surface),
                    "Call prices as a function of strike and maturity.",
                    "Strike", "Maturity", "Price")
    print("Close the plot window to continue...")
    plt.show()


def main():
    """ The main entry point function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="The stochastic model.",
                        choices=vl.characteristic_function_names(),
                        default="BlackScholes")
    parser.add_argument("-p", "--params",
                        help="The parameter dictionary.",
                        default="{}",
                        type=json.loads)

    args = parser.parse_args()
    plot_call_prices(args.model, args.params)


if __name__ == "__main__":
    main()
