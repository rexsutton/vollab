#!/usr/bin/env python
"""

    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2004-2017.


    Plot call prices calculated by Fast Fourier Transform.

"""
import argparse
import functools
import json

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import env
import vollab as vl


def plot_surface(x_axis, y_axis, z_values, tol=1e-8):
    """
        Plot a surface using MatPlotLib.
        Tolerance controls what appears to be a bug when plotting flat surfaces.
    Args:
        x_axis: The x axis.
        y_axis: The y axis.
        z_values: The matrix of surface values.
        tol: Tolerance for rounding numbers to precision.
    """
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis, indexing='ij')
    z_matrix = np.zeros([len(x_axis), len(y_axis)], dtype=np.float64)

    for i in range(0, len(x_axis)):
        for j in range(0, len(y_axis)):
            z_matrix[i, j] = np.floor(z_values[i][j] / tol)*tol

    fig = plt.figure()
    sub = fig.add_subplot(111, projection='3d')
    sub.plot_surface(x_mesh, y_mesh, z_matrix)
    sub.title.set_text("Call prices as a function of strike and maturity.")
    sub.set_xlabel("Strike")
    sub.set_ylabel("Maturity")
    sub.set_zlabel("Price")
    print "Close the plot window to continue..."
    plt.show()


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
    # # generate a surface __call__ the following maturity times in years
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    selected_strikes, tenors, surface = vl.compute_call_prices_matrix(characteristic_function,
                                                                      market_params,
                                                                      strike_selector,
                                                                      tenors)
    plot_surface(selected_strikes, tenors, np.transpose(surface))


def main():
    """ The main entry point function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="The stochastic model.",
                        default="Heston")
    parser.add_argument("-p", "--params",
                        help="The parameter dictionary.",
                        default="{}",
                        type=json.loads)

    args = parser.parse_args()
    plot_call_prices(args.model, args.params)


if __name__ == "__main__":
    main()
