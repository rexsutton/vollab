#!/usr/bin/env python
"""

    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2004-2017.

    Plot implied volatility calculated by Fast Fourier Transform.

    Uses the Lets Be Rational library for fast calculaiton of Black-Scholes implied volatility.

"""
import argparse
import json
import functools

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
    # round surface numbers to tolerance
    z_floored = np.zeros([len(x_axis), len(y_axis)], dtype=np.float64)
    for i in range(0, len(x_axis)):
        for j in range(0, len(y_axis)):
            z_floored[i, j] = np.floor(z_values[i][j] / tol)*tol
    # do the plotting
    fig = plt.figure()
    sub = fig.add_subplot(111, projection='3d')
    sub.plot_surface(x_mesh, y_mesh, z_floored)
    sub.title.set_text("Local volatility as a function of strike and maturity.")
    sub.set_xlabel("Strike")
    sub.set_ylabel("Maturity")
    sub.set_zlabel("Volatility")
    print "Close the plot window to continue..."
    plt.show()


def plot_local_vol(characteristic_function_name, params):
    """
        Display plot of local volatility surface.
    Args:
        characteristic_function_name: The name of the model.
        params: Dictionary of parameter values.
    """
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
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
    # calculate the local vol surface
    strikes, tenors, local_vol_surface = vl.compute_local_vol_matrix(characteristic_function,
                                                                     market_params,
                                                                     strike_selector,
                                                                     tenors)
    #print "implied vol"
    #plot_surface(tenors, strikes, implied_vol_surface, tol)
    tol = 1e-3 if characteristic_function_name == 'BlackScholes' else 1e-8
    plot_surface(strikes, tenors, np.transpose(local_vol_surface), tol)


def main():
    """ The main entry point function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="The stochastic model",
                        default="Heston")
    # parameters are a dictionary.
    parser.add_argument("-p", "--params",
                        help="The parameter dictionary.",
                        default="{}",
                        type=json.loads)
    # parse args
    args = parser.parse_args()
    # do the plot
    plot_local_vol(args.model, args.params)


if __name__ == "__main__":
    main()
