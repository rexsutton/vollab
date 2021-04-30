"""
    Authors: The Vollab Developers 2004-2021
    License: BSD 3 clause
"""


import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors


def plot_paths(paths):
    """
    Plot paths matrix generated by Monte Carlo.
    Args:
        paths: The paths matrix.
    """
    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.plot(paths)
    sub.title.set_text("Sample paths.")
    sub.set_ylabel("Values")
    sub.set_xlabel("Time Step")


def add_sub_surface_plot(fig, num_rows, num_cols, plot_number, x_axis, y_axis, z_values,
                         title=None, x_label=None, y_label=None, z_label=None, indexing='ij', tol=1e-8):
    """
        Plot a surface using MatPlotLib.
        Default indexing is matrix.
        Tolerance controls what appears to be a bug when plotting flat surfaces.
    Args:
        fig: The figure.
        num_rows: The number of rows in the figure.
        num_cols: The number of cols in the figure.
        plot_number: The number of the sub plot in the figure.
        x_axis: The x axis.
        y_axis: The y axis.
        z_values: The matrix of surface values.
        title: The title.
        x_label: The x label.
        y_label: The y label.
        z_label: The z label.
        indexing: 'ij' or 'xy', matrix or cartesian.
        tol: Tolerance for rounding numbers to fix flat-surface display.
    """
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis, indexing=indexing)
    z_matrix = np.floor(np.array(z_values) / tol) * tol
    sub = fig.add_subplot(num_rows, num_cols, plot_number, projection='3d')
    sub.plot_surface(x_mesh, y_mesh, z_matrix, cmap=matplotlib.cm.plasma)
    if title:
        sub.title.set_text(title)
    if x_label:
        sub.set_xlabel(x_label)
    if y_label:
        sub.set_ylabel(y_label)
    if z_label:
        sub.set_zlabel(z_label)


def plot_surface(x_axis, y_axis, z_values,
                 title=None, x_label=None, y_label=None, z_label=None,
                 indexing='ij',
                 tol=1e-8):
    """
        Plot a surface using MatPlotLib.
        Default indexing is matrix.
        Tolerance controls what appears to be a bug when plotting flat surfaces.
    Args:
        x_axis: The x axis.
        y_axis: The y axis.
        z_values: The matrix of surface values.
        title: The title of the plot.
        x_label: The x axis label.
        y_label: The y axis label.
        z_label: The z axis label.
        indexing: 'ij' or 'xy', matrix or cartesian.
        tol: Tolerance for rounding numbers to precision, to fix flat-surface display.
    """
    fig = plt.figure()
    add_sub_surface_plot(fig, 1, 1, 1, x_axis, y_axis, z_values,
                         title, x_label, y_label, z_label,
                         indexing, tol)


def add_heat_map(fig, num_rows, num_cols, plot_number, x_axis, y_axis, z_values,
                 title=None, x_label=None, y_label=None,
                 x_step=None, y_step=None,
                 zero=False):
    """
        Plot a heat map using MatPlotLib.
        Default indexing is matrix.
        Tolerance controls what appears to be a bug when plotting flat surfaces.
    Args:
        fig: The figure.
        num_rows: The number of rows in the figure.
        num_cols: The number of cols in the figure.
        plot_number: The number of the sub plot in the figure.
        x_axis: The x axis.
        y_axis: The y axis.
        z_values: The matrix of surface values.
        title: The title.
        x_label: The x label.
        y_label: The y label.
        x_step: The ticks on the x axis.
        y_step: The ticks on the y axis.
        zero: If true normalize colours around zero.
    """
    norm = None
    if zero:
        temp = z_values.T.flatten()
        mn = np.amin(temp)
        mx = np.amax(temp)
        if (mn < 0.0) and (mx > 0.0):
            mag = max(abs(mn), mx)
            norm = colors.TwoSlopeNorm(vmin=-mag,
                                       vcenter=0.,
                                       vmax=mag)

    sub = fig.add_subplot(num_rows, num_cols, plot_number)
    im = sub.imshow(z_values.T, cmap=matplotlib.cm.plasma, norm=norm, interpolation='nearest', origin='lower')
    if title:
        sub.title.set_text(title)
    if x_label:
        sub.set_xlabel(x_label)
    if y_label:
        sub.set_ylabel(y_label)
    fig.colorbar(im)

    if x_step is None:
        x_step = (len(x_axis) // 4)

    if y_step is None:
        y_step = (len(y_axis) // 4)

    x_ticks = np.arange(len(x_axis))[::x_step]
    x_tick_labels = ["{0:.0f}".format(x_axis[i]) for i in x_ticks]
    sub.set_xticks(x_ticks)
    sub.set_xticklabels(x_tick_labels)

    y_ticks = np.arange(len(y_axis))[::y_step]
    y_tick_labels = ["{0:.0f}".format(y_axis[i]) for i in y_ticks]
    sub.set_yticks(y_ticks)
    sub.set_yticklabels(y_tick_labels)


def plot_heat_map(x_axis, y_axis, z_values,
                  title=None, x_label=None, y_label=None,
                  x_step=None, y_step=None,
                  zero=False):
    """
        Plot a surface using MatPlotLib.
        Default indexing is matrix.
    Args:
        x_axis: The x axis.
        y_axis: The y axis.
        z_values: The matrix of surface values.
        title: The title.
        x_label: The x label.
        y_label: The y label.
        x_step: The ticks on the x axis.
        y_step: The ticks on the y axis.
        zero: If true normalize colours around zero.
    """
    fig = plt.figure()
    add_heat_map(fig, 1, 1, 1, x_axis, y_axis, z_values,
                 title, x_label, y_label,
                 x_step, y_step, zero)
