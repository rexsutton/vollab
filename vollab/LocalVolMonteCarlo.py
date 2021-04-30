"""
    Authors: The Vollab Developers 2004-2021
    License: BSD 3 clause
"""

import numpy as np

from .SplineSurface import *


class LocalVolMonteCarlo(object):  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    Monte Carlo method for generating samples from a process on the local vol surface.
    """

    @staticmethod
    def create_with_constant_time_step(
                 local_vol_surface,
                 step_size=0.1,
                 num_steps=50,
                 num_paths=8192,
                 spot=50.0,
                 drift=0.0):
        time_axis = np.linspace(0.0, num_steps * step_size, 1 + num_steps)
        return LocalVolMonteCarlo(local_vol_surface, time_axis, num_paths, spot, drift)

    def __init__(self,  # pylint: disable=too-many-arguments
                 local_vol_surface,
                 time_axis,
                 num_paths=8192,
                 spot=50.0,
                 drift=0.0):
        """
        Simple initialisation.

        Args:
            local_vol_surface: The local volatility surface.
            time_axis: The simulation time axis.
            num_paths: The number of sample paths.
            spot: The starting point of the process.
            drift: The drift of the process.
        """
        self.time_axis = time_axis
        self.num_paths = num_paths
        self.spot = spot
        self.drift = drift
        self.local_vol_surface = SplineSurface(local_vol_surface.x_axis, local_vol_surface.y_axis,
                                               local_vol_surface.surface_matrix,
                                               self.time_axis)
        self.local_vol_zero = self.local_vol_surface(self.spot, 0.0)

    def sample(self):
        """
        Returns:
            Samples of the process and the volatility.

        """
        num_times = len(self.time_axis)
        num_steps = num_times - 1

        norms = np.random.normal(0.0, 1.0, self.num_paths * num_steps)
        norms = norms.reshape([num_steps, self.num_paths])

        log_process = np.full([self.num_paths], np.log(self.spot))
        process = np.empty([num_times, self.num_paths])
        process[0, :] = np.full([self.num_paths], self.spot)
        volatility = np.empty([num_times, self.num_paths])
        volatility[0, :] = np.full([self.num_paths], self.local_vol_zero)

        time_steps = np.diff(self.time_axis)

        idx_time = 1
        for time, time_step in zip(self.time_axis[1:], time_steps):
            sigma = volatility[idx_time - 1]
            log_process += (self.drift - 0.5 * sigma * sigma) * time_step \
                           + np.sqrt(time_step) * sigma * norms[idx_time - 1]
            process[idx_time][:] = np.exp(log_process)
            volatility[idx_time][:] = self.local_vol_surface(process[idx_time], time)
            idx_time += 1

        return process, volatility
