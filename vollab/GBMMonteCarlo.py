"""
    Authors: The Vollab Developers 2004-2021
    License: BSD 3 clause
"""

import numpy as np


class GBMMonteCarlo:  # pylint: disable=too-many-instance-attributes
    """
    Monte Carlo method for generating Geometric Brownian Motion.
    """

    @staticmethod
    def create_with_constant_time_step(step_size=0.1,
                                       num_steps=50,
                                       num_paths=8192,
                                       spot=50.0,
                                       drift=0.0,
                                       sigma=0.2):
        time_axis = np.linspace(0.0, num_steps * step_size, 1 + num_steps)
        return GBMMonteCarlo(time_axis, num_paths, spot, drift, sigma)

    def __init__(self,  # pylint: disable=too-many-arguments
                 time_axis,
                 num_paths=8192,
                 spot=50.0,
                 drift=0.0,
                 sigma=0.2):
        """
        Member-wise initialisation.

        Args:
            num_paths: The number of sample paths.
            spot: The starting point of the process.
            drift: The drift of the process.
            sigma: The volatility.
        """
        self.time_axis = time_axis
        self.num_paths = num_paths
        self.spot = spot
        self.drift = drift
        self.sigma = sigma

    def sample(self):
        """
        Sample the process.

        Returns:
            Samples paths of the process.
        """
        num_times = len(self.time_axis)
        num_steps = num_times - 1

        norms = np.random.normal(size=self.num_paths * num_steps).reshape([num_steps, self.num_paths])
        log_process = np.full([self.num_paths], np.log(self.spot))
        process = np.empty([num_times, self.num_paths])
        process[0, :] = self.spot

        time_steps = np.diff(self.time_axis)
        nu = self.sigma * self.sigma

        for idx, time_step in enumerate(time_steps):
            log_process += (self.drift - 0.5 * nu) * time_step + np.sqrt(nu * time_step) * norms[idx]
            process[1 + idx, :] = np.exp(log_process)

        return process
