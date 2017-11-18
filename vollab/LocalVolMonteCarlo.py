"""

    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2004-2017.

    Monte Carlo method for generating samples from a process on the local vol surface.

"""

import numpy as np


class LocalVolMonteCarlo(object): #pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    Monte Carlo method for generating samples from a process on the local vol surface.
    """

    def __init__(self, #pylint: disable=too-many-arguments
                 local_vol_surface,
                 step_size=0.1,
                 num_steps=50,
                 num_paths=(4*2048),
                 spot=50.0,
                 drift=0.0):
        """
        Simple initialisation.

        Args:
            local_vol_surface: The local volatility surface.
            step_size: The MonteCarlo step size in years.
            num_steps: The number of steps.
            num_paths: The number of sample paths.
            spot: The starting point of the process.
            drift: The drift of the process.
        """
        self.local_vol_surface = local_vol_surface
        self.step_size = step_size
        self.num_steps = num_steps
        self.num_paths = num_paths
        self.spot = spot
        self.drift = drift
        self.sqrt_step_size = np.sqrt(self.step_size)
        self.times = np.linspace(step_size, num_steps * step_size, num_steps)
        self.time_axis = np.linspace(0.0, num_steps*step_size, 1 + num_steps)
        self.str_time_axis = [str(t) for t in self.time_axis]
        self.local_vol_zero = self.local_vol_surface(self.spot, 0.0)


    def sample(self):
        """
        Returns:
            Samples of the process and the volatility.

        """
        norms = np.random.normal(0.0, 1.0, self.num_paths * self.num_steps)
        norms = norms.reshape([self.num_steps, self.num_paths])

        num_times = 1 + self.num_steps
        log_process = np.full([self.num_paths], np.log(self.spot))
        process = np.empty([num_times, self.num_paths])
        process[0][:] = np.full([self.num_paths], self.spot)
        volatility = np.empty([num_times, self.num_paths])
        volatility[0][:] = np.full([self.num_paths], self.local_vol_zero)

        idx_time = 1
        for time in self.times:
            sigma = volatility[idx_time-1]
            log_process += (self.drift - 0.5 * sigma * sigma) * self.step_size\
                         + self.sqrt_step_size  * sigma * norms[idx_time-1]
            process[idx_time][:] = np.exp(log_process)
            volatility[idx_time][:] = self.local_vol_surface(process[idx_time], time)
            idx_time += 1

        return process, volatility
