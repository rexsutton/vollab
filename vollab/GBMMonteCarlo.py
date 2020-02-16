"""
    !!! Not certified fit for any purpose, use at your own risk !!!

    Copyright (c) Rex Sutton 2004-2017.

    Monte Carlo method for generating geometric brownian motion.

"""

import numpy as np


class ABMMonteCarlo:

    def __init__(self,
                 step_size=0.1,
                 num_steps=50,
                 num_paths=(4 * 2048),
                 spot=50.0,
                 drift=0.0,
                 nu=0.2*0.2):

        self.spot = spot
        self.drift = drift
        self.nu = nu
        self.step_size = step_size
        self.num_steps = num_steps
        self.num_paths = num_paths
        self.time = np.linspace(step_size, num_steps * step_size, num_steps)
        self.time_axis = np.linspace(0.0, num_steps * step_size, 1 + num_steps)
        self.str_time_axis = [str(t) for t in self.time_axis]
        self.norms = None

    def sample(self):
        self.norms = np.random.normal(0.0, 1.0, self.num_paths * self.num_steps).reshape([self.num_steps, self.num_paths])
        num_times = 1 + self.num_steps
        process = np.empty([num_times, self.num_paths])
        process[0][:] = self.spot
        for idx_time in range(1, num_times):
            process[idx_time][:] = process[idx_time-1][:] + np.sqrt(self.nu * self.step_size) * self.norms[idx_time-1][:]
        return process


class GBMMonteCarlo(object): #pylint: disable=too-many-instance-attributes
    """
    Monte Carlo method for generating samples from the Heston stochastic volatility model .
    """

    def __init__(self, #pylint: disable=too-many-arguments
                 step_size=0.1,
                 num_steps=50,
                 num_paths=(4*2048),
                 spot=50.0,
                 drift=0.0,
                 sigma=0.2):
        """
        Member-wise initialisation.

        Args:
            step_size: The Monte Carlo step size in years.
            num_steps: The number of steps.
            num_paths: The number of sample paths.
            spot: The starting point of the process.
            drift: The drift of the process.
            sigma: The volatility.
        """
        self.spot = spot
        self.drift = drift
        self.sigma = sigma
        self.step_size = step_size
        self.num_steps = num_steps
        self.num_paths = num_paths
        self.time = np.linspace(step_size, num_steps * step_size, num_steps)
        self.time_axis = np.linspace(0.0, num_steps * step_size, 1 + num_steps)
        self.str_time_axis = [str(t) for t in self.time_axis]
        self.norms = None

    @staticmethod
    def process_step(log_value, drift, sigma, norm, time_step): # pylint: disable=invalid-name
        """
        Return the updated process value.

        Args:
            log_value: The current value of the process.
            nu: The current variance.
            norm: A standard random normal variate.
            time_step: The size of the times step.

        Returns:
            The updated process value.

        """
        nu = sigma * sigma
        return log_value + (drift - 0.5 * nu) * time_step + np.sqrt(nu * time_step) * norm

    def sample(self):
        """

        Sample the Heston model.

        Returns:
            Samples paths of the process and the variance.
        """

        norms = np.random.normal(0.0, 1.0, self.num_paths * self.num_steps)
        norms = norms.reshape([self.num_steps, self.num_paths])

        num_times = 1 + self.num_steps
        log_process = np.full([self.num_paths], np.log(self.spot))
        process = np.empty([num_times, self.num_paths])
        process[0][:] = self.spot

        for idx_time in range(1, num_times):
            log_process = self.process_step(log_process,
                                            self.drift,
                                            self.sigma,
                                            norms[idx_time-1],
                                            self.step_size)
            process[idx_time][:] = np.exp(log_process)

        return process
